# Imports
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
import h5py
from torch.optim.lr_scheduler import StepLR
from snap2midi.utils.eval_mir import frame_metrics
from .kong_dataset import KongDataset, Sampler, EvalSampler, collate_fn
from .kong import KongModel
from typing import Any

# define some helper function for counting the parameters in a model
def count_parameters(model):
    """
    Count the number of trainable parameters in a model.    
    Args:
        model (torch.nn.Module): The model to count parameters for. 
    Returns:
        int: The number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def bce_mask(output, target, mask):
    """
        Binary crossentropy with masking.

        Args:
            output (torch.Tensor): Model's output
            target (torch.Tensor): Target tensor
            mask (torch.Tensor): Mask
        
        Returns:
            BCE Loss   
    """
    eps = 1e-7
    output = torch.clamp(output, eps, 1. - eps)
    matrix = - target * torch.log(output) - (1. - target) * torch.log(1. - output)
    return torch.sum(matrix * mask) / torch.sum(mask)

def regress_bce(output_dict, target_dict):
    """
        Calculate the regression loss using binary cross-entropy.

        Args:
        ------
            output_dict (dict): Model output dictionary containing the predicted values.
            target_dict (dict): Target dictionary containing the ground truth values.

        Returns:
        -------
            loss_dict (dict): Dictionary containing the individual losses and total loss.
    """
    onset_loss = bce_mask(output_dict['reg_onset_roll'], \
                     target_dict['label_reg_onsets'], target_dict['mask_roll'])
    offset_loss = bce_mask(output_dict['reg_offset_roll'], \
                      target_dict['label_reg_offsets'], target_dict['mask_roll'])
    frame_loss = bce_mask(output_dict['frame_roll'], \
                     target_dict['label_frames'], target_dict['mask_roll'])
    velocity_loss = bce_mask(output_dict['velocity_roll'], \
                        target_dict['label_velocities'] / 128, target_dict['label_onsets'])
    total_loss = onset_loss + offset_loss + frame_loss + velocity_loss

    loss_dict = {
        'onset_loss': onset_loss,
        'offset_loss': offset_loss,
        'frame_loss': frame_loss,
        'total_loss': total_loss,
        'velocity_loss': velocity_loss
    }

    return loss_dict


def frame_metrics_batch(output_dict: dict, target_dict: dict, config: dict) -> dict:
    """
        Calculate frame metrics.

        Args:
        -----
            output_dict (dict): Model ouput dictionary
            target_dict (dict): Target dictionary
            config (dict): Config dictionary

        Returns:
        ---------
            metrics (dict): Dictionary of frame metrics

    """

    # Initialize metrics dictionary
    metrics: dict = {}
    frame_roll = output_dict["frame_roll"].cpu().detach().numpy()
    pred = (frame_roll >= config["frame_threshold"])
    target = target_dict["label_frames"].cpu().detach().numpy()

    scores = frame_metrics(pred.flatten(), target.flatten())
    for key in scores:
        metrics[key] = np.round(scores[key], 2).item()

    return metrics


@torch.no_grad()
def evaluate(model: Any, dataloader: Any, device: str, config=None):
    
    """
        Evaluate the model on the given dataloader.

        Args:
        -----
            model (torch.nn.Module): The model to evaluate.
            dataloader (torch.utils.data.DataLoader): The dataloader to use for evaluation.
            device (str): The device to use for evaluation.
            config (dict): Configuration dictionary.

    Returns:
    --------
        loss_dict (dict): Dictionary containing the loss values.
        metrics_frames (dict): Dictionary containing the frame metrics.
        metrics_note (dict): Dictionary containing the note metrics.
    """
    loss_fn = regress_bce

    if not config:
        raise ValueError("Config not passed to evaluate function!")

    # Set the model to evaluation mode
    model.eval()
    loss_dict: dict = {f'valid_total_loss': 0, f'valid_onset_loss': 0, \
            f'valid_offset_loss': 0, f'valid_frame_loss': 0, \
            f'valid_velocity_loss': 0}
    num_samples = 0

    # Initialize metrics_frames dictionary
    metrics_frames: dict = {}
    
    for i, (batch_dict) in enumerate(dataloader):    
        # move everything to device
        for key in batch_dict.keys():
            batch_dict[key] = batch_dict[key].to(device)

        # Forward pass
        output_dict = model(batch_dict["audio"])

        # calculate the loss
        loss = loss_fn(output_dict, batch_dict)

        # Update loss dictionary
        loss_dict[f'valid_total_loss'] += loss['total_loss'].item()
        loss_dict[f'valid_onset_loss'] += loss["onset_loss"].item()
        loss_dict[f'valid_offset_loss'] += loss["offset_loss"].item()
        loss_dict[f'valid_frame_loss'] += loss["frame_loss"].item()
        loss_dict[f'valid_velocity_loss'] += loss["velocity_loss"].item()

        num_samples += batch_dict["audio"].shape[0]
        
        # Calculate frame metrics
        frames_scores = frame_metrics_batch(output_dict, batch_dict, config)
        for key in frames_scores:
            if key not in metrics_frames:
                metrics_frames[key] = []
            metrics_frames[key].append(frames_scores[key])

    # Average the loss
    for key in loss_dict:
        loss_dict[key] = loss_dict[key] / num_samples
        loss_dict[key] = round(loss_dict[key], 4)
    
    # Average the metrics across the batch
    for key in metrics_frames:
        metrics_frames[key] = sum(metrics_frames[key])/len(metrics_frames[key])
        metrics_frames[key] = round(metrics_frames[key], 2)

    return loss_dict, metrics_frames

def main(config):
    """
        Main function to train the Kong model.
        
        Args:
        -----
            config (dict): Configuration dictionary containing paths, hyperparameters, etc.
    """

    # Initialize wandb
    wandb.init(project=config["project_name"], \
            config=config)

    # Load the model
    with h5py.File("data/kong/extraction_config.h5", "r") as hf:
        extraction_config = dict(hf.attrs)

        # convert byte strings to normal strings and np.ints to int etc
        for key in extraction_config.keys():
            if isinstance(extraction_config[key], bytes):
                extraction_config[key] = extraction_config[key].decode("utf-8")
            elif isinstance(extraction_config[key], np.generic):
                extraction_config[key] = extraction_config[key].item()

    model = KongModel(extraction_config, config["momentum"], cmp=config["cmp"], \
                      factors=config["factors"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Number of parameters: {count_parameters(model)}")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], amsgrad=True)
    scheduler = StepLR(optimizer, step_size=config['learning_rate_decay_steps'], \
                       gamma=config['learning_rate_decay_rate'])
    
    # Create datasets 
    extend_pedal = extraction_config["extend_pedal"]
    train_dataset = KongDataset("data/kong/train/", extend_pedal=extend_pedal)
    valid_dataset = KongDataset("data/kong/val/", extend_pedal=extend_pedal)

    # Sampler for training
    train_sampler = Sampler("data/kong/train/", split="train", batch_size=config["batch_size"])
    valid_sampler = EvalSampler("data/kong/val/", split="val", batch_size=config["batch_size"])

    if config['resume']:
        checkpoint = torch.load(config['resume_path'], weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        train_sampler.load_state_dict(checkpoint['sampler'])
        iter = checkpoint['iter']
    else:
        iter = 0

    # Create dataloaders for each dataset
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, \
                                  pin_memory=True, collate_fn=collate_fn) 
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, \
                        pin_memory=True, batch_sampler=valid_sampler, collate_fn=collate_fn)

    # Create runs directory if it does not exist
    Path(config["save_dir"]).mkdir(exist_ok=True, parents=True)

    best_loss = float('inf')
    loss_fn = regress_bce
    loss_dict: dict = {'train_total_loss': 0, 'train_onset_loss': 0, 'train_offset_loss': 0, \
            'train_frame_loss': 0, 'train_velocity_loss': 0}

    for i, batch_dict in tqdm(enumerate(train_dataloader)):
        # save checkpoint every 20000 iterations
        if iter % 20000 == 0:
            torch.save({
                'iter': iter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'sampler': train_sampler.state_dict(),
            }, config["save_dir"] + f"/checkpoint_{iter}.pt")

        # log loss every 5000 iterations
        if iter % 5000 == 0 and iter > 0:
            # Average the loss
            for key in loss_dict:
                loss_dict[key] = loss_dict[key] / 5000
                loss_dict[key] = round(loss_dict[key], 4)
            
            # Get validation loss and metrics
            val_loss_dict, val_metrics_frames = evaluate(model, valid_dataloader, device, config)
            results = {**loss_dict, **val_loss_dict, **val_metrics_frames}         
            print(f"Iteration {iter}: {results}")
            wandb.log(results)

            # reset the loss dictionary
            for key in loss_dict:
                loss_dict[key] = 0.

        if iter % 200000 == 0 and iter > 0:
            wandb.finish()
            return  # stop training after 200000 iterations

        model.train() # set model to training mode
        # move everything to device
        for key in batch_dict.keys():
            batch_dict[key] = batch_dict[key].to(device)

        # Forward pass
        output_dict = model(batch_dict["audio"])
    
        # calculate the loss
        loss = loss_fn(output_dict, batch_dict)

        # Update loss dictionary 
        loss_dict['train_total_loss'] += loss['total_loss'].item()
        loss_dict['train_onset_loss'] += loss['onset_loss'].item()
        loss_dict['train_offset_loss'] += loss['offset_loss'].item()
        loss_dict['train_frame_loss'] += loss['frame_loss'].item()
        loss_dict['train_velocity_loss'] += loss['velocity_loss'].item()

        # Zero gradients
        optimizer.zero_grad()

        # Backprop
        loss['total_loss'].backward()

        # Gradient Descent
        optimizer.step()

        scheduler.step()
        iter += 1
