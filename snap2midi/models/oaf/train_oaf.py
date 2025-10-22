# Imports
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
import wandb
import json
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ExponentialLR
from snap2midi.utils.eval_mir import transcription_metrics, multipitch_metrics, note_extract, notes_to_frames
from .dataset_oaf import OAFDataset
from .oaf import OnsetsAndFrames
from typing import Any
from collections import defaultdict
import torch.nn.functional as F

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

# Create some custom loss functions
def loss_velocity(velocity_pred: torch.Tensor, velocity_label: torch.Tensor, \
                  onset_label: torch.Tensor) -> torch.Tensor:
        denominator = onset_label.sum()
        if denominator.item() == 0:
            return denominator
        else:
            return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator
        
def weighted_bce_loss(output, target, weights):
    loss_matrix = F.binary_cross_entropy_with_logits(output, target, reduction='none')
    weighted_loss = loss_matrix * weights
    return weighted_loss.mean()

def transcription_metrics_batch(pred: tuple, gt: tuple, \
                                threshold: float, frame_rate: float, pitch_offset: int = 21) -> dict:
    """
    Calculate transcription metrics for a batch of predictions and ground truth.
    F1-scores should be ignored and recalculated based on the precision and recall
    values in the returned dict. This is because the F1-score is not a metric that 
    can be averaged across batches. The F1-score is a harmonic mean of precision and 
    recall, and it is not meaningful to average the F1-scores across batches. Instead, 
    we can calculate the final F1-score based on the avg. precision and recall values.
    
    Args:
        pred (Tuple): Predicted piano rolls.
        gt (Tuple): Ground truth piano rolls.
        threshold (float): Threshold for binarizing predictions.
        frame_rate (int): Frames per second.
        pitch_offset (int): Offset for the pitch values, default is 21.

    Returns:
        metrics (dict): Dictionary containing the calculated metrics.
    """
    # Initialize metrics dictionary
    metrics = defaultdict(list)
    
    on_preds, frame_preds, vel_preds = pred
    y_onsets, y_frames, y_velocities = gt
        
    for each in range(y_onsets.shape[0]):
        note_preds, int_preds, _ = note_extract(on_preds[each], frame_preds[each], vel_preds[each], \
                onset_thresh=threshold, frame_thresh=threshold)
        
        note_gt, int_gt, _ = note_extract(y_onsets[each], y_frames[each], y_velocities[each], \
                onset_thresh=threshold, frame_thresh=threshold)
        
        # Ensure that the pitches are within the range of 21-108
        note_gt += pitch_offset
        note_preds += pitch_offset

        scores = transcription_metrics(note_preds, int_preds, note_gt, int_gt, frame_rate=frame_rate)
        
        if scores is None:
            continue

        for key in ["Precision", "Recall", "Precision_no_offset", "Recall_no_offset"]:
            if key in metrics:
                metrics[key].append(scores[key])
            else:
                metrics[key] = [scores[key]]
                

    # Average the metrics across the batch
    for key in metrics:
        metrics[key] = np.mean(metrics[key]).item()
        metrics[key] = np.round(metrics[key], 4).item()
    
    return metrics

def multipitch_metrics_batch(pred: tuple, gt: tuple, \
                             threshold: float, frame_rate: float, pitch_offset: int = 21) -> dict:
    """
    Calculate multipitch metrics for a batch of predictions and ground truth.
    Args:
        pred (Tuple): Predicted piano rolls.
        gt (Tuple): Ground truth piano rolls.
        threshold (float): Threshold for binarizing predictions.
        frame_rate (float): Frames per second.

    Returns:
        metrics (dict): Dictionary containing the calculated metrics.
    """

    on_preds, frame_preds, vel_preds = pred
    y_onsets, y_frames, y_velocities = gt

    # Initialize metrics dictionary
    metrics = defaultdict(list)
    for each in range(y_onsets.shape[0]):
        note_preds, int_preds, _ = note_extract(on_preds[each], frame_preds[each], vel_preds[each], \
                onset_thresh=threshold, frame_thresh=threshold)
        
        note_gt, int_gt, _ = note_extract(y_onsets[each], y_frames[each], y_velocities[each], \
                onset_thresh=threshold, frame_thresh=threshold)

        # Convert notes to frames
        preds_roll = notes_to_frames(note_preds, int_preds, y_onsets[0].shape)
        y_roll = notes_to_frames(note_gt, int_gt, y_onsets[0].shape)

        scores = multipitch_metrics(y_roll, preds_roll, frame_rate=frame_rate, pitch_offset=pitch_offset)
        if scores is None:
            continue

        for key in ["Precision", "Recall"]:
            if key in metrics:
                metrics[key].append(scores[key])
            else:
                metrics[key] = [scores[key]]
    
    # Average the metrics across the batch
    for key in metrics:
        metrics[key] = np.mean(metrics[key]).item()
        metrics[key] = np.round(metrics[key], 4).item()
    
    return metrics

def save(audio: torch.Tensor, y: tuple, preds: tuple, \
         threshold: float, save_dir: str, batch_index: int) -> None:
    """
    Save audio and piano roll predictions to .npz files.

    Args:
        audio (torch.Tensor): Batch of audio data.
        y (tuple): Ground truth piano rolls.
        preds (tuple): Predicted piano rolls.
        threshold (float): Threshold for binarizing predictions.
        save_dir (str): Directory to save the results.
        batch_index (int): Index of the current batch.
    """
    on_preds, frame_preds, vel_preds = preds
    y_onsets, y_frames, y_velocities = y

    for each in range(audio.shape[0]):
        audio_arr = audio[each].squeeze(0).detach().cpu().numpy()
        note_preds, int_preds, vel_pred = note_extract(on_preds[each], frame_preds[each], vel_preds[each], \
                onset_thresh=threshold, frame_thresh=threshold)
        
        note_gt, int_gt, vel_gt = note_extract(y_onsets[each], y_frames[each], y_velocities[each], \
                onset_thresh=threshold, frame_thresh=threshold)

        # Convert notes to frames
        preds_roll = notes_to_frames(note_preds, int_preds, y_onsets[0].shape)
        y_roll = notes_to_frames(note_gt, int_gt, y_onsets[0].shape)

        Path(save_dir + f"/results/").mkdir(parents=True, exist_ok=True)
        result_dict: dict[str, Any] = {'audio': audio_arr, 'original_roll': y_roll, 'pred_roll': preds_roll}
        np.savez(save_dir + f"/results/{batch_index}_{each}.npz", **result_dict)

@torch.no_grad()
def evaluate(model: Any, dataloader: Any, device: str, \
        loss_fn: Any, frame_rate: float | int, threshold:float=0.3, \
        save_dir: str | None=None, pitch_offset: int = 21):
    model.eval()
    loss_dict: dict = {f'test_total_loss': 0, f'test_onset_loss': 0, \
     f'test_frame_loss': 0, f'test_velocity_loss': 0}
    num_samples = 0

    # Initialize metrics_frames dictionary
    metrics_frames = defaultdict(list)

    # Initialize metrics_note dictionary
    metrics_note = defaultdict(list)
    
    for i, (x, y_frame, y_onset, y_velocity,\
             label_weights, audio) in enumerate(dataloader):
        x = x.to(device)
        y_frame = y_frame.to(device)
        y_onset = y_onset.to(device)
        y_velocity = y_velocity.to(device).float()
        label_weights = label_weights.to(device).float()

        # Forward pass
        on_preds, _, frame_preds, vel_preds = model(x)

        # Loss
        # onset_loss + offset_loss + frame_loss + velocity_loss
        onset_loss = loss_fn(on_preds, y_onset)
        frame_loss = weighted_bce_loss(frame_preds, y_frame, label_weights)
        velocity_loss = loss_velocity(vel_preds, y_velocity, y_onset)
        loss = onset_loss + frame_loss + velocity_loss

        # Update loss dictionary
        loss_dict[f'test_total_loss'] += loss.item()
        loss_dict[f'test_onset_loss'] += onset_loss.item()
        loss_dict[f'test_frame_loss'] += frame_loss.item()
        loss_dict[f'test_velocity_loss'] += velocity_loss.item()

        num_samples += x.shape[0]

        preds = (torch.sigmoid(on_preds), torch.sigmoid(frame_preds),\
                  vel_preds)
        y = (y_onset, y_frame, y_velocity)

        
        # Calculate transcription metrics
        notes_scores = transcription_metrics_batch(preds, y, threshold, \
                frame_rate)
        
        if notes_scores is None:
            continue
        
        for key in notes_scores:
            if key in metrics_note:
                metrics_note[key].append(notes_scores[key])
            else:
                metrics_note[key] = [notes_scores[key]]

        # Calculate multipitch metrics
        frames_scores = multipitch_metrics_batch(preds, y, threshold, \
                frame_rate)
        for key in frames_scores:
            if key in metrics_frames:
                metrics_frames[key].append(frames_scores[key])
            else:
                metrics_frames[key] = [frames_scores[key]]
        
        # save audio and piano roll
        if save_dir is not None:
            save(audio, y, preds, threshold, save_dir, i)
    
    # Average the metrics across the batch
    for key in metrics_frames:
        metrics_frames[key] = sum(metrics_frames[key])/len(metrics_frames[key])
        metrics_frames[key] = round(metrics_frames[key], 4)
    for key in metrics_note:
        metrics_note[key] = sum(metrics_note[key])/len(metrics_note[key])
        metrics_note[key] = round(metrics_note[key], 4)

    # Average the loss
    for key in loss_dict:
        loss_dict[key] = loss_dict[key] / num_samples
        loss_dict[key] = round(loss_dict[key], 4)
    return loss_dict, metrics_frames, metrics_note


def train_step(model: Any, dataloader: Any, device: str, \
    loss_fn: Any, optimizer: Any, scheduler: Any, clip_gradient_norm: float=3.0):
    model.train()
    loss_dict: dict = {'train_total_loss': 0, 'train_onset_loss': 0, \
            'train_frame_loss': 0, 'train_velocity_loss': 0}
    num_samples = 0
    num_iter_batch = 0

    for i, (x, y_frame, y_onset, y_velocity,\
             label_weights, audio) in enumerate(dataloader):

        x = x.to(device)
        y_frame = y_frame.to(device)
        y_onset = y_onset.to(device)
        y_velocity = y_velocity.to(device).float()
        label_weights = label_weights.to(device).float()

        # Forward pass
        on_preds, _, frame_preds, vel_preds = model(x)

        # Loss
        onset_loss = loss_fn(on_preds, y_onset)
        frame_loss = weighted_bce_loss(frame_preds, y_frame, label_weights)
        velocity_loss = loss_velocity(vel_preds, y_velocity, y_onset)
        loss = onset_loss + frame_loss + velocity_loss

        # Update loss dictionary
        loss_dict['train_total_loss'] += loss.item()
        loss_dict['train_onset_loss'] += onset_loss.item()
        loss_dict['train_frame_loss'] += frame_loss.item()
        loss_dict['train_velocity_loss'] += velocity_loss.item()

        num_samples += x.shape[0]
        num_iter_batch += 1

        # Zero gradients
        optimizer.zero_grad()

        # Backprop
        loss.backward()

        # Clip gradients if necessary
        if clip_gradient_norm:
            clip_grad_norm_(model.parameters(), clip_gradient_norm)

        # Gradient Descent
        optimizer.step()

        # Update the scheduler
        global last_iter, decay_rate_steps, save_dir
        if (last_iter+1) % decay_rate_steps == 0:
            # We have to do this manually for ExponentialLR
            scheduler.step()

        if (last_iter+1) % 5000 == 0:
            # Save a checkpoint every 5000 iterations
            # We do this because there is no validation set according to the paper
            torch.save({
                'iter': last_iter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, save_dir + f"/checkpoint_{last_iter+1}.pt")
        
        last_iter += 1

    # Average the loss
    for key in loss_dict:
        loss_dict[key] = loss_dict[key] / num_samples
        loss_dict[key] = round(loss_dict[key], 4)
    return loss_dict, num_iter_batch

def main(config):
    # Initialize wandb
    wandb.init(project=config["project_name"], \
            config=config)
    
    # Create datasets 
    train_dataset = OAFDataset(["./data/oaf/train"])
    test_dataset = OAFDataset(["./data/oaf/test"])
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of testing samples: {len(test_dataset)}")


    # Create dataloaders for each dataset
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], \
            shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, \
            shuffle = False)

    # Load the model
    model = OnsetsAndFrames(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.initialize_weights()
    print(f"Number of trainable parameters: {count_parameters(model)}\n")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = ExponentialLR(optimizer, gamma=config['learning_rate_decay_rate'])
    
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # set this to be global variables so that we can access them in the training loop
    global last_iter, decay_rate_steps, save_dir
    last_iter = -1
    decay_rate_steps = config['learning_rate_decay_steps']
    save_dir = config["save_dir"]

    if config['resume']:
        checkpoint = torch.load(config['resume_path'], weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        last_iter = checkpoint['iter']

    # Create runs directory if it does not exist
    Path(config["save_dir"]).mkdir(exist_ok=True, parents=True)

    last_iter = max(0, last_iter)

    best_loss = float('inf')
    total_iterations = config["iterations"] - last_iter
    pbar = tqdm(total=total_iterations, initial=last_iter)
    
    while (last_iter) < (total_iterations):

        train_loss_dict, num_iter_batch = train_step(model, train_dataloader, device, \
                loss_fn, optimizer, scheduler, clip_gradient_norm=config["clip_gradient_norm"])   

        # update last_iter
        iteration = last_iter

        test_loss_dict, frame_metrics, note_metrics = evaluate(model, test_dataloader, device, loss_fn, \
                              config["frame_rate"], threshold=config["threshold"], pitch_offset=config["pitch_offset"]) 
        
        # Log frame and note metrics to wandb
        results = {}

        # Add frame metrics to results
        for key in frame_metrics:
            results[f"frame_{key}"] = frame_metrics[key]

        # Add note metrics to results
        for key in note_metrics:
            results[f"note_{key}"] = note_metrics[key]
        
        # Log results to wandb
        results = results | train_loss_dict | test_loss_dict
        wandb.log(results)

        print(f"Iteration {iteration+1}/{config['iterations']}, Train Loss: {train_loss_dict['train_total_loss']:.4f}, \
                Test Loss: {test_loss_dict['test_total_loss']:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        pbar.update(num_iter_batch)
    
    pbar.close()
    wandb.finish()

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path")
    args = parser.parse_args()

    # load JSON file
    with open(args.config_path, 'r') as filename:
        content = filename.read()
    
    # parse JSON file
    config = json.loads(content)

    # Initialize wandb
    wandb.init(project=config["project_name"], \
            config=config)
    main(config)
    wandb.finish()
