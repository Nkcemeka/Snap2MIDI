# Imports
import torch
import torch.nn as nn
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import wandb
from snap2midi.train.hft.hft import *
from .hft_dataset import HFTDataset
import torch.optim as optim

# define some helper functions
def count_parameters(model):
    """
    Count the number of trainable parameters in a model.    
    Args:
        model (torch.nn.Module): The model to count parameters for. 
    Returns:
        int: The number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(m):
    """
        Initialize weights of the model using Xavier uniform initialization.

        Args:
            m (torch.nn.Module): The module to initialize weights for.
    """
    if hasattr(m, 'weight') and (m.weight.dim() > 1):
        nn.init.xavier_uniform_(m.weight.data)

def train_step(model, optimizer, loss_onset1, loss_offset1, loss_frames1, loss_velocity1, loss_onset2, \
               loss_offset2, loss_frames2, loss_velocity2, \
               weight_A, weight_B, dataloader, device, verbose_flag):
    """
        Perform a single training step on the model.
        
        Args:
            model (torch.nn.Module): The model to train.
            optimizer (torch.optim.Optimizer): The optimizer to use for training.
            loss_onset1 (torch.nn.Module): Loss function for onset detection (1st).
            loss_offset1 (torch.nn.Module): Loss function for offset detection (1st).
            loss_frames1 (torch.nn.Module): Loss function for frame classification (1st).
            loss_velocity1 (torch.nn.Module): Loss function for velocity prediction (1st).
            loss_onset2 (torch.nn.Module): Loss function for onset detection (2nd).
            loss_offset2 (torch.nn.Module): Loss function for offset detection (2nd).
            loss_frames2 (torch.nn.Module): Loss function for frame classification (2nd).
            loss_velocity2 (torch.nn.Module): Loss function for velocity prediction (2nd).
            weight_A (float): Weight for the 1st loss.
            weight_B (float): Weight for the 2nd loss.
            dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
            device (torch.device): Device to run the model on (CPU or GPU).
            verbose_flag (bool): Flag to control verbosity of the training process.

        Returns:
            dict: A dictionary containing the average losses for the training step.
    """
    model.train() # set the model to training mode
    loss_dict: dict = {'total_loss': 0.0, 
                       'loss_onset_1st': 0.0, 
                       'loss_offset_1st': 0.0, 
                       'loss_frames_1st': 0.0, 
                       'loss_velocity_1st': 0.0, 
                       'loss_onset_2nd': 0.0, 
                       'loss_offset_2nd': 0.0, 
                       'loss_frames_2nd': 0.0, 
                       'loss_velocity_2nd': 0.0}

    for i, (input_spec, label_onset, label_offset, label_frames,\
            label_velocity) in tqdm(enumerate(dataloader), desc="Training", disable=not verbose_flag):
        
        # set inputs and targets to device
        input_spec = input_spec.to(device, non_blocking=True)
        label_onset = label_onset.to(device, non_blocking=True)
        label_offset = label_offset.to(device, non_blocking=True)
        label_frames = label_frames.to(device, non_blocking=True)
        label_velocity = label_velocity.to(device, non_blocking=True)

        if verbose_flag is True:
            print(f"Batch {i+1}/{len(dataloader)}")
            print(f"Input shape: {input_spec.shape}")
            print(f"Label onset shape: {label_onset.shape}")
            print(f"Label offset shape: {label_offset.shape}")
            print(f"Label frames shape: {label_frames.shape}")
            print(f"Label velocity shape: {label_velocity.shape}")
        
        #FLZBG (forward pass, loss computation, backward pass, zero gradients, backward pass, gradient descent)
        # forward pass
        output_onset_1st, output_offset_1st, output_frames_1st, output_velocity_1st, attention, \
            output_onset_2nd, output_offset_2nd, output_frames_2nd, output_velocity_2nd = model(input_spec)
        
        # flatten the outputs and labels
        output_onset_1st = output_onset_1st.contiguous().view(-1)
        output_offset_1st = output_offset_1st.contiguous().view(-1)
        output_frames_1st = output_frames_1st.contiguous().view(-1)
        output_velocity_1st = output_velocity_1st.contiguous().view(-1, output_velocity_1st.shape[-1])
        output_onset_2nd = output_onset_2nd.contiguous().view(-1)
        output_offset_2nd = output_offset_2nd.contiguous().view(-1)
        output_frames_2nd = output_frames_2nd.contiguous().view(-1)
        output_velocity_2nd = output_velocity_2nd.contiguous().view(-1, output_velocity_2nd.shape[-1])
        label_onset = label_onset.contiguous().view(-1)
        label_offset = label_offset.contiguous().view(-1)
        label_frames = label_frames.contiguous().view(-1)
        label_velocity = label_velocity.contiguous().view(-1)

        # compute losses
        # 1st loss
        loss_onset_1st = loss_onset1(output_onset_1st, label_onset)
        loss_offset_1st = loss_offset1(output_offset_1st, label_offset)
        loss_frames_1st = loss_frames1(output_frames_1st, label_frames)
        loss_velocity_1st = loss_velocity1(output_velocity_1st, label_velocity)
        loss_1st = loss_onset_1st + loss_offset_1st + loss_frames_1st + loss_velocity_1st

        # 2nd loss
        loss_onset_2nd = loss_onset2(output_onset_2nd, label_onset)
        loss_offset_2nd = loss_offset2(output_offset_2nd, label_offset)
        loss_frames_2nd = loss_frames2(output_frames_2nd, label_frames)
        loss_velocity_2nd = loss_velocity2(output_velocity_2nd, label_velocity)
        loss_2nd = loss_onset_2nd + loss_offset_2nd + loss_frames_2nd + loss_velocity_2nd

        # total loss
        loss = weight_A * loss_1st + weight_B * loss_2nd
        loss_dict['total_loss'] += loss.item()
        loss_dict['loss_onset_1st'] += loss_onset_1st.item()
        loss_dict['loss_offset_1st'] += loss_offset_1st.item()
        loss_dict['loss_frames_1st'] += loss_frames_1st.item()
        loss_dict['loss_velocity_1st'] += loss_velocity_1st.item()
        loss_dict['loss_onset_2nd'] += loss_onset_2nd.item()
        loss_dict['loss_offset_2nd'] += loss_offset_2nd.item()
        loss_dict['loss_frames_2nd'] += loss_frames_2nd.item()
        loss_dict['loss_velocity_2nd'] += loss_velocity_2nd.item()

        # zero gradients
        optimizer.zero_grad()

        # backward pass
        loss.backward()

        # gradient descent
        optimizer.step()
    
    # average the losses
    for key in loss_dict:
        loss_dict[key] /= len(dataloader)

    return loss_dict

def validate_step(model, loss_onset1, loss_offset1, loss_frames1, loss_velocity1, \
                  loss_onset2, loss_offset2, loss_frames2, loss_velocity2, \
                  weight_A, weight_B, dataloader, device, verbose_flag, prefix="valid"):    
    """
        Perform a single validation step on the model.  

        Args:
            model (torch.nn.Module): The model to validate.
            loss_onset1 (torch.nn.Module): Loss function for onset detection (1st).
            loss_offset1 (torch.nn.Module): Loss function for offset detection (1st).
            loss_frames1 (torch.nn.Module): Loss function for frame classification (1st).
            loss_velocity1 (torch.nn.Module): Loss function for velocity prediction (1st).
            loss_onset2 (torch.nn.Module): Loss function for onset detection (2nd).
            loss_offset2 (torch.nn.Module): Loss function for offset detection (2nd).
            loss_frames2 (torch.nn.Module): Loss function for frame classification (2nd).
            loss_velocity2 (torch.nn.Module): Loss function for velocity prediction (2nd).
            weight_A (float): Weight for the 1st loss.
            weight_B (float): Weight for the 2nd loss.
            dataloader (torch.utils.data.DataLoader): DataLoader for the validation data.
            device (torch.device): Device to run the model on (CPU or GPU).
            verbose_flag (bool): Flag to control verbosity of the validation process.

        Returns:
            dict: A dictionary containing the average losses for the validation step.
    """
    model.eval()
    loss_dict = {
        f'{prefix}_total_loss': 0,
        f'{prefix}_loss_onset_1st': 0,
        f'{prefix}_loss_offset_1st': 0,
        f'{prefix}_loss_frames_1st': 0,
        f'{prefix}_loss_velocity_1st': 0,
        f'{prefix}_loss_onset_2nd': 0,
        f'{prefix}_loss_offset_2nd': 0,
        f'{prefix}_loss_frames_2nd': 0,
        f'{prefix}_loss_velocity_2nd': 0,
    }

    with torch.no_grad():
        for i, (input_spec, label_onset, label_offset, label_frames,\
            label_velocity) in tqdm(enumerate(dataloader), desc="Validation", disable=not verbose_flag):
            # Move data to the appropriate device
            input_spec = input_spec.to(device, non_blocking=True)
            label_onset = label_onset.to(device, non_blocking=True)
            label_offset = label_offset.to(device, non_blocking=True)
            label_frames = label_frames.to(device, non_blocking=True)
            label_velocity = label_velocity.to(device, non_blocking=True)

            # Forward pass
            output_onset_1st, output_offset_1st, output_frames_1st, output_velocity_1st, attention, \
            output_onset_2nd, output_offset_2nd, output_frames_2nd, output_velocity_2nd = model(input_spec)

            # flatten the outputs and labels
            output_onset_1st = output_onset_1st.contiguous().view(-1)
            output_offset_1st = output_offset_1st.contiguous().view(-1)
            output_frames_1st = output_frames_1st.contiguous().view(-1)
            output_velocity_1st = output_velocity_1st.contiguous().view(-1, output_velocity_1st.shape[-1])
            output_onset_2nd = output_onset_2nd.contiguous().view(-1)
            output_offset_2nd = output_offset_2nd.contiguous().view(-1)
            output_frames_2nd = output_frames_2nd.contiguous().view(-1)
            output_velocity_2nd = output_velocity_2nd.contiguous().view(-1, output_velocity_2nd.shape[-1])
            label_onset = label_onset.contiguous().view(-1)
            label_offset = label_offset.contiguous().view(-1)
            label_frames = label_frames.contiguous().view(-1)
            label_velocity = label_velocity.contiguous().view(-1)

            # compute losses
            # 1st loss
            loss_onset_1st = loss_onset1(output_onset_1st, label_onset)
            loss_offset_1st = loss_offset1(output_offset_1st, label_offset)
            loss_frames_1st = loss_frames1(output_frames_1st, label_frames)
            loss_velocity_1st = loss_velocity1(output_velocity_1st, label_velocity)
            loss_1st = loss_onset_1st + loss_offset_1st + loss_frames_1st + loss_velocity_1st

            # 2nd loss
            loss_onset_2nd = loss_onset2(output_onset_2nd, label_onset)
            loss_offset_2nd = loss_offset2(output_offset_2nd, label_offset)
            loss_frames_2nd = loss_frames2(output_frames_2nd, label_frames)
            loss_velocity_2nd = loss_velocity2(output_velocity_2nd, label_velocity)
            loss_2nd = loss_onset_2nd + loss_offset_2nd + loss_frames_2nd + loss_velocity_2nd

            # total loss
            loss = weight_A * loss_1st + weight_B * loss_2nd
            loss_dict[f'{prefix}_total_loss'] += loss.item()
            loss_dict[f'{prefix}_loss_onset_1st'] += loss_onset_1st.item()
            loss_dict[f'{prefix}_loss_offset_1st'] += loss_offset_1st.item()
            loss_dict[f'{prefix}_loss_frames_1st'] += loss_frames_1st.item()
            loss_dict[f'{prefix}_loss_velocity_1st'] += loss_velocity_1st.item()
            loss_dict[f'{prefix}_loss_onset_2nd'] += loss_onset_2nd.item()
            loss_dict[f'{prefix}_loss_offset_2nd'] += loss_offset_2nd.item()
            loss_dict[f'{prefix}_loss_frames_2nd'] += loss_frames_2nd.item()
            loss_dict[f'{prefix}_loss_velocity_2nd'] += loss_velocity_2nd.item()

    # Average the losses
    for key in loss_dict:
        loss_dict[key] /= len(dataloader)

    return loss_dict

def main(config: dict):
    wandb.init(project=config['project_name'], config=config)

    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    torch.backends.cudnn.deterministic = True
    
    # Create HFT dataset instances for train, validation, and test sets
    train_dataset = HFTDataset("data/hft", config, split="train")
    valid_dataset = HFTDataset("data/hft", config, split="val")
    test_dataset = HFTDataset("data/hft", config, split="test")

    # Create DataLoaders for each dataset
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # set losses
    loss_onset1 = nn.BCELoss()
    loss_offset1 = nn.BCELoss()
    loss_frames = nn.BCELoss()
    loss_velocity1 = nn.CrossEntropyLoss()
    loss_onset2 = nn.BCELoss()
    loss_offset2 = nn.BCELoss()
    loss_frames2 = nn.BCELoss()
    loss_velocity2 = nn.CrossEntropyLoss()

    # Load/initialize the model
    encoder = HFTEncoder(n_margin=config['margin_b'],
                         n_frame=config['num_frame'],
                         n_bin=config['n_bins'],
                         cnn_channel=config["cnn_channel"],
                         cnn_kernel=config["cnn_kernel"],
                         d=config["d"],
                         n_layers=config["enc_layer"],
                         num_heads=config["enc_head"],
                         pff_dim=config["pff_dim"],
                         dropout=config["dropout"],
                         device="cuda" if torch.cuda.is_available() else "cpu")

    decoder = HFTDecoder(n_frame=config['num_frame'],
                         n_bin=config['n_bins'],
                         n_note=config['num_note'],
                         n_velocity=config['num_velocity'],
                         d=config["d"],
                         n_layers=config["dec_layer"],
                         num_heads=config["dec_head"],
                         pff_dim=config["pff_dim"],
                         dropout=config["dropout"],
                         device="cuda" if torch.cuda.is_available() else "cpu")
    
    model = HFT(encoder=encoder, decoder=decoder)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.apply(init_weights)  # Initialize weights
    print(f"Number of trainable parameters: {count_parameters(model)}")

    
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    last_epoch = -1
    if config["resume"] == 1:
        # Load the model from the checkpoint
        checkpoint = torch.load(config["resume_path"], map_location="cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        torch.set_rng_state(checkpoint['random']['torch'])
        torch.random.set_rng_state(checkpoint['random']['torch_random'])
        torch.cuda.set_rng_state(checkpoint['random']['cuda'])
        torch.cuda.torch.cuda.set_rng_state_all(checkpoint['random']['cuda_all'])
        last_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {last_epoch + 1}")
    
    # Create runs directory if it does not exist
    Path(config["save_dir"]).mkdir(exist_ok=True, parents=True)
    last_epoch = max(1, last_epoch)  # We use 1 since we save with epoch + 1
    best_loss = float('inf')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for epoch in tqdm(range(last_epoch-1, config["epochs"]), desc="Training Epochs"):
        train_loss_dict = train_step(model, optimizer, loss_onset1, loss_offset1, loss_frames, loss_velocity1, \
                                     loss_onset2, loss_offset2, loss_frames2, loss_velocity2, \
                                     config["weight_A"], config["weight_B"], train_dataloader, \
                                     device, config["verbose"])
        
        valid_loss_dict = validate_step(model, loss_onset1, loss_offset1, loss_frames, loss_velocity1, \
                                        loss_onset2, loss_offset2, loss_frames2, loss_velocity2, \
                                        config["weight_A"], config["weight_B"], valid_dataloader, \
                                        device, config["verbose"], prefix="valid")
        
        # Step the scheduler
        scheduler.step(valid_loss_dict['valid_total_loss'])
        
        # Log results to wandb
        results = {}

        for key, value in train_loss_dict.items():
            results[f'{key}'] = value

        for key, value in valid_loss_dict.items():
            results[f'{key}'] = value
        
        wandb.log(results)
        if valid_loss_dict['valid_total_loss'] < best_loss:
            best_loss = valid_loss_dict['valid_total_loss']
            # Save the model checkpoint
            checkpoint_path = Path(config["save_dir"]) / f"checkpoint_{epoch + 1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_total_loss': train_loss_dict['total_loss'],
                'valid_total_loss': valid_loss_dict['valid_total_loss'],
                'random': {
                    'torch': torch.get_rng_state(),
                    'torch_random': torch.random.get_rng_state(),
                    'cuda': torch.cuda.get_rng_state() ,
                    'cuda_all' : torch.cuda.get_rng_state_all()
                }
            }, checkpoint_path)
            print(f"Saved new best model to {checkpoint_path}")
        
        print(f"Epoch {epoch + 1}/{config['epochs']}: "
              f"Train Loss: {train_loss_dict['total_loss']:.4f}, "
              f"Valid Loss: {valid_loss_dict['valid_total_loss']:.4f}")

    print(f"Evaluating the model on the test set...")
    best_model = HFT(encoder=encoder, decoder=decoder)
    best_model = best_model.to(device)
    checkpoint_path = config["save_dir"]
    best_checkpoints = list(Path(checkpoint_path).glob("checkpoint_*.pt"))
    best_checkpoints.sort(key=lambda x: int(x.stem.split("_")[1]))
    best_checkpoint = str(best_checkpoints[-1])
    best_model.load_state_dict(torch.load(best_checkpoint, weights_only=True)["model_state_dict"])

    test_loss_dict = validate_step(best_model, loss_onset1, loss_offset1, loss_frames, loss_velocity1, \
                                  loss_onset2, loss_offset2, loss_frames2, loss_velocity2, \
                                  config["weight_A"], config["weight_B"], test_dataloader, \
                                  device, config["verbose"], prefix="test")
    print(f"Test Loss: {test_loss_dict['test_total_loss']:.4f}")
    results_test = {}
    for key, value in test_loss_dict.items():
        results_test[f'{key}'] = value
    wandb.log(results_test)

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train HFT model")
    parser.add_argument('--config_path', type=str, required=True, help='Path to the training config file')
    args = parser.parse_args()

    # Load configuration
    with open(args.config_path, 'r') as f:
        train_config = json.load(f)

    # Train model
    # initialize wandb
    wandb.init(project=train_config['project_name'], config=train_config)
    main(train_config)
    wandb.finish()
