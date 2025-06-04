"""
    File: train_onsets_and_frames.py
    Author: Chukwuemeka L. Nkama
    Date: 4/11/2025
    Description: Training script for Onsets and Frames model!
"""

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
from torch.optim.lr_scheduler import StepLR
from snap2midi.utils.eval_mir import transcription_metrics, multipitch_metrics, note_extract, notes_to_frames, notes_to_frames_vels
from .kong_dataset import KongDataset
from .kong import KongModel
from typing import Any
from snap2midi.train_scripts.losses import regress_bce
from snap2midi.utils.eval_mir import output_dict_to_events
import mir_eval

def transcription_metrics_batch(output_dict: dict, target_dict: dict, \
                                config: dict, offset_ratio: float|None=None) -> dict:
    """
    Calculate transcription metrics for a batch of predictions and ground truth.
    """
    # Initialize metrics dictionary
    if offset_ratio is not None:
        metrics: dict = {
            "Precision": [],
            "Recall": []}
    else:
        metrics = {
            "Precision_no_offset": [],
            "Recall_no_offset": []}
    
    keys = output_dict.keys()
    for each in range(target_dict["note_events"].shape[0]):
        output_dict_batch = {key: output_dict[key][each].cpu().detach().numpy() for key in keys}
        est_note_events, est_pedal_events = output_dict_to_events(
            output_dict_batch, onset_threshold=config["onset_threshold"],
            offset_threshold=config["offset_threshold"],
            frame_threshold=config["frame_threshold"],
            pedal_offset_threshold=config["pedal_offset_threshold"],
            frames_per_second=config["frame_rate"]
        )

        if est_note_events is None:
            continue

        target_note_events = target_dict["note_events"][each].cpu().detach().numpy()

        # Get the locations where target_note_events is not -100
        # target_note_events is a tensor of shape (num_notes, 4)
        mask = target_note_events[:, 0] != -100
        # Filter the target_note_events using the mask
        target_note_events = target_note_events[mask]

        ref_intervals = target_note_events[:, :2]
        est_intervals = est_note_events[:, :2]
        ref_pitches = target_note_events[:, 2]
        est_pitches = est_note_events[:, 2]

        assert np.min(ref_intervals) >= 0, "Reference intervals contain negative values!"
        assert np.min(est_intervals) >= 0, "Estimated intervals contain negative values!"

        # convert pitches to Hertz
        ref_pitches = 440 * (2 ** ((ref_pitches - 69) / 12))
        est_pitches = 440 * (2 ** ((est_pitches - 69) / 12))

        scores = mir_eval.transcription.evaluate(
            ref_intervals, ref_pitches, est_intervals, est_pitches)

        if scores is None:
            continue

        if offset_ratio is None:
            metrics["Precision_no_offset"].append(scores["Precision_no_offset"])
            metrics["Recall_no_offset"].append(scores["Recall_no_offset"])
        else:
            metrics["Precision"].append(scores["Precision"])
            metrics["Recall"].append(scores["Recall"])

    # Average the metrics across the batch
    for key in metrics:
        metrics[key] = np.mean(metrics[key])
        metrics[key] = np.round(metrics[key], 4)
    
    return metrics

def multipitch_metrics_batch(output_dict: dict, target_dict: dict, config: dict) -> dict:
    """
        Calculate multipitch metrics for a batch of predictions and ground truth.

    """

    # Initialize metrics dictionary
    metrics: dict = {'Precision': [], 'Recall': [], 'Accuracy': []}
    keys = output_dict.keys()
    for each in range(target_dict["note_events"].shape[0]):
        output_dict_batch = {key: output_dict[key][each].cpu().detach().numpy() for key in keys}
        est_note_events, est_pedal_events = output_dict_to_events(
            output_dict_batch, onset_threshold=config["onset_threshold"],
            offset_threshold=config["offset_threshold"],
            frame_threshold=config["frame_threshold"],
            pedal_offset_threshold=config["pedal_offset_threshold"],
            frames_per_second=config["frame_rate"]
        )

        if est_note_events is None:
            continue

        target_note_events = target_dict["note_events"][each].cpu().detach().numpy()

        # Get the locations where target_note_events is not -100
        # target_note_events is a tensor of shape (num_notes, 4)
        mask = target_note_events[:, 0] != -100
        # Filter the target_note_events using the mask
        target_note_events = target_note_events[mask]

        # Convert notes to frames
        preds_roll = notes_to_frames(est_note_events[:, 2].astype(int), (est_note_events[:, :2]*config["frame_rate"]).astype(int), 
                                     target_dict["label_frames"][0].shape)
        y_roll = notes_to_frames(target_note_events[:, 2].astype(int), (target_note_events[:, :2]*config["frame_rate"]).astype(int), 
                                     target_dict["label_frames"][0].shape)

        scores = multipitch_metrics(preds_roll, y_roll, frame_rate=config["frame_rate"])
        if scores is None:
            continue

        metrics["Precision"].append(scores["Precision"])
        metrics["Recall"].append(scores["Recall"])
        metrics["Accuracy"].append(scores["Accuracy"])
    
    # Average the metrics across the batch
    for key in metrics:
        metrics[key] = np.mean(metrics[key])
        metrics[key] = np.round(metrics[key], 4)
    
    return metrics

def save(target_dict: dict, output_dict: dict, config: dict, \
         save_dir: str, batch_index: int) -> None:
    """
    Save audio and piano roll predictions to .npz files.

    Args:
        audio (torch.Tensor): Batch of audio data.
        y (torch.Tensor): Ground truth piano rolls.
        preds (torch.Tensor): Predicted piano rolls.
        threshold (float): Threshold for binarizing predictions.
        save_dir (str): Directory to save the results.
        batch_index (int): Index of the current batch.
    """
    for each in range(target_dict["label_frames"].shape[0]):
        audio_arr = target_dict["audio"][each].squeeze(0).detach().cpu().numpy()

        est_note_events, est_pedal_events = output_dict_to_events(
            output_dict[each], onset_threshold=config["onset_threshold"],
            offset_threshold=config["offset_threshold"],
            frame_threshold=config["frame_threshold"],
            pedal_offset_threshold=config["pedal_offset_threshold"],
            frames_per_second=config["frame_rate"]
        )

        if est_note_events is None:
            continue

        target_note_events = target_dict["note_events"][each]

        # Get the locations where target_note_events is not -100
        # target_note_events is a tensor of shape (num_notes, 4)
        mask = target_note_events[:, 0] != -100
        # Filter the target_note_events using the mask
        target_note_events = target_note_events[mask]

        preds_roll = notes_to_frames_vels(est_note_events[:, 2], est_note_events[:, :2], est_note_events[:, 3],
            target_dict["label_frames"][0].shape)
        y_roll = notes_to_frames_vels(target_note_events[:, 2], target_note_events[:, :2], target_note_events[:, 3],
            target_dict["label_frames"][0].shape)

        Path(save_dir + f"/results/").mkdir(parents=True, exist_ok=True)
        result_dict = {'audio': audio_arr, 'original_roll': y_roll, 'pred_roll': preds_roll}
        np.savez(save_dir + f"/results/{batch_index}_{each}.npz", **result_dict)

@torch.no_grad()
def evaluate(model: Any, dataloader: Any, device: str, 
        prefix: str="valid", offset_ratio=None, config=None, save_dir: str | None=None):
    
    loss_fn = regress_bce

    if not config:
        raise ValueError("Config not passed to evaluate function!")

    model.eval()
    loss_dict: dict = {f'{prefix}_total_loss': 0, f'{prefix}_onset_loss': 0, \
            f'{prefix}_offset_loss': 0, f'{prefix}_frame_loss': 0, \
            f'{prefix}_velocity_loss': 0}
    num_samples = 0

    # Initialize metrics_frames dictionary
    metrics_frames: dict = {'Precision': [], 'Recall': [], 'Accuracy': []}

    # Initialize metrics_note dictionary
    if offset_ratio is not None:
        metrics_note: dict = {'Precision': [], 'Recall': []}
    else:
        metrics_note = {'Precision_no_offset': [], 'Recall_no_offset': []}
    
    for i, (batch_dict) in enumerate(dataloader):    
        # move everything to device
        for key in batch_dict.keys():
            batch_dict[key] = batch_dict[key].to(device)

        # Forward pass
        output_dict = model(batch_dict["feature"])

        # calculate the loss
        loss = loss_fn(output_dict, batch_dict)

        # Update loss dictionary
        loss_dict[f'{prefix}_total_loss'] += loss['total_loss'].item()
        loss_dict[f'{prefix}_onset_loss'] += loss["onset_loss"].item()
        loss_dict[f'{prefix}_offset_loss'] += loss["offset_loss"].item()
        loss_dict[f'{prefix}_frame_loss'] += loss["frame_loss"].item()
        loss_dict[f'{prefix}_velocity_loss'] += loss["velocity_loss"].item()

        num_samples += batch_dict["feature"].shape[0]
    
        # save audio and piano roll
        if save_dir is not None:
            save(batch_dict, output_dict, config, save_dir, i)

        # Calculate multipitch metrics
        frames_scores = multipitch_metrics_batch(output_dict, batch_dict, config)
        for key in metrics_frames:
            metrics_frames[key].append(frames_scores[key])
        
        # Calculate transcription metrics
        notes_scores = transcription_metrics_batch(output_dict, batch_dict, config)
        for key in metrics_note:
            metrics_note[key].append(notes_scores[key])
    
    # Average the metrics across the batch
    for key in metrics_frames:
        metrics_frames[key] = sum(metrics_frames[key])/len(metrics_frames[key])
        metrics_frames[key] = round(metrics_frames[key], 4)
    for key in metrics_note:
        metrics_note[key] = sum(metrics_note[key])/len(metrics_note[key])
        metrics_note[key] = round(metrics_note[key], 4)
    
    # Add the F1-score to the metrics
    metrics_frames["F1"] = 2 * (metrics_frames["Precision"] * \
                metrics_frames["Recall"]) / (metrics_frames["Precision"] + \
                metrics_frames["Recall"] + 2e-16)
    
    if offset_ratio is not None:
        metrics_note["F1"] = 2 * (metrics_note["Precision"] * \
                    metrics_note["Recall"]) / (metrics_note["Precision"] + \
                    metrics_note["Recall"] + 2e-16)
    else:
        # Calculate F1-score for no offset   
        metrics_note["F1_no_offset"] = 2 * (metrics_note["Precision_no_offset"] * \
                    metrics_note["Recall_no_offset"]) / (metrics_note["Precision_no_offset"] + \
                    metrics_note["Recall_no_offset"] + 2e-16)

    # Average the loss
    for key in loss_dict:
        loss_dict[key] = loss_dict[key] / num_samples
        loss_dict[key] = round(loss_dict[key], 4)
    return loss_dict, metrics_frames, metrics_note


def train_step(model: Any, dataloader: Any, device: str, \
    optimizer: Any, scheduler: Any, clip_gradient_norm: float=3.0):

    loss_fn = regress_bce # init loss function
    model.train() # set model to training mode

    loss_dict: dict = {'train_total_loss': 0, 'train_onset_loss': 0, 'train_offset_loss': 0, \
            'train_frame_loss': 0, 'train_velocity_loss': 0}
    num_samples = 0

    for i, batch_dict in enumerate(dataloader):
        # move everything to device
        for key in batch_dict.keys():
            batch_dict[key] = batch_dict[key].to(device)

        # Forward pass
        output_dict = model(batch_dict["feature"])

        # calculate the loss
        loss = loss_fn(output_dict, batch_dict)

        # Update loss dictionary
        loss_dict['train_total_loss'] += loss['total_loss'].item()
        loss_dict['train_onset_loss'] += loss['onset_loss'].item()
        loss_dict['train_offset_loss'] += loss['offset_loss'].item()
        loss_dict['train_frame_loss'] += loss['frame_loss'].item()
        loss_dict['train_velocity_loss'] += loss['velocity_loss'].item()

        num_samples += batch_dict["feature"].shape[0]

        # Zero gradients
        optimizer.zero_grad()

        # Backprop
        loss['total_loss'].backward()

        # Clip gradients if necessary
        # if clip_gradient_norm:
        #     clip_grad_norm_(model.parameters(), clip_gradient_norm)

        # Gradient Descent
        optimizer.step()

        scheduler.step()

    # Average the loss
    for key in loss_dict:
        loss_dict[key] = loss_dict[key] / num_samples
        loss_dict[key] = round(loss_dict[key], 4)
    return loss_dict

def custom_collate_fn(batches):
    """
        Batches is a list of dictionaries and contains
        note_events and pedal_events. Batching 
        other items isn't a problem. However, 
        note_events should have shape of: (len, 4)
        and pedal_events should have shape of: (len, 2)
        where len is the number of notes in the batch.

        Unfortunatelty, some values in note_events or pedal_events 
        do not have the same length and might also be empty.
    """
    max_note_length = max([each['note_events'].shape[0] for each in batches])
    max_pedal_length = max([each['pedal_events'].shape[0] for each in batches])

    for each in batches:
        note_event = each['note_events']
        pedal_event = each['pedal_events']

        if note_event.ndim != 2:
            note_event = np.full((1, 4), -100, dtype=np.float32)
        else:
            # pad note_event to maximum length in the batch
            note_event = np.pad(note_event, ((0, max_note_length - note_event.shape[0]), (0, 0)), \
                mode='constant', constant_values=-100)
            
        each['note_events'] = note_event
        
        if pedal_event.ndim != 2:
            pedal_event = np.full((max_pedal_length, 2), -100, dtype=np.float32)
        else:
            # pad pedal_event to maximum length in the batch
            pedal_event = np.pad(pedal_event, ((0, max_pedal_length - pedal_event.shape[0]), (0, 0)), \
                mode='constant', constant_values=-100)
            
        each['pedal_events'] = pedal_event
    
    collated_dict = {}
    for key in batches[0].keys():
        values = [each[key] for each in batches]
        collated_dict[key] = torch.from_numpy(np.array(values).astype(np.float32))
        
    return collated_dict

def main(config):
    # Create datasets 
    train_dataset = KongDataset(config["paths"])
    valid_dataset = KongDataset(config["paths"], dataset_type="val")
    test_dataset = KongDataset(config["paths"], dataset_type="test")

    # Create dataloaders for each dataset
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], \
            shuffle = True, collate_fn=custom_collate_fn) 
    valid_dataloader = DataLoader(valid_dataset, batch_size=config["batch_size"], \
            shuffle = False, collate_fn=custom_collate_fn) 
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"],\
            shuffle=False, collate_fn=custom_collate_fn) 

    # Load the model
    model = KongModel(config["classes"], config["clue"], config["momentum"], cmp=config["cmp"], \
                      factors=config["factors"])
    model_name = model.__class__.__name__
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = StepLR(optimizer, step_size=config['learning_rate_decay_steps'], \
                       gamma=config['learning_rate_decay_rate'])
    last_epoch = -1

    if config['resume']:
        checkpoint = torch.load(config['resume_path'], weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        last_epoch = checkpoint['epoch']

    # Create runs directory if it does not exist
    Path(config["save_dir"]).mkdir(exist_ok=True, parents=True)

    last_epoch = max(1, last_epoch) # we use 1 since we save with epoch + 1

    best_loss = float('inf')
    for epoch in tqdm(range(last_epoch-1, config["epochs"]), \
            total=config["epochs"]-(last_epoch-1)):
        train_loss_dict = train_step(model, train_dataloader, device, \
                optimizer, scheduler, clip_gradient_norm=config["clip_gradient_norm"])        

        valid_loss_dict, frame_metrics, note_metrics = evaluate(model, valid_dataloader, device, \
                              config=config) 
        
        
        # Log frame and note metrics to wandb
        results = {}

        # Add frame metrics to results
        for key in frame_metrics:
            results[f"frame_{key}"] = frame_metrics[key]

        # Add note metrics to results
        for key in note_metrics:
            results[f"note_{key}"] = note_metrics[key]
        
        # Log results to wandb
        results = results | train_loss_dict | valid_loss_dict
        print(results)
        #wandb.log(results)

        if valid_loss_dict['valid_total_loss'] <= best_loss:
            best_loss = valid_loss_dict['valid_total_loss']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_total_loss': train_loss_dict['train_total_loss'],
                'valid_total_loss': valid_loss_dict['valid_total_loss'],
            }, config["save_dir"] + f"/checkpoint_{epoch}.pt")
        
        print(f"Epoch {epoch+1}/{config['epochs']}, Train Loss: {train_loss_dict['train_total_loss']:.4f}, \
                Valid Loss: {valid_loss_dict['valid_total_loss']:.4f}")

    print(f"Evaluating best model on test set")
    best_model = KongModel(config["classes"], config["clue"], config["momentump"], cmp=config["cmp"], \
                      factos=config["factors"])
    best_model = best_model.to(device)
    checkpoint_path = config["save_dir"]
    best_checkpoints = list(Path(checkpoint_path).glob("checkpoint_*.pt"))
    best_checkpoints.sort(key=lambda x: int(x.stem.split("_")[1]))
    best_checkpoint = str(best_checkpoints[-1])
    best_model.load_state_dict(torch.load(best_checkpoint, weights_only=True)["model_state_dict"])
    test_loss_dict, frame_metrics_test, note_metrics_test = evaluate(best_model, test_dataloader, device,\
                         config=config, \
                         save_dir=checkpoint_path, prefix='test')
    
    print(f"Test Loss: {test_loss_dict['test_total_loss']:.4f}")
    results_test = {
            "test_loss": test_loss_dict['test_total_loss']}

    # Add frame metrics to results
    for key in frame_metrics_test:
        results_test[f"test_frame_{key}"] = frame_metrics_test[key]

    # Add note metrics to results
    for key in note_metrics_test:
        results_test[f"test_note_{key}"] = note_metrics_test[key]
    
    wandb.log(results_test)

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
    #wandb.init(project=config["project_name"], \
    #        config=config)
    main(config)
    #wandb.finish()
