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
from snap2midi.utils.eval_mir import transcription_metrics, multipitch_metrics, note_extract, notes_to_frames
from .datasets.dataset_oaf import OAFDataset
from .onsets_and_frames import OnsetsAndFrames

def transcription_metrics_batch(pred, gt, threshold, frame_rate, offset_ratio=None):
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
    Returns:
        metrics (dict): Dictionary containing the calculated metrics.
    """
    # Initialize metrics dictionary
    if offset_ratio is not None:
        metrics = {
            "Precision": [],
            "Recall": []}
    else:
        metrics = {
            "Precision_no_offset": [],
            "Recall_no_offset": []}
    
    on_preds, frame_preds, vel_preds = pred
    y_onsets, y_frames, y_velocities = gt
        
    for each in range(y_onsets.shape[0]):
        note_preds, int_preds, _ = note_extract(on_preds[each], frame_preds[each], vel_preds[each], \
                onset_thresh=threshold, frame_thresh=threshold)
        
        note_gt, int_gt, _ = note_extract(y_onsets[each], y_frames[each], y_velocities[each], \
                onset_thresh=threshold, frame_thresh=threshold)

        # Convert notes to frames
        preds_roll = notes_to_frames(note_preds, int_preds, y_onsets[0].shape)
        y_roll = notes_to_frames(note_gt, int_gt, y_onsets[0].shape)

        scores = transcription_metrics(preds_roll, y_roll, frame_rate=frame_rate)
        
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

def multipitch_metrics_batch(pred, gt, threshold, frame_rate):
    """
    Calculate multipitch metrics for a batch of predictions and ground truth.
    Args:
        pred (Tuple): Predicted piano rolls.
        gt (Tuple): Ground truth piano rolls.
        threshold (float): Threshold for binarizing predictions.
        frame_rate (int): Frames per second.
    Returns:
        metrics (dict): Dictionary containing the calculated metrics.
    """

    on_preds, frame_preds, vel_preds = pred
    y_onsets, y_frames, y_velocities = gt

    # Initialize metrics dictionary
    metrics = {'Precision': [], 'Recall': [], 'Accuracy': []}
    for each in range(y_onsets.shape[0]):
        note_preds, int_preds, _ = note_extract(on_preds[each], frame_preds[each], vel_preds[each], \
                onset_thresh=threshold, frame_thresh=threshold)
        
        note_gt, int_gt, _ = note_extract(y_onsets[each], y_frames[each], y_velocities[each], \
                onset_thresh=threshold, frame_thresh=threshold)

        # Convert notes to frames
        preds_roll = notes_to_frames(note_preds, int_preds, y_onsets[0].shape)
        y_roll = notes_to_frames(note_gt, int_gt, y_onsets[0].shape)

        scores = multipitch_metrics(preds_roll, y_roll, frame_rate=frame_rate)
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

def save(audio, y, preds, threshold, save_dir, batch_index):
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
    on_preds, frame_preds, vel_preds = preds
    y_onsets, y_frames, y_velocities = y

    for each in range(audio.shape[0]):
        audio_arr = audio[each].squeeze(0).detach().cpu().numpy()
        note_preds, int_preds, _ = note_extract(on_preds[each], frame_preds[each], vel_preds[each], \
                onset_thresh=threshold, frame_thresh=threshold)
        
        note_gt, int_gt, _ = note_extract(y_onsets[each], y_frames[each], y_velocities[each], \
                onset_thresh=threshold, frame_thresh=threshold)

        # Convert notes to frames
        preds_roll = notes_to_frames(note_preds, int_preds, y_onsets[0].shape)
        y_roll = notes_to_frames(note_gt, int_gt, y_onsets[0].shape)

        Path(save_dir + f"/results/").mkdir(parents=True, exist_ok=True)
        result_dict = {'audio': audio_arr, 'original_roll': y_roll, 'pred_roll': preds_roll}
        np.savez(save_dir + f"/results/{batch_index}_{each}.npz", **result_dict)

def loss_velocity(velocity_pred, velocity_label, onset_label):
        denominator = onset_label.sum()
        if denominator.item() == 0:
            return denominator
        else:
            return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator

@torch.no_grad()
def evaluate(model, dataloader, device, \
        loss_fn, frame_rate, threshold=0.3, prefix="valid", offset_ratio=None, save_dir=None):
    model.eval()
    loss_dict = {f'{prefix}_total_loss': 0, f'{prefix}_onset_loss': 0, \
            f'{prefix}_offset_loss': 0, f'{prefix}_frame_loss': 0, \
            f'{prefix}_velocity_loss': 0}
    num_samples = 0

    # Initialize metrics_frames dictionary
    metrics_frames = {'Precision': [], 'Recall': [], 'Accuracy': []}

    # Initialize metrics_note dictionary
    if offset_ratio is not None:
        metrics_note = {'Precision': [], 'Recall': []}
    else:
        metrics_note = {'Precision_no_offset': [], 'Recall_no_offset': []}
    
    for i, (x, y_frame, y_onset, y_offset, y_velocity,\
             audio) in enumerate(dataloader):
        x = x.to(device)
        y_frame = y_frame.to(device)
        y_onset = y_onset.to(device)
        y_offset = y_offset.to(device)
        y_velocity = y_velocity.to(device).float()/128

        # Forward pass
        on_preds, off_preds, _, frame_preds, vel_preds = model(x)

        # Loss
        # onset_loss + offset_loss + frame_loss + velocity_loss
        onset_loss = loss_fn(on_preds, y_onset)
        offset_loss = loss_fn(off_preds, y_offset)
        frame_loss = loss_fn(frame_preds, y_frame)
        velocity_loss = loss_velocity(vel_preds, y_velocity, y_onset)
        loss = onset_loss + offset_loss + frame_loss + velocity_loss

        # Update loss dictionary
        loss_dict[f'{prefix}_total_loss'] += loss.item()
        loss_dict[f'{prefix}_onset_loss'] += onset_loss.item()
        loss_dict[f'{prefix}_offset_loss'] += offset_loss.item()
        loss_dict[f'{prefix}_frame_loss'] += frame_loss.item()
        loss_dict[f'{prefix}_velocity_loss'] += velocity_loss.item()

        num_samples += x.shape[0]

        preds = (torch.sigmoid(on_preds), torch.sigmoid(frame_preds),\
                  vel_preds)
        y = (y_onset, y_frame, y_velocity)
    
        # save audio and piano roll
        if save_dir is not None:
            save(audio, y, preds, threshold, save_dir, i)
        
        # Calculate multipitch metrics
        frames_scores = multipitch_metrics_batch(preds, y, threshold, \
                frame_rate)
        for key in metrics_frames:
            metrics_frames[key].append(frames_scores[key])
        
        # Calculate transcription metrics
        notes_scores = transcription_metrics_batch(preds, y, threshold, \
                frame_rate, offset_ratio=offset_ratio)
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


def train_step(model, dataloader, device, \
    loss_fn, optimizer, scheduler, clip_gradient_norm=3.0):
    model.train()
    loss_dict = {'train_total_loss': 0, 'train_onset_loss': 0, 'train_offset_loss': 0, \
            'train_frame_loss': 0, 'train_velocity_loss': 0}
    num_samples = 0
    for i, (x, y_frame, y_onset, y_offset, y_velocity,\
             audio) in enumerate(dataloader):
        x = x.to(device)
        y_frame = y_frame.to(device)
        y_onset = y_onset.to(device)
        y_offset = y_offset.to(device)
        y_velocity = y_velocity.to(device).float()/128

        # Forward pass
        on_preds, off_preds, _, frame_preds, vel_preds = model(x)

        # Loss
        # onset_loss + offset_loss + frame_loss + velocity_loss
        onset_loss = loss_fn(on_preds, y_onset)
        offset_loss = loss_fn(off_preds, y_offset)
        frame_loss = loss_fn(frame_preds, y_frame)
        velocity_loss = loss_velocity(vel_preds, y_velocity, y_onset)
        loss = onset_loss + offset_loss + frame_loss + velocity_loss

        # Update loss dictionary
        loss_dict['train_total_loss'] += loss.item()
        loss_dict['train_onset_loss'] += onset_loss.item()
        loss_dict['train_offset_loss'] += offset_loss.item()
        loss_dict['train_frame_loss'] += frame_loss.item()
        loss_dict['train_velocity_loss'] += velocity_loss.item()

        num_samples += x.shape[0]

        # Zero gradients
        optimizer.zero_grad()

        # Backprop
        loss.backward()

        # Clip gradients if necessary
        if clip_gradient_norm:
            clip_grad_norm_(model.parameters(), clip_gradient_norm)

        # Gradient Descent
        optimizer.step()

        scheduler.step()

    # Average the loss
    for key in loss_dict:
        loss_dict[key] = loss_dict[key] / num_samples
        loss_dict[key] = round(loss_dict[key], 4)
    return loss_dict

def main(config):
    # Create datasets 
    train_dataset = OAFDataset(config["train_path"])
    valid_dataset = OAFDataset(config["valid_path"])
    test_dataset = OAFDataset(config["test_path"])

    # Create dataloaders for each dataset
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], \
            shuffle = True) 
    valid_dataloader = DataLoader(valid_dataset, batch_size=config["batch_size"], \
            shuffle = False) 
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"],\
            shuffle=False) 

    # Load the model
    model = OnsetsAndFrames(config["in_features"], 128, factor=config["factor"], \
            model_complexity=config["model_complexity"])
    model_name = model.__class__.__name__
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = StepLR(optimizer, step_size=config['learning_rate_decay_steps'], \
                       gamma=config['learning_rate_decay_rate'])
    loss_fn = torch.nn.BCEWithLogitsLoss()
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
                loss_fn, optimizer, scheduler, clip_gradient_norm=config["clip_gradient_norm"])        

        valid_loss_dict, frame_metrics, note_metrics = evaluate(model, valid_dataloader, device, loss_fn, \
                              config["frame_rate"], threshold=config["threshold"]) 
        
        
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
        wandb.log(results)

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
    best_model = model = OnsetsAndFrames(config["in_features"], 128, factor=config["factor"], \
            model_complexity=config["model_complexity"])
    best_model = best_model.to(device)
    checkpoint_path = config["save_dir"]
    best_checkpoints = list(Path(checkpoint_path).glob("checkpoint_*.pt"))
    best_checkpoints.sort(key=lambda x: int(x.stem.split("_")[1]))
    best_checkpoint = str(best_checkpoints[-1])
    best_model.load_state_dict(torch.load(best_checkpoint, weights_only=True)["model_state_dict"])
    test_loss_dict, frame_metrics_test, note_metrics_test = evaluate(best_model, test_dataloader, device, loss_fn, \
                         config["frame_rate"], threshold=config["threshold"], \
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
    wandb.init(project=config["project_name"], \
            config=config)
    main(config)
    wandb.finish()
