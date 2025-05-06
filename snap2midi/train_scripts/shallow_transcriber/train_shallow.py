"""
    File: train_maestro.py
    Author: Chukwuemeka L. Nkama
    Date: 4/4/2025
    Description: Training script for the shallow transcriber!
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
from .shallow_network import ShallowTranscriber
from .datasets.dataset_shallow import ShallowDataset
from snap2midi.utils.eval_mir import transcription_metrics, multipitch_metrics

def transcription_metrics_batch(pred, gt, threshold, frame_rate, offset_ratio=None):
    """
    Calculate transcription metrics for a batch of predictions and ground truth.
    F1-scores should be ignored and recalculated based on the precision and recall
    values in the returned dict. This is because the F1-score is not a metric that 
    can be averaged across batches. The F1-score is a harmonic mean of precision and 
    recall, and it is not meaningful to average the F1-scores across batches. Instead, 
    we can calculate the final F1-score based on the avg. precision and recall values.
    
    Args:
        pred (torch.Tensor): Predicted piano rolls.
        gt (torch.Tensor): Ground truth piano rolls.
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
        
    for each in range(pred.shape[0]):
        y_roll = gt[each].squeeze(0).detach().cpu().numpy()
        preds_arr = torch.sigmoid(pred[each].squeeze(0))
        preds_roll = (preds_arr.detach().cpu().numpy() > threshold)

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
        pred (torch.Tensor): Predicted piano rolls.
        gt (torch.Tensor): Ground truth piano rolls.
        threshold (float): Threshold for binarizing predictions.
        frame_rate (int): Frames per second.
    Returns:
        metrics (dict): Dictionary containing the calculated metrics.
    """
    # Initialize metrics dictionary
    metrics = {'Precision': [], 'Recall': [], 'Accuracy': []}
    for each in range(pred.shape[0]):
        y_roll = gt[each].squeeze(0).detach().cpu().numpy()
        preds_arr = torch.sigmoid(pred[each].squeeze(0))
        preds_roll = (preds_arr.detach().cpu().numpy() > threshold)

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
    for each in range(audio.shape[0]):
        audio_arr = audio[each].squeeze(0).detach().cpu().numpy()
        y_arr = y[each].squeeze(0).detach().cpu().numpy()
        preds_arr = torch.sigmoid(preds[each].squeeze(0)).detach().cpu().numpy()
        preds_roll = (preds_arr > threshold)

        Path(save_dir + f"/results/").mkdir(parents=True, exist_ok=True)
        result_dict = {'audio': audio_arr, 'original_roll': y_arr, 'pred_roll': preds_roll}
        np.savez(save_dir + f"/results/{batch_index}_{each}.npz", **result_dict)
    
@torch.no_grad()
def evaluate(model, dataloader, device, \
        loss_fn, frame_rate, threshold=0.3, offset_ratio=None, save_dir=None):
    model.eval()
    total_loss = 0
    num_samples = 0

    # Initialize metrics_frames dictionary
    metrics_frames = {'Precision': [], 'Recall': [], 'Accuracy': []}

    # Initialize metrics_note dictionary
    if offset_ratio is not None:
        metrics_note = {'Precision': [], 'Recall': []}
    else:
        metrics_note = {'Precision_no_offset': [], 'Recall_no_offset': []}
    
    for i, (x, y, audio) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        preds = model(x)

        # Loss
        loss = loss_fn(preds, y)
        total_loss += loss.item()
        num_samples += x.shape[0]
    
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

    avg_loss = total_loss / num_samples
    return avg_loss, metrics_frames, metrics_note


def train_step(model, dataloader, device, \
    loss_fn, optimizer):
    model.train()
    total_loss = 0
    num_samples = 0
    for i, (x,y, _) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        preds = model(x)

        # Loss (binary cross entropy)
        loss = loss_fn(preds, y)
        total_loss += loss.item()
        num_samples += x.shape[0]

        # Zero gradients
        optimizer.zero_grad()

        # Backprop
        loss.backward()

        # Gradient Descent
        optimizer.step()

    avg_loss = total_loss / num_samples
    return avg_loss

def main(config):
    # Create datasets 
    train_dataset = ShallowDataset(config["paths"], dataset_type="train")
    valid_dataset = ShallowDataset(config["paths"], dataset_type="val")
    test_dataset = ShallowDataset(config["paths"], dataset_type="test")

    # Create dataloaders for each dataset
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], \
            shuffle = True) 
    valid_dataloader = DataLoader(valid_dataset, batch_size=config["batch_size"], \
            shuffle = False) 
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"],\
            shuffle=False) 

    # Load the model
    model = ShallowTranscriber(config["in_features"], config["hidden_units"], \
            dropout = config["dropout"], out_features = config["out_features"])
    model_name = model.__class__.__name__
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], \
            weight_decay=config["weight_decay"])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    last_epoch = -1

    if config['resume']:
        checkpoint = torch.load(config['resume_path'], weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']

    # Create runs directory if it does not exist
    Path(config["save_dir"]).mkdir(exist_ok=True, parents=True)

    last_epoch = max(1, last_epoch) # we use 1 since we save with epoch + 1

    best_loss = float('inf')
    for epoch in tqdm(range(last_epoch-1, config["epochs"]), \
            total=config["epochs"]-(last_epoch-1)):
        train_loss = train_step(model, train_dataloader, device, \
                loss_fn, optimizer)        

        valid_loss, frame_metrics, note_metrics = evaluate(model, valid_dataloader, device, loss_fn, \
                              config["frame_rate"], threshold=config["threshold"]) 
        
        results = {
            "train_loss": train_loss,
            "valid_loss": valid_loss}

        # Add frame metrics to results
        for key in frame_metrics:
            results[f"frame_{key}"] = frame_metrics[key]

        # Add note metrics to results
        for key in note_metrics:
            results[f"note_{key}"] = note_metrics[key]
        
        # Log results to wandb
        wandb.log(results)

        if valid_loss <= best_loss:
            best_loss = valid_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
            }, config["save_dir"] + f"/checkpoint_{epoch}.pt")
        
        print(f"Epoch {epoch+1}/{config['epochs']}, Train Loss: {train_loss:.4f}, \
                Valid Loss: {valid_loss:.4f}")

    print(f"Evaluating best model on test set")
    best_model = model = ShallowTranscriber(config["in_features"], config["hidden_units"], \
            dropout = config["dropout"], out_features = config["out_features"])
    best_model = best_model.to(device)
    checkpoint_path = config["save_dir"]
    best_checkpoints = sorted(Path(checkpoint_path).glob("checkpoint_*.pt"))
    best_checkpoints = list(Path(checkpoint_path).glob("checkpoint_*.pt"))
    best_checkpoints.sort(key=lambda x: int(x.stem.split("_")[1]))
    best_checkpoint = str(best_checkpoints[-1])
    best_model.load_state_dict(torch.load(best_checkpoint, weights_only=True)["model_state_dict"])
    test_loss, frame_metrics_test, note_metrics_test = evaluate(best_model, test_dataloader, device, loss_fn, \
                         config["frame_rate"], threshold=config["threshold"], \
                         save_dir=checkpoint_path)
    
    print(f"Test Loss: {test_loss:.4f}")
    results_test = {
            "test_loss": test_loss}

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
