"""
    File: train_maestro.py
    Author: Chukwuemeka L. Nkama
    Date: 4/4/2025
    Description: Training script for the shallow transcriber!
"""

# Imports
import torch
from snap2midi.datasets.dataset import SnapDataset
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
import wandb
import json
from model_loader import load_shallow

@torch.no_grad()
def evaluate(model, dataloader, device, \
        loss_fn):
    model.eval()
    total_loss = 0
    num_samples = 0
    
    for i, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        preds = model(x)

        # Loss
        loss = loss_fn(preds, y)
        total_loss += loss.item()
        num_samples += x.shape[0]

    avg_loss = total_loss / num_samples
    return avg_loss


def train_step(model, dataloader, device, \
    loss_fn, optimizer):
    model.train()
    total_loss = 0
    num_samples = 0
    for i, (x,y) in enumerate(dataloader):
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
    train_dataset = SnapDataset(config["train_path"])
    valid_dataset = SnapDataset(config["valid_path"])
    test_dataset = SnapDataset(config["test_path"])

    # Create dataloaders for each dataset
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], \
            shuffle = True) 
    valid_dataloader = DataLoader(valid_dataset, batch_size=config["batch_size"], \
            shuffle = False) 
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"],\
            shuffle=False) 

    # Load the model
    model = load_shallow(config["in_features"], config["hidden_units"], \
            config["dropout"])
    model_name = model.__class__.__name__
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    Path(config["save_dir"]).mkdir(exist_ok=True)

    last_epoch = max(0, last_epoch)

    
    best_loss = float('-inf')
    for epoch in tqdm(range(last_epoch-1, config["epochs"]), \
            total=config["epochs"]-(last_epoch-1)):
        train_loss = train_step(model, train_dataloader, device, \
                loss_fn, optimizer)        

        valid_loss = evaluate(model, valid_dataloder, device, loss_fn) 

        if valid_loss <= best_loss:
            best_loss = valid_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
            }, config["save_dir"] + f"/{model_name}/checkpoint_{epoch}.pt")

    print(f"Evaluating best model on test set")
    best_model = load_shallow(config["in_features"], config["hidden_units"], \
            config["dropout"])
    checkpoint_path = config["save_dir"] + f"/{model_name}/checkpoint_*.pt"
    best_checkpoints = sorted(Path(checkpoint_path).glob(checkpoint_path))
    best_checkpoint = str(best_checkpoints[-1])
    best_model.load_state_dict(best_checkpoint)
    test_loss = evaluate(best_model, test_dataloader, device, loss_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path")
    args = parser.parse_args()

    # load JSON file
    with open(args.config_path, 'r') as filename:
        content = filename.read()

    config = json.loads(content)
    main(config)
