# INDEX/engine.py

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The neural network model
        train_loader (DataLoader): Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scaler: Gradient scaler for mixed precision training
        device: Device to train on

    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0

    for batch in train_loader:
        images = batch['image'].to(device)
        clinical = batch['clinical'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            outputs = model(images, clinical)
            loss = criterion(outputs.squeeze(), labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """
    Validate the model on validation data.

    Args:
        model (nn.Module): The neural network model
        val_loader (DataLoader): Validation data loader
        criterion: Loss function
        device: Device to validate on

    Returns:
        dict: Dictionary containing validation metrics
    """
    model.eval()
    total_loss = 0
    volume_predictions = defaultdict(list)
    volume_targets = defaultdict(list)

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            clinical = batch['clinical'].to(device)
            labels = batch['label'].to(device)
            volume_names = batch['volume_name']

            outputs = model(images, clinical)
            loss = criterion(outputs.squeeze(), labels)

            total_loss += loss.item()
            batch_preds = outputs.squeeze().cpu().numpy()
            batch_targets = labels.cpu().numpy()

            for pred, target, vol_name in zip(batch_preds, batch_targets, volume_names):
                volume_predictions[vol_name].append(pred)
                volume_targets[vol_name].append(target)

    # Average predictions per volume
    final_predictions = []
    final_targets = []
    for vol_name in volume_predictions:
        final_predictions.append(np.mean(volume_predictions[vol_name]))
        final_targets.append(np.mean(volume_targets[vol_name]))

    final_predictions = np.array(final_predictions)
    final_targets = np.array(final_targets)

    if len(set(final_targets)) < 2:
        print("Warning: All targets are identical in validation set")
        return {
            'loss': total_loss / len(val_loader),
            'mae': 0.0,
            'mse': 0.0,
            'r2': 0.0
        }

    mae = np.mean(np.abs(final_targets - final_predictions))
    mse = np.mean((final_targets - final_predictions) ** 2)

    ss_res = np.sum((final_targets - final_predictions) ** 2)
    ss_tot = np.sum((final_targets - np.mean(final_targets)) ** 2)

    r2 = 0.0 if ss_tot == 0 else 1 - (ss_res / ss_tot)

    return {
        'loss': total_loss / len(val_loader),
        'mae': mae,
        'mse': mse,
        'r2': r2
    }