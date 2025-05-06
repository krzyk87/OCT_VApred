# DIME/engine.py

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

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

    mae = mean_absolute_error(final_targets, final_predictions)
    mse = mean_squared_error(final_targets, final_predictions)

    ss_res = np.sum((final_targets - final_predictions) ** 2)
    ss_tot = np.sum((final_targets - np.mean(final_targets)) ** 2)

    r2 = 0.0 if ss_tot == 0 else 1 - (ss_res / ss_tot)

    return {
        'loss': total_loss / len(val_loader),
        'mae': mae,
        'mse': mse,
        'r2': r2
    }

def train_model(model, train_loader, val_loader, config):
    """
    Train the model using the specified configuration.

    Args:
        model (nn.Module): The neural network model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        config (dict): Training configuration parameters

    Returns:
        tuple: Best validation metrics (MAE, MSE, R²)
    """
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config['t0'],
        T_mult=config['t_mult'],
        eta_min=config['min_lr']
    )
    scaler = torch.cuda.amp.GradScaler('cuda')

    best_val_mae = float('inf')
    best_val_mse = float('inf')
    best_val_r2 = float('-inf')
    patience_counter = 0

    for epoch in range(config['epochs']):
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, config['device'])

        # Validation
        val_metrics = validate(model, val_loader, criterion, config['device'])

        # Print progress
        print(f'Epoch {epoch+1}/{config["epochs"]}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_metrics["loss"]:.4f}')
        print(f'Val MAE: {val_metrics["mae"]:.4f}')
        print(f'Val MSE: {val_metrics["mse"]:.4f}')
        print(f'Val R2: {val_metrics["r2"]:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}\n')

        # Update best metrics
        if val_metrics['mse'] < best_val_mse:
            best_val_mse = val_metrics['mse']
            patience_counter = 0
        else:
            patience_counter += 1
            print(f'No improvement in MSE. Patience counter: {patience_counter}/{config["patience"]}')

        if val_metrics['r2'] > best_val_r2:
            best_val_r2 = val_metrics['r2']
            # Save the model weights
            model_path = os.path.join(
                config['output_dir'],
                f"best_model_fold_{config['fold'] + 1}.pt"
            )
            torch.save(model.state_dict(), model_path)

        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']

        # Early stopping check
        if patience_counter >= config['patience']:
            break

        scheduler.step()

    return best_val_mae, best_val_mse, best_val_r2