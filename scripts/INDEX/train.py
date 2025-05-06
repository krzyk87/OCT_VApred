# INDEX/train.py

import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from DIME.data_loader import OCTDatasetWithClinical, get_transforms
from DIME.model import EfficientNetWithClinical
from DIME.engine import train_epoch, validate

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train OCT image analysis model with clinical data integration for INDEX Dataset.')

    # Required arguments
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save model outputs')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing OCT image files')
    parser.add_argument('--clinical_data_path', type=str, required=True,
                        help='Path to clinical data Excel file')

    # Optional training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (default: 16)')
    parser.add_argument('--image_size', type=int, default=496, # Changed default image size
                        help='Size to resize images (default: 496)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Initial learning rate (default: 0.001)')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate (default: 0.000001)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for optimizer (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train (default: 100)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for training (default: cuda:0)')
    parser.add_argument('--t0', type=int, default=10,
                        help='T0 parameter for cosine annealing (default: 10)')
    parser.add_argument('--t_mult', type=int, default=2,
                        help='T_mult parameter for cosine annealing (default: 2)')

    return parser.parse_args()

def seed_everything(seed=42):
    """
    Set random seeds for reproducibility across numpy, torch, and CUDA.

    Args:
        seed (int): Random seed value, defaults to 42
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config['t0'],
        T_mult=config['t_mult'],
        eta_min=config['min_lr']
    )
    scaler = torch.amp.GradScaler('cuda')

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

def main():
    """
    Main training pipeline orchestrating the entire training process.
    Handles data loading, preprocessing, model training, and cross-validation.
    """
    args = parse_args()

    # Set random seed
    seed_everything(args.seed)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Configuration dictionary from command line arguments
    config = {
        # Paths
        'output_dir': args.output_dir,
        'image_dir': args.image_dir,
        'clinical_data_path': args.clinical_data_path,

        # Training parameters
        'image_size': args.image_size,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'min_lr': args.min_lr,
        'weight_decay': args.weight_decay,
        't0': args.t0,
        't_mult': args.t_mult,
        'epochs': args.epochs,
        'patience': args.patience,
        'device': torch.device(args.device if torch.cuda.is_available() else "cpu"),
        'n_splits': args.n_splits,
    }

    # Load and preprocess clinical data
    clinical_data = pd.read_excel(config['clinical_data_path'])
    clinical_columns = [
        'Baseline VA', 'DM duration (years)'
    ]
    clinical_data['Month 12 VA Binned'] = pd.qcut(clinical_data['Month 12 VA'], q=3, labels=False, duplicates='drop')

    # Image directory
    image_dir = config['image_dir']

    # Setup cross-validation
    stratified_kfold = StratifiedKFold(n_splits=config['n_splits'], shuffle=True, random_state=42)
    fold_results = []

    # Prepare data for stratification
    X = np.arange(len(clinical_data))
    y = clinical_data['Month 12 VA Binned'].values

    print("Distribution of Month 12 VA Bins:")
    print(clinical_data['Month 12 VA Binned'].value_counts())

    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(stratified_kfold.split(X, y)):
        config['fold'] = fold
        print(f'\nTraining Fold {fold + 1}/{config["n_splits"]}')

        train_clinical_data = clinical_data.iloc[train_idx].reset_index(drop=True)
        val_clinical_data = clinical_data.iloc[val_idx].reset_index(drop=True)

        print(f"Fold {fold + 1} Training Stats: "
              f"Min={train_clinical_data['Month 12 VA'].min():.2f}, "
              f"Max={train_clinical_data['Month 12 VA'].max():.2f}, "
              f"Mean={train_clinical_data['Month 12 VA'].mean():.2f}, "
              f"Std={train_clinical_data['Month 12 VA'].std():.2f}")

        print(f"Fold {fold + 1} Validation Stats: "
              f"Min={val_clinical_data['Month 12 VA'].min():.2f}, "
              f"Max={val_clinical_data['Month 12 VA'].max():.2f}, "
              f"Mean={val_clinical_data['Month 12 VA'].mean():.2f}, "
              f"Std={val_clinical_data['Month 12 VA'].std():.2f}")

        # Scale clinical features
        clinical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        clinical_features_scaled_train = clinical_transformer.fit_transform(
            train_clinical_data[clinical_columns].fillna(0)
        )
        clinical_features_scaled_val = clinical_transformer.transform(
            val_clinical_data[clinical_columns].fillna(0)
        )

        # Create datasets and dataloaders
        train_dataset = OCTDatasetWithClinical(
            train_clinical_data,
            clinical_features_scaled_train,
            image_dir,
            transform=get_transforms(config['image_size'], is_training=True)
        )

        val_dataset = OCTDatasetWithClinical(
            val_clinical_data,
            clinical_features_scaled_val,
            image_dir,
            transform=get_transforms(config['image_size'], is_training=False)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            pin_memory=True if torch.cuda.is_available() else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            pin_memory=True if torch.cuda.is_available() else False
        )

        # Initialize and train model
        model = EfficientNetWithClinical(num_clinical_features=len(clinical_columns)).to(config['device'])
        best_mae, best_mse, best_r2 = train_model(model, train_loader, val_loader, config)

        print(f'Fold {fold + 1}, Best MAE: {best_mae:.4f}, Best MSE: {best_mse:.4f}, Best R²: {best_r2:.4f}')
        fold_results.append((best_mae, best_mse, best_r2))

    # Print cross-validation results
    if fold_results:
        avg_mae = sum(result[0] for result in fold_results) / len(fold_results)
        avg_mse = sum(result[1] for result in fold_results) / len(fold_results)
        avg_r2 = sum(result[2] for result in fold_results) / len(fold_results)
        print('\n=== Cross-Validation Results ===')
        for i, (mae, mse, r2) in enumerate(fold_results, 1):
            print(f'Fold {i}: MAE={mae:.4f}, MSE={mse:.4f}, R²={r2:.4f}')
        print(f'Average MAE: {avg_mae:.4f}, Average MSE: {avg_mse:.4f}, Average R²: {avg_r2:.4f}')
    else:
        print("No folds were completed successfully.")

if __name__ == "__main__":
    main()