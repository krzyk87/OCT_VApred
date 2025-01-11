"""
Training script for OCT image analysis with clinical data integration for INDEX Dataset.
This script implements a deep learning model combining OCT images with clinical features
to predict visual acuity outcomes. It uses EfficientNet as the backbone with a custom
attention mechanism for clinical feature integration.
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import tifffile as tiff
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import math

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train OCT image analysis model with clinical data integration.')
    
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
    parser.add_argument('--image_size', type=int, default=512,
                      help='Size to resize images (default: 512)')
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


class OCTDatasetWithClinical(Dataset):
    """
    Dataset class combining OCT images with clinical features.
    
    Args:
        clinical_data (pd.DataFrame): DataFrame containing clinical information
        clinical_features_scaled (np.ndarray): Scaled clinical features
        image_dir (str): Directory containing OCT image files
        transform (albumentations.Compose, optional): Image transformations
    """
    def __init__(self, clinical_data, clinical_features_scaled, image_dir, transform=None):
        self.clinical_data = clinical_data.reset_index(drop=True)
        self.clinical_features_scaled = clinical_features_scaled
        self.image_dir = image_dir
        self.transform = transform

        if np.any(np.isnan(clinical_features_scaled)):
            raise ValueError("NaN values found in clinical features")
        if np.any(np.isinf(clinical_features_scaled)):
            raise ValueError("Infinite values found in clinical features")

        self.image_paths = self._create_image_path_mapping()
        self.slice_info = self._get_slice_info()

        self.volume_to_idx = {
            str(row['baseline image filename']).replace('.tiff', ''): idx
            for idx, row in self.clinical_data.iterrows()
            if not pd.isna(row['baseline image filename'])
        }

    def _create_image_path_mapping(self):
        """Create a mapping of volume names to their full directory paths"""
        image_paths = {}
        for root, dirs, _ in os.walk(self.image_dir):
            for dir_name in dirs:
                folder_path = os.path.join(root, dir_name)
                image_paths[dir_name] = folder_path
        return image_paths

    def _get_slice_info(self):
        """Gather information about all image slices in the dataset"""
        slice_info = []
        for idx, row in self.clinical_data.iterrows():
            volume_name = str(row['baseline image filename']).replace('.tiff', '')
            volume_dir = self.image_paths.get(volume_name)

            if volume_dir:
                for slice_file in os.listdir(volume_dir):
                    if slice_file.endswith('.tiff'):
                        slice_path = os.path.join(volume_dir, slice_file)
                        label = torch.tensor(row['Month 12 VA'], dtype=torch.float)
                        slice_info.append((slice_path, label, volume_name))
            else:
                print(f"Warning: Folder for volume {volume_name} not found.")

        return slice_info

    def __getitem__(self, idx):
        """Get a single item from the dataset"""
        slice_path, label, volume_name = self.slice_info[idx]
        volume_idx = self.volume_to_idx[volume_name]
        clinical_features = torch.tensor(self.clinical_features_scaled[volume_idx].astype(np.float32))

        image = tiff.imread(slice_path)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = torch.tensor(image.transpose(2, 0, 1).astype(np.float32))

        return {
            'image': image,
            'clinical': clinical_features,
            'label': label,
            'volume_name': slice_path,
        }

    def __len__(self):
        """Return the number of slices in the dataset"""
        return len(self.slice_info)


class EfficientNetWithClinical(nn.Module):
    """
    Combined EfficientNet and clinical features model with attention mechanism.
    
    Args:
        num_clinical_features (int): Number of clinical features
        dropout_rate (float): Dropout rate for regularization
    """
    def __init__(self, num_clinical_features=3, dropout_rate=0.3):
        super().__init__()
        
        # Initialize and customize EfficientNet backbone
        self.backbone = models.efficientnet_b0(pretrained=True)
        original_weight = self.backbone.features[0][0].weight.clone()
        self.backbone.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.backbone.features[0][0].weight.data = original_weight.mean(dim=1, keepdim=True)
        
        # Set up feature extraction
        for param in self.backbone.parameters():
            param.requires_grad = False
        for idx in [6, 7]:
            for param in self.backbone.features[idx].parameters():
                param.requires_grad = True

        # Add dimension reduction layers
        self.dim_reduce = nn.Sequential(
            nn.Conv2d(1280, 320, 1),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True)
        )

        # Clinical features processing
        self.clinical_net = nn.Sequential(
            nn.Linear(num_clinical_features, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Cross-attention mechanism
        self.clinical_attention = nn.Sequential(
            nn.Linear(16, 128),
            nn.GELU(),
            nn.Linear(128, 320),
            nn.Sigmoid()
        )
        
        # Final prediction layers
        self.final_layers = nn.Sequential(
            nn.Linear(320 + 16, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, image, clinical):
        """Forward pass combining image and clinical features"""
        x = self.backbone.features(image)
        x = self.dim_reduce(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        image_features = torch.flatten(x, 1)
        
        clinical_features = self.clinical_net(clinical)
        attention_weights = self.clinical_attention(clinical_features)
        attended_image_features = image_features * attention_weights
        
        combined = torch.cat((attended_image_features, clinical_features), dim=1)
        return self.final_layers(combined)


def get_transforms(image_size=512, is_training=True):
    """
    Get image transformations for training or validation.
    
    Args:
        image_size (int): Target image size
        is_training (bool): Whether to include training augmentations
    
    Returns:
        albumentations.Compose: Composition of transforms
    """
    if is_training:
        return A.Compose([
            A.Rotate(limit=(11), p=0.8, border_mode=0),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit_x=16.13 / image_size,
                shift_limit_y=27.43 / image_size,
                scale_limit=0,
                rotate_limit=0,
                p=0.8,
                border_mode=0
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.GaussNoise(var_limit=(0.10, 8.63), mean=0, p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.2
            ),
            A.CoarseDropout(
                max_holes=8,
                max_height=20,
                max_width=20,
                min_holes=2,
                min_height=8,
                min_width=8,
                p=0.2
            ),
            A.Normalize(mean=0.5, std=0.5),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=0.5, std=0.5),
            ToTensorV2(),
        ])


def get_transforms(image_size=496, is_training=True):
    if is_training:
        return A.Compose([
            A.Rotate(
                limit=8,
                p=0.8,
                border_mode=0
            ),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit_x=13.73/image_size,
                shift_limit_y=34.72/image_size,
                scale_limit=0,
                rotate_limit=0,
                p=0.8,
                border_mode=0
            ),
            
            A.GaussianBlur(
                blur_limit=(5, 13),
                p=0.3,
            ),
            
            A.GaussNoise(
                var_limit=(0.10, 8.58),
                mean=0,
                p=0.3,
            ),
            
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.2
            ),
            
            A.CoarseDropout(
                max_holes=8,
                max_height=20,
                max_width=20,
                min_holes=2,
                min_height=8,
                min_width=8,
                p=0.2
            ),
            
            A.Normalize(mean=0.5, std=0.5),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=0.5, std=0.5),
            ToTensorV2(),
        ])

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
    clinical_data = clinical_data[~clinical_data['study number'].isin(['SEI-039', 'SEI-038'])]
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
