"""
Evaluation script for OCT image analysis with clinical data integration.
This script implements a deep learning model combining OCT images with clinical features
to predict visual acuity outcomes. It uses EfficientNet as the backbone with a custom
attention mechanism for clinical feature integration.
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import tifffile as tiff
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import math

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate OCT image analysis model with clinical data integration.')
    
    parser.add_argument('--model_weights_dir', type=str, required=True,
                      help='Directory containing model weights for each fold')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save evaluation outputs')
    parser.add_argument('--image_dir', type=str, required=True,
                      help='Directory containing OCT image files')
    parser.add_argument('--clinical_data_path', type=str, required=True,
                      help='Path to clinical data Excel file')
    
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size for evaluation (default: 16)')
    parser.add_argument('--image_size', type=int, default=496,
                      help='Size to resize images (default: 496)')
    parser.add_argument('--n_splits', type=int, default=5,
                      help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cuda:0',
                      help='Device to use for evaluation (default: cuda:0)')
    
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

def evaluate_fold(model, val_loader, device):
    model.eval()
    predictions = []
    targets = []
    volume_predictions = {}
    volume_targets = {}
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            clinical = batch['clinical'].to(device)
            labels = batch['label'].cpu().numpy()
            volume_names = batch['volume_name']
            
            outputs = model(images, clinical).cpu().numpy()
            
            for pred, target, vol_name in zip(outputs, labels, volume_names):
                if vol_name not in volume_predictions:
                    volume_predictions[vol_name] = []
                    volume_targets[vol_name] = []
                volume_predictions[vol_name].append(pred[0])
                volume_targets[vol_name].append(target)
    
    for vol_name in volume_predictions:
        predictions.append(np.mean(volume_predictions[vol_name]))
        targets.append(np.mean(volume_targets[vol_name]))
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    metrics = {
        'mae': mean_absolute_error(targets, predictions),
        'mse': mean_squared_error(targets, predictions),
        'rmse': np.sqrt(mean_squared_error(targets, predictions)),
        'r2': r2_score(targets, predictions)
    }
    
    return metrics, predictions, targets

def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    clinical_data = pd.read_excel(args.clinical_data_path)
    clinical_data = clinical_data[~clinical_data['study number'].isin(['SEI-039', 'SEI-038'])]
    clinical_columns = ['Baseline VA', 'DM duration (years)']

    clinical_data['Month 12 VA Binned'] = pd.qcut(clinical_data['Month 12 VA'], q=3, labels=False, duplicates='drop')

    stratified_kfold = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    X = np.arange(len(clinical_data))
    y = clinical_data['Month 12 VA Binned'].values

    all_metrics = []
    all_predictions = []
    all_targets = []

    for fold, (train_idx, val_idx) in enumerate(stratified_kfold.split(X, y)):
        print(f'\nEvaluating Fold {fold + 1}/{args.n_splits}')
        
        train_clinical_data = clinical_data.iloc[train_idx].reset_index(drop=True)
        val_clinical_data = clinical_data.iloc[val_idx].reset_index(drop=True)

        clinical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        clinical_features_scaled_train = clinical_transformer.fit_transform(
            train_clinical_data[clinical_columns].fillna(0)
        )
        clinical_features_scaled_val = clinical_transformer.transform(
            val_clinical_data[clinical_columns].fillna(0)
        )

        val_dataset = OCTDatasetWithClinical(
            val_clinical_data,
            clinical_features_scaled_val,
            args.image_dir,
            transform=get_transforms(args.image_size, is_training=False)
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True if torch.cuda.is_available() else False
        )

        model = EfficientNetWithClinical(num_clinical_features=len(clinical_columns)).to(device)
        weights_path = os.path.join(args.model_weights_dir, f'best_model_fold_{fold + 1}.pt')
        if not os.path.exists(weights_path):
            print(f"Warning: Weights file not found for fold {fold + 1}")
            continue
            
        model.load_state_dict(torch.load(weights_path, map_location=device))

        metrics, predictions, targets = evaluate_fold(model, val_loader, device)
        all_metrics.append(metrics)
        all_predictions.extend(predictions)
        all_targets.extend(targets)

        print(f'Fold {fold + 1} Results:')
        for metric, value in metrics.items():
            print(f'{metric.upper()}: {value:.4f}')

    if all_metrics:
        print('\n=== Overall Cross-Validation Results ===')
        metrics_df = pd.DataFrame(all_metrics)
        mean_metrics = metrics_df.mean()
        std_metrics = metrics_df.std()

        for metric in mean_metrics.index:
            print(f'{metric.upper()}: {mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}')

        results_df = pd.DataFrame({
            'Predicted': all_predictions,
            'Actual': all_targets
        })
        results_df.to_csv(os.path.join(args.output_dir, 'evaluation_results.csv'), index=False)
    else:
        print("No folds were evaluated successfully.")

if __name__ == "__main__":
    main()