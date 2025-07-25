# DIME/data_loader.py

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile as tiff
import albumentations as A
from albumentations.pytorch import ToTensorV2

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
                        label = torch.tensor(row['EOS VA'], dtype=torch.float)
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

def get_transforms(image_size=512, is_training=True):
    if is_training:
        return A.Compose([
            A.Rotate(
                limit=(11),  # Degrees
                p=0.8,
                border_mode=0  # cv2.BORDER_CONSTANT
            ),
            A.HorizontalFlip(p=0.5),
            # Shift without scaling or rotation to isolate shifting
            A.ShiftScaleRotate(
                shift_limit_x=16.13 / image_size,  # ≈0.0548 (fraction of width)
                shift_limit_y=27.43 / image_size,  # ≈0.1478 (fraction of height)
                scale_limit=0,                     # No scaling
                rotate_limit=0,                    # No rotation
                p=0.8,
                border_mode=0
            ),
            
            # Gaussian Blur with kernel size adjusted to 3-7
            A.GaussianBlur(
                blur_limit=(3, 5),
                p=0.3,
            ),
            
            # Gaussian Noise with variance adjusted to 0.10-6.90
            A.GaussNoise(
                var_limit=(0.10, 8.63),
                mean=0,
                p=0.3,
            ),
            
            # Random Brightness and Contrast
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.2
            ),
            
            # Coarse Dropout with dimensions adjusted based on image size
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
