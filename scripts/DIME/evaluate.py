# DIME/evaluate.py

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
from collections import defaultdict

from DIME.data_loader import OCTDatasetWithClinical, get_transforms
from DIME.model import EfficientNetWithClinical

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate OCT image analysis model with clinical data integration.')

    # Required arguments
    parser.add_argument('--model_weights_dir', type=str, required=True,
                        help='Directory containing model weights for each fold')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save evaluation outputs')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing OCT image files')
    parser.add_argument('--clinical_data_path', type=str, required=True,
                        help='Path to clinical data Excel file')

    # Optional arguments (matching training)
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation (default: 16)')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Size to resize images (default: 512)')
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

def evaluate_fold(model, val_loader, device):
    model.eval()
    predictions = []
    targets = []
    volume_predictions = defaultdict(list)
    volume_targets = defaultdict(list)

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            clinical = batch['clinical'].to(device)
            labels = batch['label'].cpu().numpy()
            volume_names = batch['volume_name']

            outputs = model(images, clinical).cpu().numpy()

            for pred, target, vol_name in zip(outputs, labels, volume_names):
                volume_predictions[vol_name].append(pred[0])
                volume_targets[vol_name].append(target)

    # Average predictions per volume
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

    # Load clinical data
    clinical_data = pd.read_excel(args.clinical_data_path)
    clinical_data = clinical_data[~clinical_data['study number'].isin(['SEI-039', 'SEI-038'])]
    clinical_columns = ['Baseline VA', 'DM duration (years)', 'Age']
    clinical_data['Baseline VA'].fillna(0, inplace=True)

    # Bin target variable for stratification
    clinical_data['VA EOS SE Binned'] = pd.qcut(clinical_data['EOS VA'], q=3, labels=False, duplicates='drop')

    # Setup cross-validation
    stratified_kfold = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    X = np.arange(len(clinical_data))
    y = clinical_data['VA EOS SE Binned'].values

    all_metrics = []
    all_predictions = []
    all_targets = []

    # Evaluate each fold
    for fold, (train_idx, val_idx) in enumerate(stratified_kfold.split(X, y)):
        print(f'\nEvaluating Fold {fold + 1}/{args.n_splits}')

        # Split data
        train_clinical_data = clinical_data.iloc[train_idx].reset_index(drop=True)
        val_clinical_data = clinical_data.iloc[val_idx].reset_index(drop=True)

        # Scale clinical features (using the scaler fitted on the training data if available)
        clinical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        # **Important:** In a real scenario, you should load the fitted scaler
        # from the training phase to ensure consistent scaling.
        clinical_features_scaled_val = clinical_transformer.fit_transform(
            train_clinical_data[clinical_columns].fillna(0) # Fit on training split
        ).transform(val_clinical_data[clinical_columns].fillna(0)) # Transform validation split


        # Create validation dataset
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

        # Load model and weights
        model = EfficientNetWithClinical(num_clinical_features=len(clinical_columns)).to(device)
        weights_path = os.path.join(args.model_weights_dir, f'best_model_fold_{fold + 1}.pt')
        if not os.path.exists(weights_path):
            print(f"Warning: Weights file not found for fold {fold + 1}")
            continue

        model.load_state_dict(torch.load(weights_path, map_location=device))

        # Evaluate
        metrics, predictions, targets = evaluate_fold(model, val_loader, device)
        all_metrics.append(metrics)
        all_predictions.extend(predictions)
        all_targets.extend(targets)

        print(f'Fold {fold + 1} Results:')
        for metric, value in metrics.items():
            print(f'{metric.upper()}: {value:.4f}')

    # Calculate and print overall metrics
    if all_metrics:
        print('\n=== Overall Cross-Validation Results ===')
        metrics_df = pd.DataFrame(all_metrics)
        mean_metrics = metrics_df.mean()
        std_metrics = metrics_df.std()

        for metric in mean_metrics.index:
            print(f'{metric.upper()}: {mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}')

        # Save predictions
        results_df = pd.DataFrame({
            'Predicted': all_predictions,
            'Actual': all_targets
        })
        results_df.to_csv(os.path.join(args.output_dir, 'evaluation_results.csv'), index=False)
    else:
        print("No folds were evaluated successfully.")

if __name__ == "__main__":
    main()