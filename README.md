# Visual-Acuity-Prediction-with-Multimodal-Deep-Learning

Repo for code relating to the paper 'Enhancing Post-Treatment Visual Acuity Prediction with Multimodal Deep Learning on Small-scale Clinical and OCT Datasets'.

![Architecture Draft](model_architecture.png)

## 🚀 Setup

To set up the environment for this project:

1. Create a conda environment with the necessary dependencies:
   ```bash
   conda env create -f environment.yml
   ```

2. Activate the environment:
   ```bash
   conda activate your-environment-name
   ```

3. Install additional pip dependencies if needed:
   ```bash
   pip install -r requirements.txt
   ```

Your environment is now ready to use!

## 🔬 Model Training Scripts

The repository contains two main training scripts under the `scripts` folder:

- `train_DIME.py`: Training script for the DIME dataset
- `train_INDEX.py`: Training script for the INDEX dataset

### 🏃‍♂️ Running the Training Scripts

First, activate your conda environment:
```bash
conda activate your_environment_name
```

Then run either script using the following format:

#### DIME Training
```bash
python train_DIME.py \
  --output_dir path/to/output \
  --image_dir path/to/oct/images \
  --clinical_data_path path/to/clinical_data.xlsx
```

#### INDEX Training
```bash
python train_INDEX.py \
  --output_dir path/to/output \
  --image_dir path/to/oct/images \
  --clinical_data_path path/to/clinical_data.xlsx
```

### ⚙️ Optional Arguments

Both scripts accept the following optional arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--batch_size` | Batch size for training | 16 |
| `--image_size` | Size to resize images | 512 |
| `--learning_rate` | Initial learning rate | 0.001 |
| `--epochs` | Number of epochs to train | 100 |
| `--seed` | Random seed | 42 |

## 📊 Model Evaluation Scripts

The repository contains two main k-fold evaluation scripts to replicate results reported in the paper:

- `evaluate_DIME.py`: Evaluation script for the DIME dataset
- `evaluate_INDEX.py`: Evaluation script for the INDEX dataset

### 📈 Running Evaluation Scripts

#### DIME Evaluation
```bash
python evaluate_DIME.py \
  --model_weights_dir path/to/model/weights \
  --output_dir path/to/output \
  --image_dir path/to/oct/images \
  --clinical_data_path path/to/clinical_data.xlsx
```

#### INDEX Evaluation
```bash
python evaluate_INDEX.py \
  --model_weights_dir path/to/model/weights \
  --output_dir path/to/output \
  --image_dir path/to/oct/images \
  --clinical_data_path path/to/clinical_data.xlsx
```

## 🏆 Model Weights

The pre-trained model weights are stored in the `model_weights/` directory with subdirectories for each dataset:
- `DIME_weights/`
- `INDEX_weights/`

Each folder contains model weights for the 5 folds: `best_model_fold_1.pt`, `best_model_fold_2.pt`, etc.

To use the model weights:

```python
import torch
model = EfficientNetWithClinical()
model.load_state_dict(torch.load('model_weights/best_model_fold_1.pt'))
```

## 📜 License

This repository is licensed under the MIT License (See License Info). If you are intending to use this repository for commercial use cases, please check the licenses of all Python packages referenced in the Setup section / described in the `requirements.txt` and `environment.yml`.

## 📚 Citation

If you are using this code, please cite:

```
Anderson, M., Corona, V., Stankiewicz, A., Habib, M., Steel, D.H. and Obara, B., 
Enhancing Post-Treatment Visual Acuity Prediction with Multimodal Deep Learning 
on Small-scale Clinical and OCT Datasets. In Medical Imaging with Deep Learning.
```
