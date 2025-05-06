# Visual-Acuity-Prediction-with-Multimodal-Deep-Learning

Repo for code relating to the paper _"Enhancing Post-Treatment Visual Acuity Prediction with Multimodal Deep Learning on Small-scale Clinical and OCT Datasets."_

![Architecture Draft](model_architecture.png)

---

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

---

## 📂 Script Structure

The repository organizes the training and evaluation scripts for each dataset within dedicated subdirectories under the `scripts` folder:

- `scripts/DIME/`: Contains scripts for the DIME dataset (`data_loader.py`, `engine.py`, `evaluate.py`, `model.py`, `train.py`)
- `scripts/INDEX/`: Contains scripts for the INDEX dataset (`data_loader_index.py`, `evaluate_index.py`, `model_index.py`, `train_index.py`)

---

## 🔬 Model Training Scripts

The main training scripts are located in the respective dataset folders:

- `scripts/DIME/train.py`: Training script for the DIME dataset  
- `scripts/INDEX/train_index.py`: Training script for the INDEX dataset

---

## 🏃‍♂️ Running the Training Scripts

First, activate your conda environment:

```bash
conda activate your-environment-name
```

Then navigate to the specific dataset's script directory and run the training script:

### DIME Training

```bash
cd scripts/DIME
python train.py \
  --output_dir path/to/output \
  --image_dir path/to/oct/images \
  --clinical_data_path path/to/clinical_data.xlsx
```

### INDEX Training

```bash
cd scripts/INDEX
python train_index.py \
  --output_dir path/to/output \
  --image_dir path/to/oct/images \
  --clinical_data_path path/to/clinical_data.xlsx
```

---

## ⚙️ Optional Arguments

Both training scripts accept the following optional arguments:

| Argument           | Description                 | Default (DIME) | Default (INDEX) |
|--------------------|-----------------------------|----------------|-----------------|
| `--batch_size`     | Batch size for training      | 16             | 16              |
| `--image_size`     | Size to resize images        | 512            | 496             |
| `--learning_rate`  | Initial learning rate        | 0.001          | 0.001           |
| `--epochs`         | Number of epochs to train    | 100            | 100             |
| `--seed`           | Random seed                  | 42             | 42              |

Refer to the individual training script (`train.py` for DIME, `train_index.py` for INDEX) for a complete list of available arguments.

---

## 📊 Model Evaluation Scripts

The main k-fold evaluation scripts are located in the respective dataset folders:

- `scripts/DIME/evaluate.py`: Evaluation script for the DIME dataset  
- `scripts/INDEX/evaluate_index.py`: Evaluation script for the INDEX dataset

---

## 📈 Running Evaluation Scripts

Navigate to the specific dataset's script directory and run the evaluation script:

### DIME Evaluation

```bash
cd scripts/DIME
python evaluate.py \
  --model_weights_dir path/to/model/weights \
  --output_dir path/to/output \
  --image_dir path/to/oct/images \
  --clinical_data_path path/to/clinical_data.xlsx
```

### INDEX Evaluation

```bash
cd scripts/INDEX
python evaluate_index.py \
  --model_weights_dir path/to/model/weights \
  --output_dir path/to/output \
  --image_dir path/to/oct/images \
  --clinical_data_path path/to/clinical_data.xlsx
```

---

## 🏆 Model Weights

Pre-trained model weights are stored in the `model_weights/` directory with subdirectories for each dataset:

- `model_weights/DIME_weights/`
- `model_weights/INDEX_weights/`

Each folder contains weights for the 5 folds:

```
best_model_fold_1.pt  
best_model_fold_2.pt  
best_model_fold_3.pt  
best_model_fold_4.pt  
best_model_fold_5.pt
```

To use the model weights in your own scripts, ensure you adjust the model instantiation to match the architecture defined in the respective `model.py` or `model_index.py` file:

```python
import torch
from scripts.DIME.model import EfficientNetWithClinical as DIME_Model
from scripts.INDEX.model_index import EfficientNetWithClinical as INDEX_Model

# For DIME
dime_model = DIME_Model()
dime_model.load_state_dict(torch.load('model_weights/DIME_weights/best_model_fold_1.pt'))

# For INDEX
index_model = INDEX_Model(num_clinical_features=2)  # Adjust num_clinical_features if needed
index_model.load_state_dict(torch.load('model_weights/INDEX_weights/best_model_fold_1.pt'))
```

---

## 📜 License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

If you intend to use this repository for commercial purposes, please verify the licenses of all Python packages listed in `requirements.txt` and `environment.yml`.

---

## 📚 Citation

If you use this code, please cite:

Anderson, M., Corona, V., Stankiewicz, A., Habib, M., Steel, D.H. and Obara, B., 2025. *Enhancing post-treatment visual acuity prediction with multimodal deep learning on small-scale clinical and OCT datasets*. Medical Imaging with Deep Learning.

