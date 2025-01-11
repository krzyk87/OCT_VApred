# Visual-Acuity-Prediction-with-Multi-Modal-Deep-Learning
Repo for code relating to the paper 'Enhancing Post-Treatment Visual Acuity Prediction with Multi-Modal Deep Learning on Small-scale Clinical and OCT Datasets'.

![Architecture Draft](architecture_draft.png)

## Setup
To set up the environment for this project, use the `environment.yml` file to create a conda environment with the necessary dependencies by running `conda env create -f environment.yml`. Once the environment is created, activate it using `conda activate your-environment-name`. If you need to install any additional pip dependencies, you can use the `requirements.txt` file by running `pip install -r requirements.txt`. After completing these steps, your environment will be set up and ready to use.
  
## Model Weights

The pre-trained model weights are stored in the `model_weights/` directory. Inside this directory, you will find subdirectories for each dataset: `DIME_weights/` and `INDEX_weights/`. Each folder contains the model weights for the 5 folds, stored as `best_model_fold_1.pt`, `best_model_fold_2.pt`, and so on.

To use the model weights, download the corresponding `.pt` file for the fold you are interested in, and load it using the PyTorch as follows:

```python
import torch
model = EfficientNetWithClinical()
model.load_state_dict(torch.load('model_weights/best_model_fold_1.pt'))
```
## Model Training Scripts

Two standalone training scripts are provided for training the models on the DIME and INDEX datasets:

1. **train_DIME.py**: This script is used to train the model using the DIME dataset.
2. **train_INDEX.py**: This script is used to train the model using the INDEX dataset.

To run either script, simply execute it from the command line with Python:

```bash
python train_DIME.py  # For DIME dataset
python train_INDEX.py  # For INDEX dataset
```

## License
This repository is licensed under the MIT License (See License Info). If you are intending to use this repository for commercial use cases, please check the licenses of all Python packages referenced in the Setup section / described in the `requirements.txt` and `environment.yml`.

## Citation
If you are using this code, please cite:
TBD
