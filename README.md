# 🧠 MultilayerPerceptron - Breast Cancer Diagnosis

This project implements a **Multilayer Perceptron (Neural Network)** from scratch using only NumPy, without deep learning frameworks. The goal is to diagnose breast cancer by classifying tumors as **Benign (B)** or **Malignant (M)** based on features extracted from biopsy images.

## 📋 Table of Contents

- [Project Description](#-project-description)
- [Neural Network Architecture](#-neural-network-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Data Preparation](#1-data-preparation)
  - [Training](#2-training)
  - [Prediction](#3-prediction)
  - [Model Comparison](#4-model-comparison)
- [Model Configuration](#-model-configuration)
- [Trained Models](#-trained-models)
- [Dataset](#-dataset)
- [Implemented Features](#-implemented-features)
- [Results](#-results)

---

## 🎯 Project Description

This is an educational implementation of an artificial neural network that:

- **Implements everything from scratch**: Forward propagation, backpropagation, optimizers (SGD, Adam), activation functions (ReLU, Sigmoid, Softmax)
- **Classifies tumors**: Predicts whether a tumor is benign or malignant based on 30 numerical features
- **Handles imbalanced data**: Uses techniques like class weighting and early stopping
- **Visualizes learning**: Generates loss and accuracy plots during training
- **Compares models**: Allows evaluation of different architectures and configurations

---

## 🏗️ Neural Network Architecture

### Main Components

#### 1. **DenseLayer** (Fully Connected Layer)
- Linear transformation: $Z = XW + b$
- Activation functions:
  - **ReLU**: $f(x) = \max(0, x)$
  - **Sigmoid**: $f(x) = \frac{1}{1 + e^{-x}}$
  - **Softmax**: $f(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$
- Dropout for regularization
- Weight initialization:
  - **He Uniform** (for ReLU)
  - **Xavier** (for Sigmoid/Tanh)

#### 2. **NeuralNetwork** (Complete Network)
- Stacks multiple dense layers
- Loss function: **Binary Cross-Entropy** / **Categorical Cross-Entropy**
- Optimizers:
  - **SGD** (Stochastic Gradient Descent)
  - **Adam** (Adaptive Moment Estimation)
- L2 Regularization (Weight Decay)
- Early Stopping with best weights restoration

#### 3. **DataLoader**
- Loads and splits data: **Training** / **Validation** / **Test**
- Z-score normalization using training set statistics
- Imbalanced class handling with weighting

---

## 📁 Project Structure

```
MultilayerPerceptron/
├── data/                           # Project data
│   ├── data.csv                    # Original dataset (569 samples)
│   ├── data_training.csv           # Training set
│   ├── data_validation.csv         # Validation set
│   ├── data_test.csv               # Test set
│   └── data_*_predictions_*.csv    # Predictions from each model
│
├── model_config/                   # YAML configurations
│   ├── BreastCancerDiagnosis_config_v100.yaml
│   ├── BreastCancerDiagnosis_config_v110.yaml
│   └── BreastCancerDiagnosis_config_v120.yaml
│
├── models/                         # Trained models
│   ├── breast_cancer_diagnosis_v1.0.0.npz
│   ├── breast_cancer_diagnosis_v1.1.0.npz
│   ├── breast_cancer_diagnosis_v1.2.0.npz
│   └── *_training_log.txt          # Training logs
│
├── plots/                          # Generated plots
│   └── *_learning_curves.png
│
├── src/                            # Source code
│   ├── training/                   # Training modules
│   │   ├── NeuralNetwork.py        # Neural network implementation
│   │   ├── DenseLayer.py           # Individual dense layer
│   │   ├── DataLoader.py           # Data loading and preprocessing
│   │   ├── ModelConfig.py          # YAML configuration parser
│   │   └── train.py                # Training script
│   │
│   ├── prediction/                 # Prediction modules
│   │   └── predict.py              # Inference script
│   │
│   ├── plotting/                   # Visualization
│   │   └── compare_all_trainings.py
│   │
│   ├── Dataset_Exploration/        # Exploratory analysis
│   │   └── exploration.ipynb
│   │
│   ├── Utils/                      # Utilities
│   │   └── colors.py               # Terminal colors
│   │
│   └── split_data.py               # Data splitting
│
├── Makefile                        # Automated commands
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip

### Installation Steps

1. **Clone the repository** (or unzip the project)
```bash
cd MultilayerPerceptron
```

2. **Create virtual environment and install dependencies**
```bash
make all
# Equivalent to:
# make create-venv
# make install-deps
```

3. **Activate the virtual environment**
```bash
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### Dependencies
- **numpy**: Matrix operations and linear algebra
- **pandas**: Tabular data manipulation
- **matplotlib**: Plot visualization
- **jupyterlab**: Exploratory analysis (optional)

---

## 🚀 Usage

### 1. Data Preparation

#### Manual Split
```bash
make split
# Will ask for split ratio (e.g., 0.2 for 20% test)
```

Or directly:
```bash
python src/split_data.py data/data.csv
```

The script will automatically split the data into:
- `data_training.csv` (training)
- `data_validation.csv` (validation)

---

### 2. Training

#### Train Model v1.0.0 (Adam + Sigmoid)
```bash
make train1
# Equivalent to:
# python -m src.training.train \
#   --dataset data/data.csv \
#   --config model_config/BreastCancerDiagnosis_config_v100.yaml
```



#### Train Model v1.1.0 (Adam + ReLU + Early Stopping)
```bash
make train2
```

#### Train Model v1.2.0 (SGD + Dropout)
```bash
make train3
```

#### Training Outputs:
- **Trained model**: `models/breast_cancer_diagnosis_vX.X.X.npz`
- **Training log**: `models/breast_cancer_diagnosis_vX.X.X_training_log.txt`
- **Learning curve plot**: `plots/breast_cancer_diagnosis_vX.X.X_learning_curves.png`
<img width="2082" height="730" alt="v1 0 0-plot" src="https://github.com/user-attachments/assets/72641114-a69c-48d2-8219-496c5684f107" />
<img width="2082" height="730" alt="v1 1 0-plot" src="https://github.com/user-attachments/assets/be3a38df-03d4-43cb-bf2b-ff9dcc07f4e1" />
<img width="2082" height="730" alt="v1 2 0-plot" src="https://github.com/user-attachments/assets/8c955207-5bb4-4d0f-941a-73cea73b3c70" />

---

### 3. Prediction

#### Predict with Model v1.0.0
```bash
make predict1
```

#### Predict with Model v1.1.0
```bash
make predict2
```

#### Predict with Model v1.2.0
```bash
make predict3
```

#### Manual Command
```bash
python src/prediction/predict.py \
  --dataset data/data_test.csv \
  --model models/breast_cancer_diagnosis_v1.0.0.npz \
  --config model_config/BreastCancerDiagnosis_config_v100.yaml
```

#### Displayed Metrics:
- **Binary Cross-Entropy Loss**
- **Accuracy** (Overall accuracy)
- **Precision** (Precision for Malignant class)
- **Recall** (Sensitivity)
- **F1-Score**
- **Confusion Matrix**

<img width="333" height="400" alt="prediction-v1 0 0" src="https://github.com/user-attachments/assets/ba8e188c-f81d-4a97-a974-8f4752fa593e" />
<img width="333" height="400" alt="prediction-v1 1 0" src="https://github.com/user-attachments/assets/06a859f6-bd52-45aa-ac7a-ecb856832c98" />
<img width="333" height="400" alt="prediction-v1 2 0" src="https://github.com/user-attachments/assets/ecae01ee-7fc1-4ec3-bdd5-3366a1c75302" />

---

### 4. Model Comparison

Compare all trained models in a single plot:
```bash
make compare-all
```

Generates comparative visualizations of:
- Training and validation loss
- Accuracy per epoch
- Final metrics of each model

<img width="1030" height="405" alt="model-comparation" src="https://github.com/user-attachments/assets/d1486e8b-0bef-4f76-8f47-d5131b945d31" />
<img width="2684" height="1032" alt="comparation" src="https://github.com/user-attachments/assets/ae817ddb-da18-4123-97e5-f5621303d953" />

---

## ⚙️ Model Configuration

Models are configured via YAML files. Example structure:

```yaml
model_config:
  model_name: "breast_cancer_diagnosis_v1.0.0"
  model_version: "1.0.0"

  architecture:
    hidden_layers: [60, 15]        # Hidden layers
    input_dim: 30                  # 30 input features
    activation_fn: "sigmoid"       # relu, sigmoid
    use_dropout_rate: true         # Enable dropout

  classes:
    - "B"  # Benign
    - "M"  # Malignant

training_params:
  test_size: 0.1                  # 15% for testing
  val_size: 0.1                   # 15% for validation
  random_state: 40                # Random seed
  epochs: 900
  batch_size: 24
  dropout_rate: 0.0               # Dropout rate

  optimizer:
    name: "adam"                  # sgd, adam
    learning_rate: 0.0001
    weight_decay: 0.0001          # L2 regularization

  loss_function:
    use_pos_weight: true          # Positive class weighting
    pos_weight: 0.0               # 0.0 = automatic

  early_stopping:
    patience: 30                  # 0 = disabled
    delta: 0.0001                 # Minimum improvement required
    restore_best_weights: false   # Restore best weights
```

---

## 📊 Trained Models

### Model v1.0.0
- **Architecture**: [30] → [60, 15] → [2]
- **Activation**: Sigmoid
- **Optimizer**: Adam (lr=0.0001, wd=0.0001)
- **Epochs**: 900
- **Batch Size**: 24
- **Early Stopping**: Yes (patience=30, delta=0.0001)

### Model v1.1.0
- **Architecture**: [30] → [48, 24, 12] → [2]
- **Activation**: ReLU
- **Optimizer**: Adam (lr=0.0001)
- **Epochs**: 500
- **Batch Size**: 2
- **Early Stopping**: Yes (patience=10, delta=0.0001)

### Model v1.2.0
- **Architecture**: [30] → [24, 24, 12] → [2]
- **Activation**: Sigmoid
- **Optimizer**: SGD (lr=0.001, wd=0.001)
- **Epochs**: 1200
- **Batch Size**: 4
- **Dropout**: 0.1
- **Early Stopping**: Yes (patience=15, delta=0.0002)

---

## 📦 Dataset

### Wisconsin Breast Cancer Dataset
- **Source**: UCI Machine Learning Repository
- **Samples**: 569 cases
  - **Benign (B)**: 357 (62.7%)
  - **Malignant (M)**: 212 (37.3%)
- **Features**: 30 numerical values
  - Radius, texture, perimeter, area, smoothness, etc.
  - Computed for cell nucleus in digitized images

### CSV Format
```
ID,Diagnosis,Feature1,Feature2,...,Feature30
842302,M,17.99,10.38,122.8,1001,...
```

- **Column 0**: Patient ID
- **Column 1**: Diagnosis (M=Malignant, B=Benign)
- **Columns 2-31**: 30 numerical features

---

## ✨ Implemented Features

### Optimization Algorithms
- [x] **SGD** (Stochastic Gradient Descent)
- [x] **Adam** (Adaptive Moment Estimation)
  - With bias correction
  - First and second moment estimates

### Activation Functions
- [x] **ReLU** (Rectified Linear Unit)
- [x] **Sigmoid**
- [x] **Softmax** (output layer)

### Regularization
- [x] **Dropout** during training
- [x] **Weight Decay** (L2 regularization)
- [x] **Early Stopping** with best weights restoration

### Weight Initialization
- [x] **He Uniform** (for ReLU)
- [x] **Xavier/Glorot** (for Sigmoid)

### Loss Functions
- [x] **Binary Cross-Entropy**
- [x] **Categorical Cross-Entropy**
- [x] **Class weighting** for imbalanced datasets

### Normalization
- [x] **Z-score normalization** (standardization)
- Statistics saved for inference

### Metrics
- [x] Accuracy
- [x] Precision
- [x] Recall
- [x] F1-Score
- [x] Confusion Matrix

---

## 📈 Results

Models achieve the following approximate metrics on the test set:

| Model   | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| v1.0.0  | ~98.25%  | ~95.83%   | ~100%  | ~97.87%  |
| v1.1.0  | ~100%    | ~100%     | ~100%  | ~100%    |
| v1.2.0  | ~96.49%  | ~95.65%   | ~95.65% | ~95.65% |

*Note: Results may vary depending on random data split*

---

## 🧪 Exploratory Analysis

Includes a Jupyter Notebook for exploratory data analysis:

```bash
make jupyter
# Navigate to: src/Dataset_Exploration/exploration.ipynb
```

The notebook includes:
- Class distribution
- Feature correlation
- Outlier visualization
- Descriptive statistics

---

## 📝 Training Logs

Each training generates a detailed log in `models/*_training_log.txt`:

```
Training Log: breast_cancer_diagnosis_v1.0.0
================================================================================
Optimizer: adam
Batch size: 24
Learning rate: 0.0001
Weight decay: 0.0001
Epochs: 900
================================================================================

epoch 01/900 - loss: 0.6234 - acc: 0.6544 - val_loss: 0.5832 - val_acc: 0.7012
epoch 02/900 - loss: 0.5123 - acc: 0.7456 - val_loss: 0.4821 - val_acc: 0.7893
...

================================================================================
TRAINING SUMMARY
================================================================================
Final Training Loss:     0.0542
Final Training Accuracy: 0.9812 (98.12%)
Final Validation Loss:   0.0623
Final Validation Accuracy: 0.9756 (97.56%)
```

---

## 🛡️ Data Validation

The project implements strict validation:
- **Data leakage** prevention (normalization only with training stats)
- Stratified data splitting
- Input dimension validation
- Missing value handling

---

## 🎓 Educational Purpose

This project is designed to:
- Understand neural network fundamentals
- Implement backpropagation manually
- Experiment with hyperparameters
- Compare optimizers and activation functions
- Visualize the learning process

**Not optimized for production**, but excellent for learning how neural networks work "under the hood".

---

## 📄 License

This project is open source and available for educational purposes.

---

## 👤 Author

Project developed as a practical introduction to artificial neural networks.

---

## 🔗 References

- [Wisconsin Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- [Neural Networks and Deep Learning - Michael Nielsen](http://neuralnetworksanddeeplearning.com/)
- [Deep Learning Book - Ian Goodfellow](https://www.deeplearningbook.org/)

---

## 🆘 Troubleshooting

### Error: "No module named 'src'"
```bash
# Make sure you're in the project root directory
cd /home/adiaz-uf/MultilayerPerceptron
source venv/bin/activate
```

### Error: "ModuleNotFoundError: No module named 'yaml'"
```bash
make install-deps
# or
pip install -r requirements.txt
```

### Plots are not generated
Verify that matplotlib is installed correctly and the `plots/` directory exists:
```bash
mkdir -p plots
```

### Inconsistent results
Adjust the `random_state` parameter in YAML configurations for reproducibility.

---

**Happy learning! 🎉**
