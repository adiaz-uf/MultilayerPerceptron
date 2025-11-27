import sys
import os
import numpy as np
import pandas as pd
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.training.ModelConfig import ModelConfig
from src.training.NeuralNetwork import NeuralNetwork
from src.training.DataLoader import DataLoader
from src.Utils import BOLD_RED, BOLD_GREEN, BOLD_CYAN, BOLD_MAGENTA, BOLD_YELLOW, RESET


def load_data_for_prediction(data_path):
    """
    Load data from CSV for prediction.
    The CSV always has: id, diagnosis, feature1, feature2, ..., feature30
    - If diagnosis is empty: prediction mode
    - If diagnosis has B/M values: evaluation mode
    
    Args:
        data_path: Path to CSV file
        
    Returns:
        Tuple of (X, y, ids, has_labels)
        - X: Features array (30 features)
        - y: Labels array (None if diagnosis is empty)
        - ids: Sample IDs
        - has_labels: Boolean indicating if diagnosis column has values
    """
    df = pd.read_csv(data_path, header=None)
    
    # Column structure: [id, diagnosis, 30 features]
    # Columns 0 = id, 1 = diagnosis, 2-31 = features
    
    ids = df.iloc[:, 0].values
    diagnosis_col = df.iloc[:, 1]
    X = df.iloc[:, 2:].values  # All columns from index 2 onwards (30 features)
    
    # Check if diagnosis column has values (not empty/NaN)
    has_labels = diagnosis_col.notna().sum() > 0
    
    y = None
    if has_labels:
        # Convert diagnosis to numeric: M=1, B=0
        y = diagnosis_col.map({'M': 1, 'B': 0}).values.reshape(-1, 1)
    
    return X, y, ids, has_labels


def load_and_predict(model_path, data_path, config_path):
    """
    Load a trained model and make predictions on a dataset.
    
    Args:
        model_path: Path to the saved model file (.npz)
        data_path: Path to the dataset CSV file
        config_path: Path to the model configuration file (.yaml)
    """
    print(f"{BOLD_CYAN}=" * 60)
    print("BREAST CANCER DIAGNOSIS - PREDICTION MODE")
    print(f"{BOLD_CYAN}={RESET}" * 60)
    
    # Load configuration
    print(f"\nLoading configuration from: {config_path}")
    config = ModelConfig(config_path)
    
    # Initialize neural network
    print(f"Building neural network architecture...{BOLD_MAGENTA}")
    model = NeuralNetwork(config)
    
    # Load trained weights
    print(f"{RESET}\nLoading trained weights from: {model_path}")
    normalization_params = model.load_model(model_path)
    
    if normalization_params is None:
        print(f"{BOLD_YELLOW}Warning: No normalization parameters found in model.{RESET}")
        return 1
    
    # Load data
    print(f"Loading dataset from: {data_path}")
    X, y_true, ids, has_labels = load_data_for_prediction(data_path)
    
    if has_labels:
        print(f"{BOLD_GREEN}Dataset contains labels (evaluation mode){RESET}")
    else:
        print(f"{BOLD_YELLOW}Dataset without labels (prediction mode){RESET}")
    
    print(f"Samples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}\n")
    
    # Normalize using saved parameters from training
    mean = normalization_params['mean']
    std = normalization_params['std']
    X_normalized = (X - mean) / std
    print("Using normalization parameters from training")
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.forward(X_normalized, training=False)
    
    # Convert to class predictions
    pred_classes = (y_pred[:, 1] > y_pred[:, 0]).astype(int)  # 1=M, 0=B
    confidences = np.max(y_pred, axis=1)
    
    # Display results
    print(f"{BOLD_CYAN}\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60 + f"{RESET}")
    print(f"Dataset size: {X.shape[0]} samples\n")
    
    # If we have labels, calculate metrics
    if has_labels:
        # Convert labels to one-hot encoding
        def to_one_hot(y):
            n_samples = y.shape[0]
            one_hot = np.zeros((n_samples, 2))
            one_hot[np.arange(n_samples), y.flatten().astype(int)] = 1
            return one_hot
        
        y_true_one_hot = to_one_hot(y_true)
        
        # Calculate metrics
        loss = model.binary_cross_entropy(y_true_one_hot, y_pred)
        accuracy = model.accuracy(y_true_one_hot, y_pred)
        
        # Calculate confusion matrix
        y_true_classes = y_true.flatten().astype(int)
        
        TP = np.sum((pred_classes == 1) & (y_true_classes == 1))
        TN = np.sum((pred_classes == 0) & (y_true_classes == 0))
        FP = np.sum((pred_classes == 1) & (y_true_classes == 0))
        FN = np.sum((pred_classes == 0) & (y_true_classes == 1))
        
        # Compute precision
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0

        # Compute recall
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        # Compute F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{BOLD_GREEN}EVALUATION METRICS:{RESET}")
        print(f"  Binary Cross-Entropy Loss: {loss:.6f}")
        print(f"  Accuracy:                  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision (Malignant):     {precision:.4f} ({precision*100:.2f}%)")
        print(f"  Recall (Malignant):        {recall:.4f} ({recall*100:.2f}%)")
        print(f"  F1-Score:                  {f1:.4f}")
        
        print(f"\n{BOLD_CYAN}CONFUSION MATRIX:{RESET}")
        print(f"                Predicted")
        print(f"                B      M")
        print(f"Actual   B    {BOLD_GREEN}{TN:3d}    {BOLD_RED}{FP:3d}{RESET}")
        print(f"         M    {BOLD_RED}{FN:3d}    {BOLD_GREEN}{TP:3d}{RESET}")
    else:
        print(f"{BOLD_GREEN}PREDICTIONS:{RESET}")
        print(f"  Benign (B):    {np.sum(pred_classes == 0)} samples")
        print(f"  Malignant (M): {np.sum(pred_classes == 1)} samples")
    
    # Show sample predictions
    print(f"\n" + "-" * 70)
    if has_labels:
        print("Sample Predictions (first 10):")
        print("-" * 70)
        print(f"{'ID':>10} | {'True':>10} | {'Pred':>10} | {'Conf':>8} | {'Result':>6}")
        print("-" * 70)
        
        for i in range(min(10, len(y_true))):
            true_label = "Benign" if y_true[i][0] == 0 else "Malignant"
            pred_label = "Benign" if pred_classes[i] == 0 else "Malignant"
            
            true_color = BOLD_GREEN if y_true[i][0] == 0 else BOLD_RED
            pred_color = BOLD_GREEN if pred_classes[i] == 0 else BOLD_RED
            
            result = f"{BOLD_GREEN}✓{RESET}" if true_label == pred_label else f"{BOLD_RED}✗{RESET}"
            
            print(f"{ids[i]:>10} | {true_color}{true_label:<10}{RESET} | "
                  f"{pred_color}{pred_label:<10}{RESET} | {confidences[i]:>8.2%} | {result:>6}")
    else:
        print("Sample Predictions (first 10):")
        print("-" * 70)
        print(f"{'ID':>10} | {'Prediction':>10} | {'Confidence':>12} | {'Prob(B)':>8} | {'Prob(M)':>8}")
        print("-" * 70)
        
        for i in range(min(10, len(pred_classes))):
            pred_label = "Benign" if pred_classes[i] == 0 else "Malignant"
            pred_color = BOLD_GREEN if pred_classes[i] == 0 else BOLD_RED
            
            print(f"{ids[i]:>10} | {pred_color}{pred_label:<10}{RESET} | "
                  f"{confidences[i]:>11.2%} | {y_pred[i,0]:>7.2%} | {y_pred[i,1]:>7.2%}")
    
    print("=" * 70)
    
    # Save predictions to CSV file
    output_dir = os.path.dirname(data_path) or '.'
    base_name = os.path.splitext(os.path.basename(data_path))[0]
    output_file = os.path.join(output_dir, f"{base_name}_predictions_v{config.model_version}.csv")
    
    print(f"\n{BOLD_CYAN}Saving predictions to: {output_file}{RESET}")
    
    with open(output_file, 'w') as f:
        # Write header
        if has_labels:
            f.write("ID,True_Label,Predicted_Label,Confidence,Prob_Benign,Prob_Malignant,Correct\n")
        else:
            f.write("ID,Predicted_Label,Confidence,Prob_Benign,Prob_Malignant\n")
        
        # Write predictions for all samples
        for i in range(len(pred_classes)):
            pred_label = "B" if pred_classes[i] == 0 else "M"
            
            if has_labels:
                true_label = "B" if y_true[i][0] == 0 else "M"
                correct = "Yes" if true_label == pred_label else "No"
                f.write(f"{ids[i]},{true_label},{pred_label},{confidences[i]:.6f},"
                       f"{y_pred[i,0]:.6f},{y_pred[i,1]:.6f},{correct}\n")
            else:
                f.write(f"{ids[i]},{pred_label},{confidences[i]:.6f},"
                       f"{y_pred[i,0]:.6f},{y_pred[i,1]:.6f}\n")
    
    print(f"{BOLD_GREEN}✓ Successfully saved {len(pred_classes)} predictions{RESET}")
    print(f"{BOLD_CYAN}={'='*60}{RESET}\n")
    
    return 0


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description='Make predictions using a trained Multilayer Perceptron for Breast Cancer Diagnosis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='data/data_validation.csv',
        help='Path to the dataset CSV file (default: data/data_validation.csv)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/breast_cancer_diagnosis_v1.0.0.npz',
        help='Path to the trained model file (default: models/breast_cancer_diagnosis_v1.0.0.npz)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='model_config/BreastCancerDiagnosis_config_v100.yaml',
        help='Path to the model configuration YAML file (default: model_config/BreastCancerDiagnosis_config_v100.yaml)'
    )
    
    args = parser.parse_args()
    
    try:
        # Validate paths
        if not os.path.exists(args.dataset):
            print(f"{BOLD_RED}Error: Dataset file not found: {args.dataset}{RESET}")
            return 1
        
        if not os.path.exists(args.model):
            print(f"{BOLD_RED}Error: Model file not found: {args.model}{RESET}")
            return 1
        
        if not os.path.exists(args.config):
            print(f"{BOLD_RED}Error: Config file not found: {args.config}{RESET}")
            return 1
        
        print(f"{BOLD_GREEN}Configuration:{RESET}")
        print(f"  Dataset: {args.dataset}")
        print(f"  Model:   {args.model}")
        print(f"  Config:  {args.config}\n")
        
        return load_and_predict(args.model, args.dataset, args.config)
        
    except Exception as e:
        print(f"{BOLD_RED}\nError during prediction: {e}{RESET}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())