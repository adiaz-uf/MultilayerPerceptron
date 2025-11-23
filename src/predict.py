import sys
import os
import numpy as np
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.training.ModelConfig import ModelConfig
from src.training.NeuralNetwork import NeuralNetwork
from src.training.DataLoader import DataLoader

from Utils import BOLD_RED, BOLD_GREEN, BOLD_CYAN, BOLD_MAGENTA, RESET


def load_and_predict(model_path, data_path, config_path):
    """
    Load a trained model and make predictions on a dataset.
    
    Args:
        model_path: Path to the saved model file (.npy)
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
        print("Warning: No normalization parameters found in model. Using data statistics.")
    
    # Load data
    print(f"Loading dataset from: {data_path}\n")
    data_loader = DataLoader(config)
    
    # Check if it's a full dataset or already split
    if os.path.exists(data_path):
        # Load the specific file
        X, y = data_loader._load_raw_data(data_path)
        
        # Normalize using saved parameters from training
        if normalization_params:
            mean = normalization_params['mean']
            std = normalization_params['std']
            X_normalized = (X - mean) / std
            print("Using normalization parameters from training")
        else:
            # Fallback: use data statistics (not recommended)
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            std[std == 0] = 1e-8
            X_normalized = (X - mean) / std
            print("Using data statistics for normalization (not ideal)")
        
        y_true = y
    else:
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    # Convert labels to one-hot encoding
    def to_one_hot(y):
        n_samples = y.shape[0]
        one_hot = np.zeros((n_samples, 2))
        one_hot[np.arange(n_samples), y.flatten().astype(int)] = 1
        return one_hot
    
    y_true_one_hot = to_one_hot(y_true)
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.forward(X_normalized, training=False)
    
    # Calculate binary cross-entropy loss
    loss = model.binary_cross_entropy(y_true_one_hot, y_pred)
    
    # Calculate accuracy
    accuracy = model.accuracy(y_true_one_hot, y_pred)
    
    # Display results
    print(f"{BOLD_CYAN}\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"{RESET}Dataset size:              {X_normalized.shape[0]} samples")
    print(f"Binary Cross-Entropy Loss: {loss:.6f}")
    print(f"Accuracy:                  {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Show some example predictions
    print("\n" + "-" * 60)
    print("Sample Predictions (first 10):")
    print("-" * 60)
    print(f"{'Sample':>8} | {'True':>10} | {'Pred (B)':>10} | {'Pred (M)':>10} | {'Result':>10}")
    print("-" * 60)
    
    for i in range(min(10, len(y_true))):
        true_label = f"{BOLD_GREEN}Benign{RESET}   " if y_true[i][0] == 0 else f"{BOLD_RED}Malignant{RESET}"
        pred_benign = y_pred[i][0]
        pred_malignant = y_pred[i][1]

        if pred_benign > pred_malignant:
            predicted_label = f"{BOLD_GREEN}Benign{RESET}   " 
        else:
            predicted_label = f"{BOLD_RED}Malignant{RESET}"

        result = f"{BOLD_GREEN}✓{RESET}" if true_label == predicted_label else f"{BOLD_RED}✗{RESET}"
        
        print(f"{i+1:>8} | {true_label:>10}  | {pred_benign:>10.4f} | {pred_malignant:>10.4f} | {result:>20}")
    
    print("=" * 60)
    
    # Save predictions to CSV file
    output_dir = os.path.dirname(data_path) or '.'
    base_name = os.path.splitext(os.path.basename(data_path))[0]
    output_file = os.path.join(output_dir, f"{base_name}_predictions.csv")
    
    print(f"\n{BOLD_CYAN}Saving predictions to: {output_file}{RESET}")
    
    with open(output_file, 'w') as f:
        # Write header
        f.write("Sample,True_Label,Pred_Benign,Pred_Malignant,Predicted_Label,Confidence,Correct\n")
        
        # Write predictions for all samples
        for i in range(len(y_true)):
            true_label = "B" if y_true[i][0] == 0 else "M"
            pred_benign = y_pred[i, 0]
            pred_malignant = y_pred[i, 1]
            predicted_label = "B" if pred_benign > pred_malignant else "M"
            confidence = max(pred_benign, pred_malignant)
            correct = "Yes" if true_label == predicted_label else "No"
            
            f.write(f"{i+1},{true_label},{pred_benign:.6f},{pred_malignant:.6f},{predicted_label},{confidence:.6f},{correct}\n")
    
    print(f"{BOLD_GREEN}Successfully saved {len(y_true)} predictions{RESET}")
    
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
