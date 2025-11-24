import sys
import os
import glob
import argparse
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.training.ModelConfig import ModelConfig
from src.training.NeuralNetwork import NeuralNetwork
from src.Utils import BOLD_CYAN, BOLD_GREEN, BOLD_YELLOW, BOLD_RED, RESET


def find_all_models(models_dir='models'):
    """
    Find all .npz model files in the specified directory.
    """
    pattern = os.path.join(models_dir, '*.npz')
    model_files = sorted(glob.glob(pattern))
    return model_files


def load_model_history(model_path, config_dir='model_config'):
    """
    Load a trained model and extract its training history.
    Automatically detects the config file based on model version.
    """
    try:
        # Try to infer config from model name
        # e.g., breast_cancer_diagnosis_v1.1.0.npz -> BreastCancerDiagnosis_config_v110.yaml
        model_name = os.path.basename(model_path)
        
        # Extract version (e.g., v1.1.0 -> v110)
        import re
        version_match = re.search(r'v(\d+)\.(\d+)\.(\d+)', model_name)
        if version_match:
            v1, v2, v3 = version_match.groups()
            config_version = f"v{v1}{v2}{v3}"
            config_path = os.path.join(config_dir, f"BreastCancerDiagnosis_config_{config_version}.yaml")
            
            # If specific config doesn't exist, try default
            if not os.path.exists(config_path):
                config_path = os.path.join(config_dir, "BreastCancerDiagnosis_config_v100.yaml")
        else:
            # Fallback to default config
            config_path = os.path.join(config_dir, "BreastCancerDiagnosis_config_v100.yaml")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        config = ModelConfig(config_path)
        
        # Initialize neural network
        model = NeuralNetwork(config)
        
        # Load model
        model.load_model(model_path)
        
        label = os.path.basename(model_path).replace('.npz', '')
        
        # Check if history has data
        if not model.history['train_loss'] or len(model.history['train_loss']) == 0:
            print(f"{BOLD_YELLOW}Warning: Model has no training history{RESET}")
            return None, None, None
        
        # Extract key config info for summary
        config_info = {
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'hidden_layers': config.hidden_layers,
            'activation': config.activation_fn,
            'epochs': len(model.history['train_loss'])
        }
        
        return model.history, label, config_info
    
    except Exception as e:
        print(f"{BOLD_RED}✗ Error loading {model_path}: {e}{RESET}")
        return None, None, None


def plot_comparison(models_data, output_path='plots/all_trainings_comparison.png'):
    """
    Plot multiple training runs on the same graph.
    
    Args:
        models_data: List of tuples (history, label, config_info)
        output_path: Path where to save the comparison plot
    """
    # Create plots directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Define colors for different models (extended palette)
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 
              'gray', 'olive', 'cyan', 'magenta', 'lime', 'navy', 'teal', 
              'coral', 'gold', 'indigo', 'maroon', 'turquoise', 'salmon']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot Loss curves
    for i, (history, label, config_info) in enumerate(models_data):
        color = colors[i % len(colors)]
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Training loss (solid line, thinner)
        ax1.plot(epochs, history['train_loss'], 
                color=color, linestyle='-', linewidth=1.5,
                label=f'{label} (Train)', alpha=0.6)
        
        # Validation loss (dashed line, thicker)
        ax1.plot(epochs, history['val_loss'], 
                color=color, linestyle='--', linewidth=2.5,
                label=f'{label} (Val)', alpha=0.9)
    
    ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax1.set_title(f'Training vs Validation Loss - {len(models_data)} Models', 
                  fontsize=15, fontweight='bold')
    if len(models_data) <= 5:
        ax1.legend(loc='best', fontsize=8, ncol=1)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(bottom=0)
    
    # Plot Accuracy curves
    for i, (history, label, config_info) in enumerate(models_data):
        color = colors[i % len(colors)]
        epochs = range(1, len(history['train_acc']) + 1)
        
        # Training accuracy (solid line, thinner)
        ax2.plot(epochs, history['train_acc'], 
                color=color, linestyle='-', linewidth=1.5,
                label=f'{label} (Train)', alpha=0.6)
        
        # Validation accuracy (dashed line, thicker)
        ax2.plot(epochs, history['val_acc'], 
                color=color, linestyle='--', linewidth=2.5,
                label=f'{label} (Val)', alpha=0.9)
    
    ax2.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax2.set_title(f'Training vs Validation Accuracy - {len(models_data)} Models', 
                  fontsize=15, fontweight='bold')
    if len(models_data) <= 5:
        ax2.legend(loc='best', fontsize=8, ncol=1)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    print(f"\n{BOLD_GREEN}✓ Comparison plot saved to: {output_path}{RESET}")


def print_summary_table(models_data):
    """Print a formatted summary table of all models."""
    
    # Check if there are any models with data
    if not models_data:
        print(f"\n{BOLD_RED}✗ No models with training history found{RESET}")
        return
    
    print(f"\n{BOLD_CYAN}{'='*100}")
    print(f"{'MODEL COMPARISON SUMMARY':^100}")
    print(f"{'='*100}{RESET}\n")
    
    # Header
    header = f"{'Model':<30} {'Epochs':>7} {'Batch':>6} {'LR':>8} {'Layers':<15} {'Final Val Loss':>14} {'Final Val Acc':>14}"
    print(header)
    print("-" * 100)
    
    # Sort by validation accuracy (descending)
    sorted_data = sorted(models_data, 
                        key=lambda x: x[0]['val_acc'][-1] if x[0]['val_acc'] else 0, 
                        reverse=True)
    
    # Rows
    for history, label, config_info in sorted_data:
        final_val_loss = history['val_loss'][-1]
        final_val_acc = history['val_acc'][-1]
        
        layers_str = str(config_info['hidden_layers'])
        if len(layers_str) > 15:
            layers_str = layers_str[:12] + "..."
        
        # Truncate long model names
        display_label = label if len(label) <= 30 else label[:27] + "..."
        
        # Color code based on performance
        if final_val_acc >= 0.98:
            color = BOLD_GREEN
        elif final_val_acc >= 0.95:
            color = BOLD_CYAN
        else:
            color = BOLD_RED
        
        row = f"{color}{display_label:<30}{RESET} {config_info['epochs']:>7} {config_info['batch_size']:>6} {config_info['learning_rate']:>8.5f} {layers_str:<15} {final_val_loss:>14.4f} {final_val_acc:>13.2%}"
        print(row)
    
    print("-" * 100)
    
    # Best model highlight
    best_model = sorted_data[0]
    best_history, best_label, best_config = best_model
    print(f"\n{BOLD_GREEN}BEST MODEL: {best_label}{RESET}")
    print(f"   Validation Accuracy: {best_history['val_acc'][-1]:.4f} ({best_history['val_acc'][-1]*100:.2f}%)")
    print(f"   Validation Loss:     {best_history['val_loss'][-1]:.4f}")
    print(f"   Configuration:       Batch={best_config['batch_size']}, LR={best_config['learning_rate']}, Layers={best_config['hidden_layers']}")
    print()


def main():
    """Main entry point for comparing all training runs."""
    parser = argparse.ArgumentParser(
        description='Compare ALL training runs in the models directory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Directory containing model files (default: models)')
    
    parser.add_argument('--config-dir', type=str, 
                       default='model_config',
                       help='Directory containing configuration files (default: model_config)')
    
    parser.add_argument('--output', type=str, 
                       default='plots/all_trainings_comparison.png',
                       help='Output path for the comparison plot (default: plots/all_trainings_comparison.png)')
    
    args = parser.parse_args()
    
    print(f"{BOLD_CYAN}{'='*100}")
    print(f"{'ALL TRAININGS COMPARISON TOOL':^100}")
    print(f"{'='*100}{RESET}\n")
    
    # Find all models
    print(f"{BOLD_CYAN}Searching for models in: {args.models_dir}{RESET}")
    model_files = find_all_models(args.models_dir)
    
    if not model_files:
        print(f"{BOLD_RED}No model files (.npz) found in {args.models_dir}{RESET}")
        return 1
    
    print(f"{BOLD_GREEN}Found {len(model_files)} model(s){RESET}\n")
    
    # Verify config directory exists
    if not os.path.exists(args.config_dir):
        print(f"{BOLD_RED}Config directory not found: {args.config_dir}{RESET}")
        return 1
    
    # Load all models
    print(f"{BOLD_CYAN}Loading models...{RESET}\n")
    models_data = []
    
    for i, model_path in enumerate(model_files, 1):
        print(f"  [{i}/{len(model_files)}] Loading {os.path.basename(model_path)}...", end=' ')
        history, label, config_info = load_model_history(model_path, args.config_dir)
        
        if history is not None:
            models_data.append((history, label, config_info))
            print(f"{BOLD_GREEN}{len(history['train_loss'])} epochs{RESET}")
        else:
            print(f"{BOLD_RED}Failed{RESET}")
    
    if not models_data:
        print(f"\n{BOLD_RED}No valid models loaded{RESET}")
        return 1
    
    print(f"\n{BOLD_GREEN}Successfully loaded {len(models_data)} model(s){RESET}")
    
    # Create comparison plot
    print(f"\n{BOLD_CYAN}Generating comparison plot...{RESET}")
    plot_comparison(models_data, args.output)
    
    # Print summary table
    print_summary_table(models_data)
    
    print(f"{BOLD_CYAN}{'='*100}{RESET}")
    print(f"{BOLD_GREEN}Comparison completed successfully!{RESET}\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())