import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

from .ModelConfig import ModelConfig
from .DataLoader import DataLoader
from .NeuralNetwork import NeuralNetwork
from ..Utils import BOLD_RED, BOLD_GREEN, BOLD_YELLOW, RESET


def plot_learning_curves(model, config):
    """
    Plot and save training and validation loss curves.
    
    Args:
        model: NeuralNetwork object with training history
        config: ModelConfig object with model name
    """
    import os
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Create figure with subplots for loss and accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(model.history['train_loss']) + 1)
    
    # Plot loss
    ax1.plot(epochs, model.history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, model.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, model.history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, model.history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f'plots/{config.model_name.replace(".npz", "")}_learning_curves.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"\n{BOLD_GREEN}Learning curves saved to: {plot_filename}{RESET}")
    
    plt.close()


def train_model(config, dataset_path):
    """
    Train the neural network model.
    
    Args:
        config: ModelConfig object with training parameters
        dataset_path: Path to the dataset CSV file
    """
    print(f"Model Config: {config}\n")
    
    # Initialize DataLoader with config
    data_loader = DataLoader(config)
    
    data_loader.load_from_csv(dataset_path)
    
    # Convert labels to one-hot encoding for softmax
    # y is currently [0] or [1], we need [[1,0]] or [[0,1]]
    def to_one_hot(y):
        n_samples = y.shape[0]
        one_hot = np.zeros((n_samples, 2))
        one_hot[np.arange(n_samples), y.flatten().astype(int)] = 1
        return one_hot
    
    y_train = to_one_hot(data_loader.y_train)
    y_test = to_one_hot(data_loader.y_test)
    
    # Initialize Neural Network
    model = NeuralNetwork(config)
    
    # Get positive class weight if enabled
    pos_weight = data_loader.pos_weight if config.use_pos_weight is True else None
    if pos_weight:
        print(f"\nUsing positive class weighting: {pos_weight:.4f}")
    
    # Training loop
    print(f"\nStarting training for {config.epochs} epochs...")
    print(f"   Optimizer: {config.optimizer_name}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Weight decay: {config.weight_decay}\n")
    
    # Open log file for training metrics
    os.makedirs('models', exist_ok=True)
    log_filename = f'models/{config.model_name.replace(".npz", "")}_training_log.txt'
    log_file = open(log_filename, 'w')
    
    # Write header to log file
    log_file.write(f"Training Log: {config.model_name}\n")
    log_file.write(f"=" * 80 + "\n")
    log_file.write(f"Optimizer: {config.optimizer_name}\n")
    log_file.write(f"Batch size: {config.batch_size}\n")
    log_file.write(f"Learning rate: {config.learning_rate}\n")
    log_file.write(f"Weight decay: {config.weight_decay}\n")
    log_file.write(f"Epochs: {config.epochs}\n")
    log_file.write(f"=" * 80 + "\n\n")
    
    for epoch in range(config.epochs):
        # Training phase
        total_train_loss = 0.0
        num_samples = 0
        
        # Get batches using DataLoader's method
        for X_batch, y_batch_raw in data_loader.get_training_batch():
            # Convert batch labels to one-hot
            y_batch = to_one_hot(y_batch_raw)
            
            # Forward pass
            y_pred = model.forward(X_batch, training=True)
            
            # Calculate loss
            loss = model.categorical_cross_entropy(y_batch, y_pred, pos_weight=pos_weight)

            # Get the actual size of this specific batch
            current_batch_size = X_batch.shape[0]
            
            # Accumulate weighted loss
            total_train_loss += loss * current_batch_size
            num_samples += current_batch_size
            
            # Backward pass
            model.backward(y_batch, y_pred)
        
        # Calculate training metrics
        train_loss = total_train_loss / num_samples
        train_pred = model.forward(data_loader.X_train, training=False)
        train_acc = model.accuracy(y_train, train_pred)
        
        # Validation metrics
        val_pred = model.forward(data_loader.X_test, training=False)
        val_loss = model.categorical_cross_entropy(y_test, val_pred, pos_weight=pos_weight)
        val_acc = model.accuracy(y_test, val_pred)
        
        # Store history
        model.history['train_loss'].append(train_loss)
        model.history['val_loss'].append(val_loss)
        model.history['train_acc'].append(train_acc)
        model.history['val_acc'].append(val_acc)
        
        # Print progress to console
        print(f"{BOLD_YELLOW}epoch {epoch+1:02d}/{config.epochs} - "
              f"{BOLD_RED}loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
              f"{BOLD_GREEN}val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}{RESET}")
        
        # Write progress to log file
        log_file.write(f"epoch {epoch+1:02d}/{config.epochs} - "
                      f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
                      f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}\n")
        log_file.flush()  # Ensure it's written immediately

        # TODO: implement early stopping
    
    # Write final summary to log file
    log_file.write(f"\n" + "=" * 80 + "\n")
    log_file.write(f"TRAINING SUMMARY\n")
    log_file.write(f"=" * 80 + "\n")
    log_file.write(f"Final Training Loss:     {model.history['train_loss'][-1]:.4f}\n")
    log_file.write(f"Final Training Accuracy: {model.history['train_acc'][-1]:.4f} ({model.history['train_acc'][-1]*100:.2f}%)\n")
    log_file.write(f"Final Validation Loss:   {model.history['val_loss'][-1]:.4f}\n")
    log_file.write(f"Final Validation Accuracy: {model.history['val_acc'][-1]:.4f} ({model.history['val_acc'][-1]*100:.2f}%)\n")
    log_file.write(f"\n")
    log_file.write(f"Best Training Accuracy:     {max(model.history['train_acc']):.4f} (epoch {model.history['train_acc'].index(max(model.history['train_acc']))+1})\n")
    log_file.write(f"Best Validation Accuracy:   {max(model.history['val_acc']):.4f} (epoch {model.history['val_acc'].index(max(model.history['val_acc']))+1})\n")
    log_file.write(f"Lowest Training Loss:       {min(model.history['train_loss']):.4f} (epoch {model.history['train_loss'].index(min(model.history['train_loss']))+1})\n")
    log_file.write(f"Lowest Validation Loss:     {min(model.history['val_loss']):.4f} (epoch {model.history['val_loss'].index(min(model.history['val_loss']))+1})\n")
    log_file.write(f"=" * 80 + "\n")
    
    # Close log file
    log_file.close()
    
    print(f"\n{BOLD_GREEN}Training completed!{RESET}")
    print(f"Training log saved to: {log_filename}")
    
    # Save model with normalization parameters
    
    os.makedirs('models', exist_ok=True)
    model_path = f'models/{config.model_name}'
    
    normalization_params = {
        'mean': data_loader.mean,
        'std': data_loader.std
    }
    
    model.save_model(model_path, normalization_params=normalization_params)
    
    # Plot learning curves
    plot_learning_curves(model, config)
    
    return 0


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train a Multilayer Perceptron for Breast Cancer Diagnosis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='data/data.csv',
        help='Path to the dataset CSV file (default: data/data.csv)'
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
        import os
        if not os.path.exists(args.dataset):
            print(f"{BOLD_RED}Error: Dataset file not found: {args.dataset}{RESET}")
            return 1
        
        if not os.path.exists(args.config):
            print(f"{BOLD_RED}Error: Config file not found: {args.config}{RESET}")
            return 1
        
        print(f"{BOLD_GREEN}Configuration:{RESET}")
        print(f"  Dataset: {args.dataset}")
        print(f"  Config:  {args.config}\n")
        
        # Load model configuration
        model_config = ModelConfig(args.config)
        
        # Train the model
        return train_model(model_config, args.dataset)
             
    except Exception as e:
        print(f"{BOLD_RED}Error during training: {e}{RESET}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())