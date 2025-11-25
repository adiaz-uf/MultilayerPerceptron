import numpy as np
import os

from .DenseLayer import DenseLayer

class NeuralNetwork:
    """
    Multilayer Perceptron (Neural Network) implementation.
    """
    def __init__(self, config):
        """
        Initialize neural network from config.
        
        Args:
            config: ModelConfig object containing architecture details
        """
        self.config = config
        self.layers = []
        self._build_network()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

        # Early stopping state
        self.best_weights = None
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.stopped_epoch = None
    
    def _build_network(self):
        """Build network architecture from config."""
        # Input layer size
        prev_size = self.config.input_dim
        
        # Hidden layers
        for layer_size in self.config.hidden_layers:
            layer = DenseLayer(
                input_size=prev_size,
                output_size=layer_size,
                activation=self.config.activation_fn.lower(),
                weights_initializer='he_uniform',
                use_dropout=self.config.dropout_rate > 0,
                dropout_rate=self.config.dropout_rate
            )
            self.layers.append(layer)
            prev_size = layer_size
        
        # Output layer with softmax
        output_layer = DenseLayer(
            input_size=prev_size,
            output_size=self.config.output_dim,
            activation='softmax',
            weights_initializer='xavier',
            use_dropout=False
        )
        self.layers.append(output_layer)
        
        print(f"\nNeural Network Architecture:")
        print(f"  Input features: {self.config.input_dim}")
        for i, layer in enumerate(self.layers):
            print(f"  Layer {i+1}: {layer}")
    
    def forward(self, X, training=True):
        """
        Forward pass through entire network.
        
        Args:
            X: Input data of shape (batch_size, input_dim)
            training: Whether in training mode
            
        Returns:
            Output predictions of shape (batch_size, output_dim)
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output, training=training)
        return output
    
    def backward(self, y_true, y_pred):
        """
        Backward pass through entire network.
        
        Args:
            y_true: True labels of shape (batch_size, output_dim)
            y_pred: Predicted probabilities of shape (batch_size, output_dim)
        """
        # Gradient of loss with respect to output
        # For softmax + cross-entropy: dL/dy = y_pred - y_true
        dA = y_pred - y_true
        
        # Backpropagate through layers in reverse
        for layer in reversed(self.layers):
            dA = layer.backward(
                dA, 
                learning_rate=self.config.learning_rate,
                optimizer=self.config.optimizer_name,
                weight_decay=self.config.weight_decay
            )

    def binary_cross_entropy(self, y_true, y_pred):
        """
        Calculate binary cross-entropy loss.
        
        Args:
            y_true: True labels (0 or 1), shape (N, 1) or (N, 2) one-hot
            y_pred: Predicted probabilities, shape (N, 2) from softmax
            
        Returns:
            Binary cross-entropy loss value
        """
        # Convert one-hot to binary if needed
        if y_true.shape[1] == 2:
            y_true = y_true[:, 1].reshape(-1, 1)  # Take second column (Malignant)
        
        # Get probability of positive class (Malignant)
        if y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1].reshape(-1, 1)
        
        # Clip to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        
        # Binary cross-entropy formula
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        return loss
    
    def categorical_cross_entropy(self, y_true, y_pred, pos_weight=None):
        """
        Calculate categorical cross-entropy loss (Standard for Softmax).
        
        Args:
            y_true: True labels (one-hot encoded), shape (N, num_classes)
            y_pred: Predicted probabilities from Softmax, shape (N, num_classes)
            pos_weight: Optional weight for positive class (for imbalanced datasets)
        
        Returns:
            Categorical cross-entropy loss value
        """
        # Clip to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        
        # Standard categorical cross-entropy: -sum(y_true * log(y_pred))
        loss = -np.sum(y_true * np.log(y_pred), axis=1)
        
        # Apply positive class weighting if provided
        if pos_weight is not None and y_true.shape[1] == 2:
            # For binary classification with one-hot encoding
            # Apply weight to positive class (index 1)
            weights = np.ones(y_true.shape[0])
            weights[y_true[:, 1] == 1] = pos_weight
            loss = loss * weights
        
        return np.mean(loss)
    
    def accuracy(self, y_true, y_pred):
        """Calculate accuracy."""
        # Convert probabilities to class predictions
        predictions = (y_pred[:, 1] > 0.5).astype(int).reshape(-1, 1)
        true_labels = y_true[:, 1].reshape(-1, 1)
        return np.mean(predictions == true_labels)
    
    def check_early_stopping(self, current_val_loss, epoch):
        """
        Check if training should stop early based on validation loss.
        
        Args:
            val_loss: Current validation loss
            epoch: Current epoch number
            
        Returns:
            bool: True if training should stop, False otherwise
        """

        patience = self.config.early_stopping_patience

        if patience <= 0:
            return False
    
        delta = self.config.early_stopping_delta
        restore_weights = self.config.restore_best_weights

        if current_val_loss < (self.best_val_loss - delta):
            self.best_val_loss = current_val_loss
            self.best_epoch = epoch
            self.patience_counter = 0

            if restore_weights:
                self.best_weights = self._copy_weights()

            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= patience:

                if restore_weights:
                    self._restore_weights(self.best_weights)
                    print(f"\nRestoring model weights from epoch {self.best_epoch}")

                return True

        return False
    
    def _copy_weights(self):
        """Create a deep copy of all layer weights, biases and optimizer state."""
        weights_copy = []
        for layer in self.layers:
            layer_state = {
                'weights': layer.weights.copy(),
                'biases': layer.biases.copy()
            }
        
            # Save Adam optimizer state if it exists
            if layer.adam_m_w is not None:
                layer_state['adam_m_w'] = layer.adam_m_w.copy()
                layer_state['adam_v_w'] = layer.adam_v_w.copy()
                layer_state['adam_m_b'] = layer.adam_m_b.copy()
                layer_state['adam_v_b'] = layer.adam_v_b.copy()
                layer_state['adam_t'] = layer.adam_t
            
            weights_copy.append(layer_state)
        return weights_copy
    
    def _restore_weights(self, weights_copy):
        """Restore weights, biases, and optimizer state from a saved copy."""
        for i, layer_state in enumerate(weights_copy):
            self.layers[i].weights = layer_state['weights'].copy()
            self.layers[i].biases = layer_state['biases'].copy()
            
            # Restore Adam optimizer state if it was saved
            if 'adam_m_w' in layer_state:
                self.layers[i].adam_m_w = layer_state['adam_m_w'].copy()
                self.layers[i].adam_v_w = layer_state['adam_v_w'].copy()
                self.layers[i].adam_m_b = layer_state['adam_m_b'].copy()
                self.layers[i].adam_v_b = layer_state['adam_v_b'].copy()
                self.layers[i].adam_t = layer_state['adam_t']
    
    def __repr__(self):
        return f"NeuralNetwork(layers={len(self.layers)}, params={self.count_parameters()})"
    
    def count_parameters(self):
        """Count total number of trainable parameters."""
        total = 0
        for layer in self.layers:
            total += layer.weights.size + layer.biases.size
        return total
    
    def save_model(self, filepath, normalization_params=None):
        """
        Save model weights and architecture to .npz format.
        Structure:
        {
            'model_name': 'breast_cancer_diagnosis_v1.0.0',
            'input_dim': 30,
            'output_dim': 2,
            'hidden_layers': [24, 24, 24],
            'activation_fn': 'sigmoid',
            'dropout_rate': 0.2,
            'num_layers': 4,
            'norm_mean': array([...]),  # 30 valores
            'norm_std': array([...]),   # 30 valores
            'layer_0_weights': array(...),
            'layer_0_biases': array(...),
            'layer_0_activation': 'sigmoid',
            ...
        }
        
        Args:
            filepath: Path to save the model (e.g., 'models/model.npz')
            normalization_params: Dict with 'mean' and 'std' for data normalization
        """
        # Ensure .npz extension
        if not filepath.endswith('.npz'):
            filepath = filepath.replace('.npy', '').replace('.pt', '') + '.npz'
        
        # Prepare data dictionary
        save_dict = {
            'model_name': self.config.model_name,
            'input_dim': self.config.input_dim,
            'output_dim': self.config.output_dim,
            'hidden_layers': np.array(self.config.hidden_layers),
            'activation_fn': self.config.activation_fn,
            'dropout_rate': self.config.dropout_rate,
            'num_layers': len(self.layers),
        }
        
        # Add normalization parameters if provided
        if normalization_params:
            save_dict['norm_mean'] = normalization_params['mean']
            save_dict['norm_std'] = normalization_params['std']
        
        # Save weights and biases for each layer
        for i, layer in enumerate(self.layers):
            save_dict[f'layer_{i}_weights'] = layer.weights
            save_dict[f'layer_{i}_biases'] = layer.biases
            save_dict[f'layer_{i}_activation'] = np.array(layer.activation_name)
            save_dict[f'layer_{i}_input_size'] = layer.input_size
            save_dict[f'layer_{i}_output_size'] = layer.output_size
        
        # Save training history
        if self.history and any(self.history.values()):
            save_dict['history_train_loss'] = np.array(self.history['train_loss'])
            save_dict['history_val_loss'] = np.array(self.history['val_loss'])
            save_dict['history_train_acc'] = np.array(self.history['train_acc'])
            save_dict['history_val_acc'] = np.array(self.history['val_acc'])
        
        # Save to compressed .npz file
        np.savez_compressed(filepath, **save_dict)
        print(f"\n> Saving model '{filepath}' to disk...")
        print(f"  Total parameters: {self.count_parameters()}")
    
    def load_model(self, filepath):
        """
        Load model weights and architecture from .npz file.
        
        Args:
            filepath: Path to the saved model file
            
        Returns:
            normalization_params: Dict with 'mean' and 'std' if available
        """
        # Ensure .npz extension
        if not filepath.endswith('.npz'):
            filepath = filepath.replace('.npy', '').replace('.pt', '') + '.npz'
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load from .npz file
        data = np.load(filepath, allow_pickle=True)
        
        # Verify architecture matches
        if int(data['input_dim']) != self.config.input_dim:
            raise ValueError(f"Input dimension mismatch: expected {self.config.input_dim}, got {int(data['input_dim'])}")
        
        if int(data['output_dim']) != self.config.output_dim:
            raise ValueError(f"Output dimension mismatch: expected {self.config.output_dim}, got {int(data['output_dim'])}")
        
        num_layers = int(data['num_layers'])
        if num_layers != len(self.layers):
            raise ValueError(f"Layer count mismatch: expected {len(self.layers)}, got {num_layers}")
        
        # Load weights and biases for each layer
        for i in range(num_layers):
            self.layers[i].weights = data[f'layer_{i}_weights']
            self.layers[i].biases = data[f'layer_{i}_biases']
        
        # Load training history if available
        if 'history_train_loss' in data.files:
            self.history['train_loss'] = data['history_train_loss'].tolist()
            self.history['val_loss'] = data['history_val_loss'].tolist()
            self.history['train_acc'] = data['history_train_acc'].tolist()
            self.history['val_acc'] = data['history_val_acc'].tolist()
        
        print(f"\nModel loaded from '{filepath}'")
        
        # Extract normalization parameters if available
        normalization_params = None
        if 'norm_mean' in data.files and 'norm_std' in data.files:
            normalization_params = {
                'mean': data['norm_mean'],
                'std': data['norm_std']
            }
            print(f"Normalization parameters loaded")
        
        return normalization_params