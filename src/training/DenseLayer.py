import numpy as np

class DenseLayer:
    """
    Fully connected layer (Dense layer) for neural network.
    """
    def __init__(self, input_size, output_size, activation='relu', 
                 weights_initializer='he_uniform', use_dropout=False, dropout_rate=0.0):
        """
        Args:
            input_size: Number of input features
            output_size: Number of neurons in this layer
            activation: Activation function ('relu', 'sigmoid', 'softmax')
            weights_initializer: Weight initialization method ('he_uniform', 'xavier')
            use_dropout: Whether to apply dropout
            dropout_rate: Dropout probability
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        
        # Initialize weights and biases
        self.weights = self._initialize_weights(weights_initializer)
        self.biases = np.zeros((1, output_size))
        
        # Cache for backpropagation
        self.input = None
        self.z = None  # Pre-activation
        self.output = None  # Post-activation
        self.dropout_mask = None
        
        # Gradients
        self.dW = None
        self.db = None
    
    def _initialize_weights(self, method):
        """Initialize weights using specified method."""
        if method == 'he_uniform':
            # He initialization for ReLU
            limit = np.sqrt(6.0 / self.input_size)
            return np.random.uniform(-limit, limit, (self.input_size, self.output_size))
        
        elif method == 'xavier':
            # Xavier initialization for sigmoid/tanh
            limit = np.sqrt(6.0 / (self.input_size + self.output_size))
            return np.random.uniform(-limit, limit, (self.input_size, self.output_size))
        
        else:
            # Default: small random values
            return np.random.randn(self.input_size, self.output_size) * 0.01
    
    def forward(self, X, training=True):
        """
        Forward pass through the layer.
        
        Args:
            X: Input data of shape (batch_size, input_size)
            training: Whether in training mode (for dropout)
            
        Returns:
            Output of shape (batch_size, output_size)
        """
        self.input = X
        
        # Linear transformation: Z = XW + b
        self.z = np.dot(X, self.weights) + self.biases
        
        # Apply activation function
        self.output = self._activate(self.z)
        
        # Apply dropout during training
        if self.use_dropout and training:
            self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, 
                                                   size=self.output.shape)
            self.output = self.output * self.dropout_mask / (1 - self.dropout_rate)
        
        return self.output
    
    # Activation functions
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip to avoid overflow
    
    def _relu(self, z):
        return np.maximum(0, z)
    
    def _softmax(self, z):
        # Subtract max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _activate(self, z):
        """Apply activation function."""
        if self.activation_name == 'relu':
            return self._relu(z)
        
        elif self.activation_name == 'sigmoid':
            return self._sigmoid(z)
        
        elif self.activation_name == 'softmax':
            return self._softmax(z)
        
        else:
            return z  # Linear activation
    
    def backward(self, dA, learning_rate):
        """
        Backward pass through the layer.
        
        Args:
            dA: Gradient of loss with respect to output of this layer
            learning_rate: Learning rate for weight updates
            
        Returns:
            Gradient with respect to input (for previous layer)
        """
        # Apply dropout mask to gradient
        if self.use_dropout and self.dropout_mask is not None:
            dA = dA * self.dropout_mask / (1 - self.dropout_rate)
        
        # Calculate gradient of activation function
        dZ = self._activation_gradient(dA)
        
        # Calculate gradients
        batch_size = self.input.shape[0]
        self.dW = np.dot(self.input.T, dZ) / batch_size
        self.db = np.sum(dZ, axis=0, keepdims=True) / batch_size
        
        # Gradient for previous layer
        dA_prev = np.dot(dZ, self.weights.T)
        
        # Update weights and biases
        self.weights -= learning_rate * self.dW
        self.biases -= learning_rate * self.db
        
        return dA_prev
    
    # Gradient activation
    def _relu_gradient(self, dA):
        """Gradient of ReLU activation."""
        dZ = dA.copy()
        dZ[self.z <= 0] = 0
        return dZ
    
    def _sigmoid_gradient(self, dA):
        """Gradient of sigmoid activation."""
        sig = self.output
        return dA * sig * (1 - sig)
    
    def _softmax_gradient(self, dA):
        """Gradient of softmax activation (simplified with cross-entropy)."""
        return dA
    
    def _activation_gradient(self, dA):
        """Calculate gradient of activation function."""
        if self.activation_name == 'relu':
            return self._relu_gradient(dA)
        
        elif self.activation_name == 'sigmoid':
            return self._sigmoid_gradient(dA)
        
        elif self.activation_name == 'softmax':
            return self._softmax_gradient(dA)
        
        else:
            return dA  # Linear
    
    def __repr__(self):
        return (f"DenseLayer(input={self.input_size}, output={self.output_size}, "
                f"activation={self.activation_name})")