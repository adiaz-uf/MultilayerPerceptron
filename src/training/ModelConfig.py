import yaml
import os
from typing import List, Dict, Any

class ModelConfig:
    def __init__(self, config_path: str):
        """
        Central configuration handler.
        Parses YAML, validates types, and flattening structure for easier access.
        
        Args:
            config_path: Path to the .yaml configuration file.
        """
        self.config_path = config_path
        self._config = self._load_yaml()
        
        # 1. Architecture Configs
        model_conf = self._config.get('model_config', {})
        arch = model_conf.get('architecture', {})
        
        self.model_name: str = model_conf.get('model_name', 'unnamed_model')
        self.model_version: str = model_conf.get('model_version', 'x.x.x')
        
        # Validation: Input dim is critical for matrix shapes
        if 'input_dim' not in arch:
            raise ValueError("CRITICAL: 'input_dim' missing in architecture config.")
        self.input_dim: int = int(arch['input_dim'])
        
        self.hidden_layers: List[int] = arch.get('hidden_layers', [24, 24])
        self.use_batch_norm: bool = arch.get('use_batch_norm', False)
        self.activation_fn: str = arch.get('activation_fn', 'sigmoid')
        
        # 2. Class & Output Dimensions
        self.classes: List[str] = model_conf.get('classes', [])
        if not self.classes:
            raise ValueError("CRITICAL: No classes defined in config.")
            
        # Output is N neurons (Softmax).
        self.output_dim: int = len(self.classes)
        
        # 3. Training Hyperparameters
        train_params = self._config.get('training_params', {})
        
        self.epochs: int = train_params.get('epochs', 100)
        self.batch_size: int = train_params.get('batch_size', 8)
        self.test_size: float = train_params.get('test_size', 0.2)
        self.random_state: int = train_params.get('random_state', 42)
        
        # Dropout Logic: Check if enabled in architecture, then get rate from training params
        use_dropout = arch.get('use_dropout_rate', False)
        dropout_val = train_params.get('dropout_rate', 0.0)
        self.dropout_rate: float = dropout_val if use_dropout else 0.0
        
        # 4. Optimizer Settings
        optim = train_params.get('optimizer', {})
        self.optimizer_name: str = optim.get('name', 'SGD')
        self.learning_rate: float = float(optim.get('learning_rate', 0.01))
        self.weight_decay: float = float(optim.get('weight_decay', 0.0))
        
        # 5. Loss Function Settings
        loss_conf = train_params.get('loss_function', {})
        self.use_pos_weight: bool = loss_conf.get('use_pos_weight', False)
        
        # 6. Early Stopping Settings
        early_stop_conf = train_params.get('early_stopping', {})
        self.early_stopping_patience: int = early_stop_conf.get('patience', 10)
        self.early_stopping_delta: float = early_stop_conf.get('delta', 0.001)

    def _load_yaml(self) -> Dict[str, Any]:
        """Safely loads the YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at: {self.config_path}")
            
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

    def __repr__(self):
        """String representation for debugging."""
        return (f"ModelConfig(input={self.input_dim}, "
                f"layers={self.hidden_layers}, "
                f"optimizer={self.optimizer_name}, "
                f"lr={self.learning_rate}, "
                f"dropout={self.dropout_rate})")
