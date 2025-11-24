import os
import numpy as np
import pandas as pd
from ..split_data import split_dataset

class DataLoader:
    def __init__(self, config):
        """
        Handles data ingestion, cleaning, splitting, and normalization.
        Args:
            config: The ModelConfig object containing training_params (test_size, random_state).
        """
        self.config = config
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        # Standardization parameters (saved to apply to future inference data)
        self.mean = None
        self.std = None
        
        # Class weights for imbalanced datasets
        self.pos_weight = None

    def load_from_csv(self, file_path: str):
        """
        Reads the raw CSV, handles the specific format of the Breast Cancer dataset.
        Uses split_dataset to split the data into training and validation sets.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found at: {file_path}")

        # 1. Split the dataset using the external function
        test_size = self.config.test_size
        random_state = self.config.random_state
        train_path, validation_path = split_dataset(file_path, test_size, random_state)

        # 2. Load the split datasets
        self.X_train, self.y_train = self._load_raw_data(train_path)
        self.X_test, self.y_test = self._load_raw_data(validation_path)

        # 3. Normalize
        self._normalize()
        
        # 4. Calculate class weights if needed
        self._calculate_class_weights()

        print(f"Data Loaded Summary:")
        print(f"  Training samples: {self.X_train.shape[0]}")
        print(f"  Test samples:     {self.X_test.shape[0]}")
        print(f"  Features:         {self.X_train.shape[1]}")
        print(f"  Target Distribution (Train): {np.mean(self.y_train):.2%} Malignant")
        
        if self.pos_weight is not None:
            print(f"  Positive class weight: {self.pos_weight:.4f}")

    def _load_raw_data(self, file_path: str):
        """
        Helper to load and parse a CSV file into X and y.
        Expected format: ID, Diagnosis(M/B), Feature1, Feature2..., Feature30
        """
        raw_data = []
        targets = []

        with open(file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line: continue
            
            parts = line.split(',')
            
            # SKEPTICAL CHECK: Ensure the row has the expected number of columns
            # ID + Diagnosis + 30 features = 32 columns
            if len(parts) < 32:
                continue 

            # 1. Parse Target (Column 1) -> 'M' is Malignant (1), 'B' is Benign (0)
            # Column 0 is ID, we discard it as it introduces noise.
            diagnosis = 1 if parts[1] == 'M' else 0
            targets.append(diagnosis)

            # 2. Parse Features (Columns 2 to end)
            try:
                features = [float(x) for x in parts[2:]]
                raw_data.append(features)
            except ValueError:
                print(f"Warning: Could not parse line: {line}")
                continue

        X = np.array(raw_data, dtype=np.float32)
        y = np.array(targets, dtype=np.float32).reshape(-1, 1)
        return X, y

    def _normalize(self):
        """
        Calculate statistics on training data and normalize both train and test sets.
        """
        # Calculate Statistics ONLY on Training Data
        self.mean = np.mean(self.X_train, axis=0)
        self.std = np.std(self.X_train, axis=0)
        
        # Avoid division by zero
        self.std[self.std == 0] = 1e-8
        
        # Apply Transformation
        self.X_train = (self.X_train - self.mean) / self.std
        self.X_test = (self.X_test - self.mean) / self.std
    
    def _calculate_class_weights(self):
        """
        Calculate positive class weight for imbalanced datasets.
        - If pos_weight is defined in config (> 0), use that value
        - If pos_weight is 0 or not set, calculate automatically from training data
        pos_weight = number of negative samples / number of positive samples
        This helps the model focus more on the minority class.
        """
        if self.config.use_pos_weight:
            if self.config.pos_weight > 0.0:
                self.pos_weight = self.config.pos_weight
            else:
                # Calculate dynamically from training data distribution
                num_positive = np.sum(self.y_train == 1)
                num_negative = np.sum(self.y_train == 0)
                
                if num_positive > 0:
                    self.pos_weight = num_negative / num_positive
                else:
                    self.pos_weight = 1.0
        else:
            self.pos_weight = None
            

    def get_training_batch(self, batch_size=None):
        """
        Generator that yields mini-batches for the training loop.
        """
        if batch_size is None:
            batch_size = self.config.batch_size
            
        num_samples = self.X_train.shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices) # Shuffle at the start of each epoch
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_idx = indices[start_idx:end_idx]
            
            yield self.X_train[batch_idx], self.y_train[batch_idx]