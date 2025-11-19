import sys

from Utils.load_dataset import load_dataset


def train_model(train_data, val_data):
    """
    Train the neural network model.
    """

    # TODO: Implementar el entrenamiento del modelo
    return 0


def main():
    # Load datasets
    try:
        print("Loading training data...")
        train_df = load_dataset(data_path='data/data_training.csv')
        
        print(f"Training set shape: {train_df.shape}\n")
        
        print("Loading validation data...")
        val_df = load_dataset(data_path='data/data_validation.csv')
        
        print(f"Validation set shape: {val_df.shape}")
             
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
    
    return train_model(train_df, val_df)


if __name__ == '__main__':
    sys.exit(main())
