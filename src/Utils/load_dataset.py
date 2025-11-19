import pandas as pd
import os


def load_dataset(data_path):
    """
    Load a dataset from a CSV file.
    
    Args:
        data_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing the dataset
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If the file is not a CSV file
        Exception: If there's an error reading the file
    """
    # Verify if the file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File '{data_path}' does not exist.")
    
    # Verify if it's a CSV file
    if not data_path.endswith('.csv'):
        raise ValueError(f"File '{data_path}' is not a CSV file.")
    
    try:
        # Load the CSV file
        df = pd.read_csv(data_path, header=None)

        # List of 10 base features
        features = [
            'radius', 'texture', 'perimeter', 'area', 'smoothness',
            'compactness', 'concavity', 'concave points', 'symmetry',
            'fractal_dimension'
        ]

        # List of 3 descriptive statistics
        stats = ['mean', 'se', 'worst']

        # 1. Build the 30 feature names (e.g., radius_mean, radius_se, ...)
        feature_names = []
        for stat in stats:
            for feature in features:
                feature_names.append(f'{feature}_{stat}')

        # The first two fixed columns
        fixed_columns = ['ID', 'diagnosis']

        # 2. Combine all 32 names in the correct order
        new_column_names = fixed_columns + feature_names

        # 3. Assign the new names onto the columns. This step Preserves all data rows.
        df.columns = new_column_names

        return df
    
    except Exception as e:
        raise Exception(f"Error reading CSV file: {e}")

