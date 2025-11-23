import pandas as pd
import sys
import os

def split_dataset(data_path, test_size = 0.2, random_state = 42):
    """
    Split the dataset into training and validation sets.
    
    Args:
        data_path: Path to the CSV file
        test_size: Proportion of the dataset to include in the validation set
    """
    # Read the CSV file
    df = pd.read_csv(data_path, header=None)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Calculate split index
    split_index = int(len(df) * (1 - test_size))
    
    # Split the data
    train_df = df[:split_index]
    validation_df = df[split_index:]
    
    # Get the directory of the input file
    data_dir = os.path.dirname(data_path)
    
    # Save the split datasets
    train_path = os.path.join(data_dir, 'data_training.csv')
    validation_path = os.path.join(data_dir, 'data_validation.csv')
    
    train_df.to_csv(train_path, index=False, header=False)
    validation_df.to_csv(validation_path, index=False, header=False)
    
    print(f"Dataset split successfully!")
    print(f"Training set: {len(train_df)} samples -> {train_path}")
    print(f"Validation set: {len(validation_df)} samples -> {validation_path}")
    
    return train_path, validation_path

def get_user_input():
    while True:
        # 1. Get user input
        ratio_input = input("Please, enter test dataset size ratio (0.0 - 1.0): ").strip()

        try:
            # 2. Convert to float (non-numeric input will raise a ValueError)
            ratio = float(ratio_input)

            # 3. Validate range
            if ratio <= 0.0 or ratio >= 1.0:
                print("Please enter a value between 0.0 and 1.0 (exclusive)")
            else:
                return ratio

        except ValueError:
            # 4. Handle non-numeric input
            print("Invalid input. Please enter a numeric value for the ratio.")

def main():
    # Check if the file path was provided
    if len(sys.argv) < 2:
        print("Error: No file path provided.")
        print("Usage: python separate.py <path_to_csv_file>")
        return 1
    
    data_path = sys.argv[1]
    
    # Verify if the file exists
    if not os.path.exists(data_path):
        print(f"Error: File '{data_path}' does not exist.")
        return 1
    
    # Verify if it's a CSV file
    if not data_path.endswith('.csv'):
        print(f"Error: File '{data_path}' is not a CSV file.")
        return 1
    
    # Get test size ratio from user
    test_size = get_user_input()
    
    # Split the dataset
    try:
        split_dataset(data_path, test_size)
    except Exception as e:
        print(f"Error splitting dataset: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())