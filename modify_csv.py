import pandas as pd

def modify_last_two_columns(input_file, output_file):
    """
    Modify the last column to 'attack' and the second last column to 2 for all rows in a CSV file.

    :param input_file: Path to the input CSV file.
    :param output_file: Path to the output CSV file.
    """
    # Load the CSV file
    df = pd.read_csv('SWaT_Dataset_Normal_v1_modified.csv')

    # Get the names of the last two columns
    last_column = df.columns[-1]
    second_last_column = df.columns[-2]

    # Modify the last column to 'attack' and the second last column to 2
    df[last_column] = 'attack'
    df[second_last_column] = 2

    # Save the modified DataFrame to a new CSV file
    df.to_csv('SWaT_Dataset_Normal_v1_modified_test.csv', index=False)

    print(f"File saved as {output_file}")

# Example usage
input_file = 'path_to_your_input_file.csv'
output_file = 'path_to_your_output_file.csv'
modify_last_two_columns(input_file, output_file)