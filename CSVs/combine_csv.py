import pandas as pd
import os

def combine_csv(files, output_file):
    """
    Combine multiple CSV files with the same columns into one CSV file.
    
    Parameters:
        files (list of str): List of paths to the CSV files to combine.
        output_file (str): Path to save the combined CSV file.
    
    Returns:
        None
    """
    # List to store dataframes
    dataframes = []
    
    # Load each CSV file into a DataFrame and append it to the list
    for file in files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            dataframes.append(df)
        else:
            print(f"File not found: {file}")
    
    # Combine all dataframes into one
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV saved to {output_file}")


if __name__ == "__main__":
    csv_names = ["CSVs/wene.csv", "CSVs/massi.csv", "CSVs/julien.csv"]
    combine_csv(csv_names, "combined.csv")
    