import pandas as pd
import os
import glob

# Directory containing your CSV files
directory = "/Users/rishabmandyam/Documents/VSCode/NEMSIS_Proj/SAS_DB"  # Replace with your actual folder path

# Get all CSV files in the directory
csv_files = glob.glob(os.path.join(directory, "*.csv"))

# Initialize an empty dictionary to store dataframes
dfs = {}

# Read each CSV file into a dataframe and store in the dictionary
for file in csv_files:
    # Extract filename without path
    filename = os.path.basename(file)
    # Read the CSV file
    df = pd.read_csv(file)
    # Store in dictionary with filename as key
    dfs[filename] = df
    print(f"Loaded {filename} with {len(df)} rows and {len(df.columns)} columns")

# Check if all dataframes have the column b'PcrKey'
for filename, df in dfs.items():
    if b'PcrKey' not in df.columns and "b'PcrKey'" not in df.columns:
        print(f"Warning: {filename} does not have column b'PcrKey' or \"b'PcrKey'\"")

# Start with the first dataframe
base_df_name = list(dfs.keys())[0]
merged_df = dfs[base_df_name]
print(f"Starting with {base_df_name} as base dataframe")

# Merge with all other dataframes
for filename, df in dfs.items():
    if filename == base_df_name:
        continue  # Skip the base dataframe
    
    try:
        # Try the literal column name first
        if b'PcrKey' in merged_df.columns and b'PcrKey' in df.columns:
            merged_df = pd.merge(merged_df, df, on=b'PcrKey', how='outer')
        # Try the string representation of the column name
        elif "b'PcrKey'" in merged_df.columns and "b'PcrKey'" in df.columns:
            merged_df = pd.merge(merged_df, df, on="b'PcrKey'", how='outer')
        else:
            print(f"Skipping {filename} due to missing PcrKey column")
            continue
        
        print(f"Merged {filename}, resulting dataframe has {len(merged_df)} rows")
    except Exception as e:
        print(f"Error merging {filename}: {str(e)}")

# Save the merged dataframe
output_path = os.path.join("/Users/rishabmandyam/Documents/VSCode/NEMSIS_Proj", "data.csv")
merged_df.to_csv(output_path, index=False)
print(f"Merged data saved to {output_path}")