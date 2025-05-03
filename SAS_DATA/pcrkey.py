# Check the first few values of PcrKey in each file
import pandas as pd
import os
import glob

directory = "/Users/rishabmandyam/Documents/VSCode/NEMSIS_Proj/SAS_DB"
csv_files = glob.glob(os.path.join(directory, "*.csv"))

for file in csv_files:
    filename = os.path.basename(file)
    df = pd.read_csv(file)
    
    # Check for both potential column name formats
    if b'PcrKey' in df.columns:
        key_col = b'PcrKey'
    elif "b'PcrKey'" in df.columns:
        key_col = "b'PcrKey'"
    else:
        print(f"{filename}: PcrKey column not found")
        continue
        
    print(f"{filename}: First 5 values of {key_col}")
    print(df[key_col].head())
    print(f"Data type: {df[key_col].dtype}")
    print("---")
