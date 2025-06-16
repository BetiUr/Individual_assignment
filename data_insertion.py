import os
import pandas as pd
import re

def load_data(directory_path):
    """
    Loads all 2021 Divvy CSV files in the specified directory into a dictionary of DataFrames.
    Keys are month numbers as strings, e.g., '01', '02', ..., '12'.
    """
    files = [file for file in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, file))]
    dataframes = {}

    for file in files:
        match = re.search(r'2021(\d{2})-divvy-tripdata\.csv', file)
        if match:
            month = match.group(1)
            file_path = os.path.join(directory_path, file)
            try:
                df = pd.read_csv(file_path)
                dataframes[month] = df
                print(f"[INFO] Loaded {file} as month {month}")
            except Exception as e:
                print(f"[ERROR] Failed to load {file}: {e}")

    if not dataframes:
        print("[WARNING] No valid files found in directory.")
    elif len(dataframes) == 12:
        print("[SUCCESS] All 12 months of data have been successfully loaded.")
    else:
        print(f"[INFO] Loaded {len(dataframes)} months of data. Expected 12.")

    return dataframes
