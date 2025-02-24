import os
import pandas as pd
import numpy as np
import pickle
import zipfile
import glob
import shutil
from sklearn.metrics import mean_squared_error
from kaggle.api.kaggle_api_extended import KaggleApi

# Define Kaggle dataset name
KAGGLE_COMPETITION = "optiver-realized-volatility-prediction"


def check_data_availability(data_folder = None):
    """
    Checks if data is available and if not present, downloads the data
    """
    print("🌍 Checking dataset availability...")

    if not os.path.exists(os.path.join(data_folder, "book_train.parquet")):
        print("⚠️ Dataset not found. Downloading from Kaggle...")
        download_kaggle_dataset(data_folder)
    else:
        print("✅ Dataset already present.")

    return

def download_kaggle_dataset(destination):
    """
    Downloads the Kaggle dataset if it's not already downloaded.
    """
    os.makedirs(destination, exist_ok=True)

    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Download dataset
    print("⏳ Downloading Kaggle dataset... This may take a while.")
    api.competition_download_files(KAGGLE_COMPETITION, path=destination)

    # Manually unzip the dataset
    zip_files = glob.glob(os.path.join(destination, "*.zip"))  # Find all ZIP files
    for zip_file in zip_files:
        print(f"📂 Unzipping {zip_file}...")
        with zipfile.ZipFile(zip_file, "r") as z:
            z.extractall(destination)  # Extract all contents
        os.remove(zip_file)  # Delete ZIP file after extraction

    print(f"✅ Dataset downloaded and extracted successfully to {destination}.")


def clean_directory(directory):
    """Remove all files and subdirectories except .gitkeep."""

    # Iterate through all files and folders inside the directory
    for file_path in glob.glob(os.path.join(directory, "*")):
        if os.path.basename(file_path) != ".gitkeep":  # Keep .gitkeep
            if os.path.isfile(file_path):
                os.remove(file_path)  # Delete files
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove non-empty directories


def assign_train_test_label(df, train_ratio=0.8):
    """
    Adds a 'train_test' column to the DataFrame,
    assigning 'train' to 80% of rows per stock_id and 'test' to the rest.
    """

    # Sort by stock_id and time_id to ensure stable ordering
    df = df.sort_values(by=["stock_id", "time_id"]).reset_index(drop=True)

    # Assign a row number within each stock_id
    df["row_num"] = df.groupby("stock_id").cumcount() + 1  # Start from 1

    # Get max row number per stock_id
    df["total_rows"] = df.groupby("stock_id")["row_num"].transform("max")

    # Compute row fraction
    df["row_fraction"] = df["row_num"] / df["total_rows"]

    # Assign train/test based on row fraction
    df["train_test"] = np.where(df["row_fraction"] <= train_ratio, "train", "test")

    # Drop helper columns
    df.drop(columns=["row_num", "total_rows", "row_fraction"], inplace=True)

    return df


def save_train_test_split(df, output_dir, train_ratio = 0.8):
    """
    Splits a DataFrame into train and test sets and saves them as CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Apply the train-test labeling
    df = assign_train_test_label(df, train_ratio)

    # Split into separate DataFrames
    train_df = df[df["train_test"] == "train"].drop(columns=["train_test"])
    test_df = df[df["train_test"] == "test"].drop(columns=["train_test"])

    # Define file paths
    train_csv_path = os.path.join(output_dir, "train.csv")
    test_csv_path = os.path.join(output_dir, "test.csv")
    # old_train = pd.read_csv(train_csv_path)
    # print(f"If old and new df is same: {(old_train == train_df).all().all()}")

    # Save train and test datasets separately
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    print(f"✅ Training data saved to {train_csv_path}")
    print(f"✅ Testing data saved to {test_csv_path}")
