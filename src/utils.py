import os
import pandas as pd
import numpy as np
import pickle
import zipfile
import glob
from sklearn.metrics import mean_squared_error
from kaggle.api.kaggle_api_extended import KaggleApi

# Define Kaggle dataset name
KAGGLE_COMPETITION = "optiver-realized-volatility-prediction"

def detect_environment():
    """
    Detects whether the script is running in Kaggle or Local.
    """
    if os.path.exists("/kaggle/input/"):
        return "kaggle"
    return "local"

def get_input_data_path(env, local_path = None):
    """
    Returns the correct data path based on execution environment.
    """
    # Kaggle environment
    if env == "kaggle":
        kaggle_path = "/kaggle/input/optiver-realized-volatility-prediction/"
        print(f"✅ Running inside Kaggle. Using dataset from {kaggle_path}")
        return kaggle_path

    # Local environment
    if env == "local" and local_path is None:
        local_path = "data/"
        print(f"Creating the data folder {local_path}")

    print("🌍 Running in Local. Checking dataset availability...")

    if not os.path.exists(os.path.join(local_path, "book_train.parquet")):
        print("⚠️ Dataset not found. Downloading from Kaggle...")
        download_kaggle_dataset(local_path)
    else:
        print("✅ Dataset already present in system.")

    return local_path

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


def assign_train_test_label(df, train_ratio=0.8):
    """
    Adds a 'train_test' column to the DataFrame,
    assigning 'train' to 80% of rows per stock_id and 'test' to the rest.
    """

    # Assign a row number within each stock_id
    df["row_num"] = df.groupby("stock_id").cumcount()

    # Normalize row number by total rows per stock_id
    df["row_fraction"] = df["row_num"] / df.groupby("stock_id")["row_num"].transform("max")

    # Assign train/test based on the fraction
    df["train_test"] = np.where(df["row_fraction"] <= train_ratio, "train", "test")

    # Drop helper columns
    df.drop(columns=["row_num", "row_fraction"], inplace=True)

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

    # Save train and test datasets separately
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    print(f"✅ Training data saved to {train_csv_path}")
    print(f"✅ Testing data saved to {test_csv_path}")
