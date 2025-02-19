import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pyarrow.parquet as pq
import pyarrow as pa


### ===== Helper Functions ===== ###

def log_return(list_stock_prices):
    """Compute the log return of stock prices."""
    return np.log(list_stock_prices).diff()


def realized_volatility(series):
    """Compute realized volatility from a series of log returns."""
    return np.sqrt(np.sum(series ** 2))


def get_wap_feature(data):
    """Calculate Weighted Average Price (WAP)."""
    data['wap1'] = (data['bid_price1'] * data['ask_size1'] + data['ask_price1'] * data['bid_size1']) / (
            data['bid_size1'] + data['ask_size1'])
    data['wap2'] = (data['bid_price2'] * data['ask_size2'] + data['ask_price2'] * data['bid_size2']) / (
            data['bid_size2'] + data['ask_size2'])
    return data


def get_log_return_feature(data):
    """Calculate log returns."""
    data['wap1_shift'] = data.groupby('time_id')['wap1'].shift(1)
    data['wap2_shift'] = data.groupby('time_id')['wap2'].shift(1)
    data = data[~data['wap1_shift'].isnull()]
    data = data[~data['wap2_shift'].isnull()]
    data['log_return1'] = np.log(data['wap1'] / data['wap1_shift'])
    data['log_return2'] = np.log(data['wap2'] / data['wap2_shift'])
    return data


def get_book_volatility_feature(data):
    """Compute volatility for WAP log returns."""
    data = data.groupby(['stock_id', 'time_id'])[['log_return1', 'log_return2']].agg(realized_volatility).reset_index()
    data.rename(columns={'log_return1': 'volatility1', 'log_return2': 'volatility2'}, inplace=True)
    return data


### ===== Feature Engineering Functions ===== ###

def generate_features_book_seconds(data, features={'wap', 'log_return'}):
    """Generate book-level features at stock-time-seconds level."""
    if 'wap' in features:
        data = get_wap_feature(data)
    if 'log_return' in features:
        data = get_log_return_feature(data)
    return data


def generate_features_book_timeid(data, features={'volatility'}):
    """Generate book-level features at stock-time_id level."""
    if 'volatility' in features:
        data = get_book_volatility_feature(data)
    return data


def generate_features_trade_seconds(data):
    """(Placeholder) Generate trade-level features at stock-time-seconds level."""
    return data  # Extend with actual trade features


def generate_features_trade_timeid(data):
    """(Placeholder) Generate trade-level features at stock-time_id level."""
    return data  # Extend with actual trade features


### ===== Generalized Feature Application Function ===== ###

def _apply_features(input_path, output_path, feature_func, features=None):
    """Applies a given feature function to all parquet files in the directory
       and saves a single output file partitioned by stock_id."""

    os.makedirs(output_path, exist_ok=True)  # Ensure output directory exists

    for filename in tqdm(os.listdir(input_path)):
        file_path = os.path.join(input_path, filename)

        df_temp = pd.read_parquet(file_path)
        stock_id = int(filename.split('=')[1].split('.')[0])
        df_temp['stock_id'] = stock_id

        # Apply selected feature function
        df_temp = feature_func(df_temp, features) if features else feature_func(df_temp)

        df_temp.to_parquet(output_path, index=False, engine="pyarrow", partition_cols=["stock_id"])


### ===== Main Function to Apply Feature Engineering (Auto-Detect Paths) ===== ###
def apply_seconds_feature_engineering(input_data_folder, output_data_folder, book_seconds_features=None,
                              trade_seconds_features=None):
    book_input_path = os.path.join(input_data_folder, "book_train.parquet")
    trade_input_path = os.path.join(input_data_folder, "trade_train.parquet")

    book_output_seconds_path = os.path.join(output_data_folder, "book_features_seconds.parquet")
    trade_output_seconds_path = os.path.join(output_data_folder, "trade_features_seconds.parquet")

    # Book Features - Seconds Level
    if book_seconds_features:
        _apply_features(book_input_path, book_output_seconds_path, generate_features_book_seconds,
                        book_seconds_features)

    # Trade Features - Seconds Level
    if trade_seconds_features:
        _apply_features(trade_input_path, trade_output_seconds_path, generate_features_trade_seconds)


def apply_timeid_feature_engineering(input_data_folder, output_data_folder, book_timeid_features=None,
                                     trade_timeid_features=None):
    """
    Apply feature engineering to book and trade data.
    Automatically detects whether running in Kaggle or locally, and sets paths accordingly.
    """

    book_seconds_input_path = os.path.join(input_data_folder, "book_features_seconds.parquet")
    trade_seconds_input_path = os.path.join(input_data_folder, "trade_features_seconds.parquet")

    if book_timeid_features and not os.path.exists(book_seconds_input_path):
        raise FileNotFoundError("⚠️ Requires book seconds-level features calculated")

    if trade_timeid_features and not os.path.exists(trade_seconds_input_path):
        raise FileNotFoundError("⚠️ Requires trade seconds-level features calculated")

    # Define Output Paths
    book_output_timeid_path = os.path.join(output_data_folder, "book_features_timeid.parquet")
    trade_output_timeid_path = os.path.join(output_data_folder, "trade_features_timeid.parquet")

    # Book Features - Time ID Level
    if book_timeid_features:
        _apply_features(book_seconds_input_path, book_output_timeid_path, generate_features_book_timeid, book_timeid_features)


    # Trade Features - Time ID Level
    if trade_timeid_features:
        _apply_features(trade_seconds_input_path, trade_output_timeid_path, generate_features_trade_timeid)

