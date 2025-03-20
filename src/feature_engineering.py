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
    """Calculate log returns, ensuring that missing shift values default to the original value (log return = 0)."""
    data['wap1_shift'] = data.groupby('time_id')['wap1'].shift(1).fillna(data['wap1'])
    data['wap2_shift'] = data.groupby('time_id')['wap2'].shift(1).fillna(data['wap2'])
    data['log_return1'] = np.log(data['wap1'] / data['wap1_shift'])
    data['log_return2'] = np.log(data['wap2'] / data['wap2_shift'])
    return data

def get_wap_balance(data):
    """Calculate the absolute difference between wap1 and wap2."""
    data['wap_balance'] = abs(data['wap1'] - data['wap2'])
    return data

def get_spreads(data):
    """Calculate bid and ask spreads."""
    data['bid_spread'] = data['bid_price1'] - data['bid_price2']
    data['ask_spread'] = data['ask_price2'] - data['ask_price1']
    return data

def get_price_spread(data):
    """Calculate the difference between the best ask and best bid prices."""
    data['price_spread'] = data['ask_price1'] - data['bid_price1']
    return data

def get_volume_imbalance(data):
    """Calculate the absolute difference between bid size and ask size."""
    data['volume_imbalance'] = abs(data['bid_size1'] - data['ask_size1'])
    return data

def flatten_name(prefix, src_names):
    ret = []
    for c in src_names:
        if c[0] in ['time_id']:
            ret.append(c[0])
        else:
            ret.append('.'.join([prefix] + list(c)))
    return ret



def aggregate_features(data):
    """Aggregate features dynamically."""

    # Define aggregation functions (aligned with Kaggle)
    features = {
        'wap1': ["mean", "std"],
        'wap2': ["mean", "std"],
        'log_return1': ["mean", "std", realized_volatility],
        'log_return2': ["mean", "std", realized_volatility],
        'wap_balance': ["mean", "std"],
        'price_spread': ["mean", "std"],
        'bid_spread': ["mean", "std"],
        'ask_spread': ["mean", "std"],
        'volume_imbalance': ["mean", "std"]
    }

    agg = data.groupby('time_id').agg(features).reset_index(drop=False)
    agg.columns = flatten_name('book', agg.columns)

    for time in [60, 120, 180, 240, 300]:
        d = data[data['seconds_in_bucket'] >= 600 - time].groupby('time_id').agg(features).reset_index(drop=False)
        d.columns = flatten_name(f'book_{time}', d.columns)
        agg = pd.merge(agg, d, on='time_id', how='left')
    return agg


### ===== Feature Engineering Functions ===== ###

def generate_features_book_seconds(data):
    """Generate book-level features at stock-time-seconds level."""
    data = get_wap_feature(data)
    data = get_log_return_feature(data)
    data = get_wap_balance(data)
    data = get_spreads(data)
    data = get_price_spread(data)
    data = get_volume_imbalance(data)
    return data

def generate_features_book_timeid(data):
    """Generate book-level features at stock-time_id level for different time intervals."""
    return aggregate_features(data)


def generate_features_trade_seconds(data):
    """(Placeholder) Generate trade-level features at stock-time-seconds level."""
    return data  # Extend with actual trade features


def generate_features_trade_timeid(data):
    """(Placeholder) Generate trade-level features at stock-time_id level."""
    return data  # Extend with actual trade features


### ===== Generalized Feature Application Function ===== ###

def _apply_features(input_path, output_path, feature_func):
    """Applies a given feature function to all parquet files in the directory
       and saves a single output file partitioned by stock_id."""

    os.makedirs(output_path, exist_ok=True)  # Ensure output directory exists

    for filename in tqdm(os.listdir(input_path)):
        file_path = os.path.join(input_path, filename)

        df_temp = pd.read_parquet(file_path)
        stock_id = int(filename.split('=')[1].split('.')[0])

        # Apply selected feature function
        df_temp = feature_func(df_temp)
        df_temp['stock_id'] = stock_id

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
        _apply_features(book_input_path, book_output_seconds_path, generate_features_book_seconds)

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
        _apply_features(book_seconds_input_path, book_output_timeid_path, generate_features_book_timeid)


    # Trade Features - Time ID Level
    if trade_timeid_features:
        _apply_features(trade_seconds_input_path, trade_output_timeid_path, generate_features_trade_timeid)

