import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.utils import detect_environment, download_kaggle_dataset, get_input_data_path, save_train_test_split
from src.feature_engineering import apply_seconds_feature_engineering, apply_timeid_feature_engineering
from src.model_base import BaseModel
from src.models import LinearRegressionModel  # Import all models

# =======================================
# ✅ USER INPUTS: Modify as Needed
# =======================================

# All input and output folders and files
# Data folder if the environment is local
local_data_folder = "data/"

train_test_target_split_folder = "train_test_split/"
os.makedirs(train_test_target_split_folder, exist_ok=True)

feature_engineering_output_folder = "feature_engineering_output/"
os.makedirs(feature_engineering_output_folder, exist_ok=True)

train_test_feature_target_folder = "feature_engineering_output/train_test_data/"
os.makedirs(train_test_feature_target_folder, exist_ok=True)

# Select features to compute
BOOK_SECONDS_FEATURES = {'wap', 'log_return'}  # Book features at stock, time_id, seconds_in_bucket level
BOOK_TIMEID_FEATURES = {'volatility'}  # Book features at stock, time_id level
TRADE_SECONDS_FEATURES = {}  # Trade features at stock, time_id, seconds_in_bucket level (empty for now)
TRADE_TIMEID_FEATURES = {}  # Trade features at stock, time_id level (empty for now)

# Choose model to train
MODEL_NAME = "LinearRegressionModel"  # Change to another model if needed

independent_variables = ['volatility1']


# =======================================
# ✅ Step 1: Detect Environment and Download Data (if local)
# =======================================
print("\n⚙️ Step 1: Detect Environment and Get Input Data Path")
env = detect_environment()
input_data_folder = get_input_data_path(env)


# =======================================
# ✅ Step 2: Create train-test csv file (each stock_id will have 80% train and 20% test records)
# =======================================
print("\n✂️️ Step 2: Train test split of target variable")
# Set a fixed random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load the original dataset
train_csv_df = pd.read_csv(os.path.join(input_data_folder, "train.csv"))

# Perform train-test split and save CSVs
save_train_test_split(df = train_csv_df, output_dir= train_test_target_split_folder, train_ratio = 0.8)


# =======================================
# ✅ Step 3: Compute and Save Second-Level Features First
# =======================================

print("\n⚙️ Step 3: Computing Second-Level Features (Book & Trade)...")
# apply_seconds_feature_engineering(
#     input_data_folder = input_data_folder,
#     output_data_folder= feature_engineering_output_folder,
#     book_seconds_features=BOOK_SECONDS_FEATURES,
#     trade_seconds_features=TRADE_SECONDS_FEATURES
# )

# =======================================
# ✅ Step 4: Compute Time-Level Features Using Saved Second-Level Features
# =======================================

print("\n⚙️ Step 4: Computing Time-Level Features (Using Second-Level Features)...")
# apply_timeid_feature_engineering(
#     input_data_folder = feature_engineering_output_folder,
#     output_data_folder= feature_engineering_output_folder,
#     book_timeid_features=BOOK_TIMEID_FEATURES,
#     trade_timeid_features=TRADE_TIMEID_FEATURES
# )

# =======================================
# ✅ Step 5: Load Processed Features and Merge Them
# =======================================

print("\n📂 Step 5: Reading processed feature files and merging book and trade time level features if required...")

# Load Book Features (TimeID level)
FEATURE_BOOK_TIMEID_PATH = os.path.join(feature_engineering_output_folder, "book_features_timeid.parquet")
if os.path.exists(FEATURE_BOOK_TIMEID_PATH):
    book_features_df = pd.read_parquet(FEATURE_BOOK_TIMEID_PATH, engine="pyarrow")
    book_features_df['stock_id'] = book_features_df['stock_id'].astype(int)
else:
    book_features_df = pd.DataFrame()

# Load Trade Features (TimeID level) - Add this when trade features are available
FEATURE_TRADE_TIMEID_PATH = os.path.join(feature_engineering_output_folder, "trade_features_timeid.parquet")
if os.path.exists(FEATURE_TRADE_TIMEID_PATH):
    trade_features_df = pd.read_parquet(FEATURE_TRADE_TIMEID_PATH, engine="pyarrow")
    trade_features_df['stock_id'] = trade_features_df['stock_id'].astype(int)
else:
    trade_features_df = pd.DataFrame()  # No trade features for now

# Merge Book and Trade Features (if trade features exist)
if not book_features_df.empty and not trade_features_df.empty:
    print("🔗 Merging book and trade features...")
    full_features_df = book_features_df.merge(trade_features_df, on=['stock_id', 'time_id'], how='left')
else:
    print("🔗 No merging required.")
    if not book_features_df.empty:
        full_features_df = book_features_df
    elif not trade_features_df.empty:
        full_features_df = trade_features_df
    else:
        full_features_df = pd.DataFrame()


# =======================================
# ✅ Step 6: Train-Test Split Based on Stock ID
# =======================================

print("\n✂️ Step 6: Splitting dataset having feature & target into training and testing sets...")
train_target = pd.read_csv(os.path.join(train_test_target_split_folder, "train.csv"))
test_target = pd.read_csv(os.path.join(train_test_target_split_folder, "test.csv"))

train_features_target = train_target.merge(full_features_df, on=['stock_id', 'time_id'], how='left')
test_features_target = test_target.merge(full_features_df, on=['stock_id', 'time_id'], how='left')

# Save processed datasets for later use
train_features_target_path = os.path.join(train_test_feature_target_folder, "train_data.parquet")
test_features_target_path = os.path.join(train_test_feature_target_folder, "test_data.parquet")

train_features_target.to_parquet(train_features_target_path, index=False, engine="pyarrow")
test_features_target.to_parquet(test_features_target_path, index=False, engine="pyarrow")

print(f"✅ Processed training data saved to {train_features_target_path}")
print(f"✅ Processed testing data saved to {test_features_target_path}")

# =======================================
# ✅ Step 7: Train the Model
# =======================================

print("\n🤖 Step 7: Training Model...")

# Get model class dynamically
model_class = globals()[MODEL_NAME]
model = model_class(MODEL_NAME.lower())

# Train Model
X_train = train_features_target[independent_variables].values
y_train = train_features_target['target'].values

model.train(X_train, y_train)

# Evaluate Model (Training Error)
model.evaluate(X_train, y_train)

# Save Model
model.save_model()
