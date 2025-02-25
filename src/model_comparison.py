import os
import json
import shutil
import pandas as pd

# Paths
BASE_DIR = os.getenv("GITHUB_WORKSPACE", os.getcwd())

BEST_MODEL_DIR = os.path.join(BASE_DIR, "models/best_model")
os.makedirs(BEST_MODEL_DIR, exist_ok=True)
NEW_MODEL_DIR = os.path.join(BASE_DIR, "models/new_model")

BEST_MODEL_PATH = os.path.join(BEST_MODEL_DIR, "best_model.pkl")
BEST_METADATA_PATH = os.path.join(BEST_MODEL_DIR, "best_model_metadata.json")

NEW_MODEL_PATH = os.path.join(NEW_MODEL_DIR, "new_model.pkl")
NEW_METADATA_PATH = os.path.join(NEW_MODEL_DIR, "new_model_metadata.json")


def load_metadata(path):
    """Load model metadata (features, train/test errors)."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def compare_models():
    """Compare the new model with the best model and update if better."""

    print("\n📌 Loading metadata...")
    best_metadata = load_metadata(BEST_METADATA_PATH)
    new_metadata = load_metadata(NEW_METADATA_PATH)

    if not new_metadata:
        print("❌ No new model metadata found. Exiting comparison.")
        return

    new_test_error = new_metadata["test_error"]

    if best_metadata:
        best_test_error = best_metadata["test_error"]
        print("\n📊 Model Errors (Lower is better):")
        print(f"New Model - Test Error: {new_test_error}")
        print(f"Best Model - Test Error: {best_test_error}")

        # ✅ Update best model if new model is better
        if new_test_error < best_test_error:
            print("\n✅ New model is better! Updating the best model...")
            update_best_model()
        else:
            print("\n❌ New model is not better. Keeping the current best model.")
    else:
        # 🚀 First time running: no best model exists, so set new model as best
        print("\n🏆 No best model found. Setting the new model as the best model...")
        update_best_model()


def update_best_model():
    """Update the best model by replacing it with the new model."""

    # Ensure best model directory exists
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)

    # Replace best model & metadata
    shutil.copy(NEW_MODEL_PATH, BEST_MODEL_PATH)
    shutil.copy(NEW_METADATA_PATH, BEST_METADATA_PATH)

    print(f"✅ Best model saved as: {BEST_MODEL_PATH}")
    print(f"✅ Metadata saved as: {BEST_METADATA_PATH}")


if __name__ == "__main__":
    compare_models()
