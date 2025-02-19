import os
import pandas as pd
from src.utils import load_data
from src.feature_engineering import apply_feature_engineering
from src.model_base import BaseModel

class CurrentModel(BaseModel):
    """Handles the current best model."""

    def __init__(self):
        super().__init__("current_model")
        self.load_model()

if __name__ == "__main__":
    print("🔍 Loading raw test data...")
    X_test, y_test = load_data("test")

    # ✅ Apply feature engineering based on model feature lists
    baseline = BaseModel("baseline_model")
    baseline.load_model()
    X_test_baseline = apply_feature_engineering(X_test, baseline.feature_list)

    new_model = BaseModel("new_model")
    new_model.load_model()
    X_test_new = apply_feature_engineering(X_test, new_model.feature_list)

    current_model = CurrentModel()
    X_test_current = apply_feature_engineering(X_test, current_model.feature_list)

    # Compute errors
    results = {
        "Baseline Model": baseline.evaluate(X_test_baseline, y_test),
        "Current Model": current_model.evaluate(X_test_current, y_test),
        "New Model": new_model.evaluate(X_test_new, y_test),
    }

    print("\n📊 Model Errors (Lower is better):")
    for model, error in results.items():
        print(f"{model}: {error}")

    # ✅ Replace current model if new model is better
    if results["New Model"] < results["Current Model"]:
        print("\n✅ New model is better! Updating current model...")
        os.replace("models/new_model.pkl", "models/current_model.pkl")
    else:
        print("\n❌ New model is not better. Keeping current model.")

    # ✅ Save evaluation results
    pd.DataFrame([results]).to_csv("evaluation_results.csv", index=False)
