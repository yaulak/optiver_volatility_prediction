import pickle
import os
from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    """
    Abstract Base Class for all ML models.
    """

    def __init__(self, model_name, feature_list):
        self.model_name = model_name
        self.feature_list = feature_list
        self.model = None

    @abstractmethod
    def train(self, X_train, y_train):
        """Train the model (to be implemented by subclasses)."""
        pass

    def predict(self, X):
        """Make predictions using the trained model."""
        return self.model.predict(X)

    def evaluate(self, X, Y):
        """Evaluate model using RMSPE."""
        X_pred = self.predict(X)
        y_error = self.rmspe(Y, X_pred)
        return y_error

    @staticmethod
    def rmspe(y_true, y_pred):
        """Calculate Root Mean Squared Percentage Error (RMSPE)."""
        return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))

    def save_model(self, new_model_folder, all_models_folder):
        """Save the trained model."""
        os.makedirs(new_model_folder, exist_ok=True)  # Ensure directory exists
        os.makedirs(all_models_folder, exist_ok=True)

        new_model_path = os.path.join(new_model_folder, "new_model.pkl")
        all_models_path = os.path.join(all_models_folder, f"{self.model_name}.pkl")
        with open(new_model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(all_models_path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"✅ Model saved in {new_model_path} and {all_models_path}")


