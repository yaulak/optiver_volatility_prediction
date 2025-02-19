import pickle
import os
from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    """
    Abstract Base Class for all ML models.
    """

    def __init__(self, model_name):
        self.model = None
        self.model_name = model_name
        self.model_path = f"models/{model_name}.pkl"
        self.feature_list = []  # This will be set in subclasses

    @abstractmethod
    def train(self, X_train, y_train):
        """Train the model (to be implemented by subclasses)."""
        pass

    def predict(self, X):
        """Make predictions using the trained model."""
        return self.model.predict(X)

    def evaluate(self, X_train, y_train):
        """Evaluate model using RMSPE (only on training data)."""
        preds_train = self.predict(X_train)
        train_error = self.rmspe(y_train, preds_train)
        print(f"📊 RMSPE (Training Error): {train_error:.6f}")
        return train_error

    @staticmethod
    def rmspe(y_true, y_pred):
        """Calculate Root Mean Squared Percentage Error (RMSPE)."""
        return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))

    def save_model(self):
        """Save the trained model."""
        os.makedirs("models", exist_ok=True)  # Ensure directory exists
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"✅ Model saved: {self.model_path}")

    def load_model(self):
        """Load an existing trained model."""
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            print(f"✅ Loaded model: {self.model_path}")
        else:
            print(f"⚠ Model not found: {self.model_path}")
