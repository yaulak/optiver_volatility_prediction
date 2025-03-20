import pickle
import os
from abc import ABC, abstractmethod
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import numpy as np
import pandas as pd

class BaseModel(ABC):
    """
    Abstract Base Class for all ML models.
    """

    def __init__(self, model_name, feature_list, param_grid=None):
        self.model_name = model_name
        self.feature_list = feature_list
        self.model = None
        self.param_grid = param_grid

    @abstractmethod
    def create_model(self):
        """Create a model instance (to be implemented by subclasses)."""
        pass

    def train(self, X, y, cv=3):
        """Train the model using Grid Search + Cross-Validation with RMSPE scoring."""

        model = self.create_model()  # Create model instance

        if self.param_grid:
            # ✅ Define a custom scoring function for RMSPE (higher is worse, so negate it)
            rmspe_scorer = make_scorer(self.rmspe, greater_is_better=False)

            # ✅ Use GridSearchCV for hyperparameter tuning with RMSPE
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=self.param_grid,
                cv=cv,  # Cross-validation folds
                scoring=rmspe_scorer,  # Use RMSPE for scoring
                n_jobs=-1,  # Use all available CPU cores
                verbose=1
            )

            grid_search.fit(X, y)

            # Save the best model and parameters
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            self.best_cv_error = -grid_search.best_score_  # RMSPE is negative, so invert it

            print(f"✅ Best Model Found: {self.best_params}")
            print(f"📊 Best Score (RMSPE): {self.best_cv_error:.4f}")
            print(f"✅ Hyperparameter tuning completed.")
        else:
            # Train without Grid Search (if no param_grid provided)
            self.model.fit(X, y)
            print("✅ Model trained without hyperparameter tuning.")

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

    def feature_importance(self, n):
        importance_df = pd.DataFrame({"Feature": self.feature_list, "Importance": self.model.feature_importances_})
        top_n_features = importance_df.sort_values(by="Importance", ascending=False).head(n)
        return top_n_features

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


