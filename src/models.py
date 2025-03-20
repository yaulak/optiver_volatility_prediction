from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from src.model_base import BaseModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

class BaselineModel(BaseModel):
    """Linear Regression Model (Baseline)"""

    def __init__(self, model_name, feature_list):
        super().__init__(model_name, feature_list)

    def create_model(self):
        """Return a new instance of the model"""
        return LinearRegression(fit_intercept=True)

    def train(self, X_train, y_train, cv = 5):
        """Train Linear Regression Model"""
        self.model = self.create_model()
        self.model.coef_ = np.array([1])  # Force coefficient to 1
        self.model.intercept_ = np.array([0])
        print("✅ Trained Baseline Model")

class LinearRegressionModel(BaseModel):
    """Linear Regression Model with Hyperparameter Tuning (Includes Regularization)"""

    def __init__(self, model_name, feature_list):
        # Define the parameter grid for hyperparameter tuning
        param_grid = {
            "model__alpha": [0.01, 0.1, 1, 10, 100],  # Regularization strength
            "model": [Ridge(), Lasso()] # Test both Ridge and Lasso
        }
        super().__init__(model_name, feature_list, param_grid)

    def create_model(self):
        """Return a new instance of Ridge/Lasso Regression with StandardScaler"""
        return Pipeline([
            ("scaler", StandardScaler()),  # ✅ Always scale features
            ("model", Ridge())  # Default to Ridge; GridSearch will replace it
        ])


class DecisionTreeModel(BaseModel):
    """Decision Tree Regression Model with Hyperparameter Tuning"""

    def __init__(self, model_name, feature_list):
        # param_grid = {
        #     "max_depth": [15, 20, None],
        #     "min_samples_split": [100, 200],
        #     "min_samples_leaf": [25, 50, 100]
        # }
        param_grid = {
            "max_depth": [15],
            "min_samples_split": [100],
            "min_samples_leaf": [100]
        }
        super().__init__(model_name, feature_list, param_grid)

    def create_model(self):
        """Return a new instance of DecisionTreeRegressor"""
        return DecisionTreeRegressor()

