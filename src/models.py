from sklearn.linear_model import LinearRegression
from src.model_base import BaseModel
import numpy as np

class BaselineModel(BaseModel):
    """Linear Regression Model (Baseline)"""

    def __init__(self, model_name, feature_list):
        super().__init__(model_name, feature_list)

    def train(self, X_train, y_train):
        """Train Linear Regression Model"""
        self.model = LinearRegression(fit_intercept=True)  # No constant
        self.model.coef_ = np.array([1])  # Force coefficient to 1
        self.model.intercept_ = np.array([0])
        print("✅ Trained Baseline Model")
