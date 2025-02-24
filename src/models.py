from sklearn.linear_model import LinearRegression
from src.model_base import BaseModel
import numpy as np

# class BaselineModel(BaseModel):
#     """Custom Linear Regression Model that enforces y = x (coefficient 1, no intercept)"""
#
#     def __init__(self, model_name, feature_list):
#         super().__init__(model_name, feature_list)
#         self.model = None  # We don't need an actual sklearn model
#
#     def train(self, X_train, y_train):
#         """Fake training: Just prints confirmation since y = x is predefined"""
#         print("✅ Trained Linear Regression Model with y = x (coefficient = 1, intercept = 0)")
#
#     def predict(self, X):
#         """Predict using y = x"""
#         return np.array(X)

class BaselineModel(BaseModel):
    """Linear Regression Model (Baseline)"""

    def __init__(self, model_name, feature_list):
        super().__init__(model_name, feature_list)

    def train(self, X_train, y_train):
        """Train Linear Regression Model"""
        self.model = LinearRegression(fit_intercept=True)  # No constant
        #self.model.fit(X_train, y_train)
        #print(f"coeff: {self.model.coef_} {type(self.model.coef_)} and {self.model.intercept_} and {type(self.model.intercept_)}")
        self.model.coef_ = np.array([1])  # Force coefficient to 1
        self.model.intercept_ = np.array([0])
        #print(f"coeff: {self.model.coef_} {type(self.model.coef_)} and {self.model.intercept_} and {type(self.model.intercept_)}")
        print("✅ Trained Baseline Model")
