from sklearn.linear_model import LinearRegression
from src.model_base import BaseModel

class LinearRegressionModel(BaseModel):
    """Linear Regression Model (Baseline)"""

    def __init__(self, model_name, feature_list):
        super().__init__(model_name, feature_list)

    def train(self, X_train, y_train):
        """Train Linear Regression Model"""
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        print("✅ Trained Linear Regression Model")
