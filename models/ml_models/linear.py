import numpy as np
from sklearn.linear_model import LinearRegression
from ..base import BaseModel

class MultiOutputLinear(BaseModel):
    def __init__(self, fit_intercept=True):
        super().__init__('MultiOutputLinear')
        self.fit_intercept = fit_intercept
        self.model = LinearRegression(fit_intercept=fit_intercept)
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        self.model.fit(X, y)
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.predict(X)
        
    def get_params(self) -> dict:
        return {
            'fit_intercept': self.fit_intercept
        }
        
    def set_params(self, **params) -> None:
        for key, value in params.items():
            setattr(self, key, value)
        self.model = LinearRegression(fit_intercept=self.fit_intercept) 