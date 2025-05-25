import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from typing import Literal, Optional
from ..base import BaseModel

class MultiOutputKNN(BaseModel):
    def __init__(self, n_neighbors: int = 5, weights: str = 'uniform'):
        super().__init__('MultiOutputKNN')
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)  # type: ignore
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        self.model.fit(X, y)
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.predict(X)
        
    def get_params(self) -> dict:
        return {
            'n_neighbors': self.n_neighbors,
            'weights': self.weights
        }
        
    def set_params(self, **params) -> None:
        for key, value in params.items():
            setattr(self, key, value)
        self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors, weights=self.weights)  # type: ignore 