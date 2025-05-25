import numpy as np
from sklearn.svm import SVR
from typing import List
from ..base import BaseModel

class MultiOutputSVM(BaseModel):
    def __init__(self, kernel: str = 'rbf', 
                 C: float = 1.0, epsilon: float = 0.1):
        super().__init__('MultiOutputSVM')
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.models: List[SVR] = []
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        n_outputs = y.shape[1]
        self.models = []
        
        for i in range(n_outputs):
            model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon)  # type: ignore
            model.fit(X, y[:, i])
            self.models.append(model)
            
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
            
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
            
        return np.column_stack(predictions)
        
    def get_params(self) -> dict:
        return {
            'kernel': self.kernel,
            'C': self.C,
            'epsilon': self.epsilon
        }
        
    def set_params(self, **params) -> None:
        for key, value in params.items():
            setattr(self, key, value) 