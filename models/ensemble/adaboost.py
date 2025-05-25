import numpy as np
from typing import List, Type
from ..base import BaseModel

class AdaBoostEnsemble(BaseModel):
    def __init__(self, base_model_class: Type[BaseModel], n_estimators: int = 10, learning_rate: float = 1.0, **base_params):
        super().__init__('AdaBoostEnsemble')
        self.base_model_class = base_model_class
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_params = base_params
        self.models: List[BaseModel] = []
        self.weights: List[float] = []
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        self.models = []
        self.weights = []
        n_samples = len(X)
        sample_weights = np.ones(n_samples) / n_samples
        
        for i in range(self.n_estimators):
            model = self.base_model_class(**self.base_params)
            model.fit(X, y, sample_weight=sample_weights, **kwargs)
            
            predictions = model.predict(X)
            errors = np.abs(predictions - y)
            if errors.ndim > 1:
                errors = np.mean(errors, axis=1)
            weighted_error = np.sum(sample_weights * errors) / np.sum(sample_weights)
            
            if weighted_error >= 0.5:
                break
                
            alpha = self.learning_rate * 0.5 * np.log((1 - weighted_error) / weighted_error)
            sample_weights *= np.exp(alpha * errors)
            sample_weights /= np.sum(sample_weights)
            
            self.models.append(model)
            self.weights.append(alpha)
            
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
            
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(X)
            predictions.append(pred * weight)
            
        return np.sum(predictions, axis=0) / np.sum(self.weights)
        
    def get_params(self) -> dict:
        return {
            'base_model_class': self.base_model_class,
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            **self.base_params
        }
        
    def set_params(self, **params) -> None:
        if 'base_model_class' in params:
            self.base_model_class = params.pop('base_model_class')
        if 'n_estimators' in params:
            self.n_estimators = params.pop('n_estimators')
        if 'learning_rate' in params:
            self.learning_rate = params.pop('learning_rate')
        self.base_params = params 