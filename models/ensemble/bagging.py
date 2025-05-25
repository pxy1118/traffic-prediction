import numpy as np
from typing import List, Type
from ..base import BaseModel

class BaggingEnsemble(BaseModel):
    def __init__(self, base_model_class: Type[BaseModel], n_estimators: int = 10, **base_params):
        super().__init__('BaggingEnsemble')
        self.base_model_class = base_model_class
        self.n_estimators = n_estimators
        self.base_params = base_params
        self.models: List[BaseModel] = []
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        self.models = []
        n_samples = len(X)
        
        for i in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            model = self.base_model_class(**self.base_params)
            model.fit(X_bootstrap, y_bootstrap, **kwargs)
            self.models.append(model)
            
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
            
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
            
        return np.mean(predictions, axis=0)
        
    def get_params(self) -> dict:
        return {
            'base_model_class': self.base_model_class,
            'n_estimators': self.n_estimators,
            **self.base_params
        }
        
    def set_params(self, **params) -> None:
        if 'base_model_class' in params:
            self.base_model_class = params.pop('base_model_class')
        if 'n_estimators' in params:
            self.n_estimators = params.pop('n_estimators')
        self.base_params = params 