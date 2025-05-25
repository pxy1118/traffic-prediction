import numpy as np
from typing import List, Type, Union
from ..base import BaseModel

class StackingEnsemble(BaseModel):
    def __init__(self, base_models: List[Union[BaseModel, Type[BaseModel]]], 
                 meta_model: Union[BaseModel, Type[BaseModel]], **meta_params):
        super().__init__('StackingEnsemble')
        self.base_models = base_models
        self.meta_model = meta_model
        self.meta_params = meta_params
        self.trained_base_models: List[BaseModel] = []
        self.trained_meta_model = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        self.trained_base_models = []
        n_samples = len(X)
        
        # Train base models
        for model_class in self.base_models:
            if isinstance(model_class, type):
                model = model_class()
            else:
                model = model_class
            model.fit(X, y, **kwargs)
            self.trained_base_models.append(model)
            
        # Generate meta-features
        meta_features = []
        for model in self.trained_base_models:
            pred = model.predict(X)
            meta_features.append(pred)
        meta_features = np.column_stack(meta_features)
        
        # Train meta-model
        if isinstance(self.meta_model, type):
            self.trained_meta_model = self.meta_model(**self.meta_params)
        else:
            self.trained_meta_model = self.meta_model
        self.trained_meta_model.fit(meta_features, y, **kwargs)
        
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        assert self.trained_meta_model is not None, "Meta model not fitted"
        meta_features = []
        for model in self.trained_base_models:
            pred = model.predict(X)
            meta_features.append(pred)
        meta_features = np.column_stack(meta_features)
        return self.trained_meta_model.predict(meta_features)
        
    def get_params(self) -> dict:
        return {
            'base_models': self.base_models,
            'meta_model': self.meta_model,
            **self.meta_params
        }
        
    def set_params(self, **params) -> None:
        if 'base_models' in params:
            self.base_models = params.pop('base_models')
        if 'meta_model' in params:
            self.meta_model = params.pop('meta_model')
        self.meta_params = params 