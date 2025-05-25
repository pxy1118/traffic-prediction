import numpy as np
from .base import BaseModel
from tqdm import tqdm

class BaggingEnsemble(BaseModel):
    def __init__(self, base_model, n_estimators=10, normalize=True):
        super().__init__(normalize)
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.models = []
        
    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X, y):
        X = self._normalize_data(X)
        self.models = []
        
        for _ in tqdm(range(self.n_estimators), desc="Training Bagging Models"):
            model = self.base_model()
            X_bootstrap, y_bootstrap = self._bootstrap_sample(X, y)
            model.fit(X_bootstrap, y_bootstrap)
            self.models.append(model)
        
        return self
    
    def predict(self, X):
        X = self._normalize_data(X)
        predictions = []
        
        for model in tqdm(self.models, desc="Making Predictions"):
            pred = model.predict(X)
            predictions.append(pred)
        
        return np.mean(predictions, axis=0)

class AdaBoostEnsemble(BaseModel):
    def __init__(self, base_model, n_estimators=10, normalize=True):
        super().__init__(normalize)
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.models = []
        self.weights = []
        
    def fit(self, X, y):
        X = self._normalize_data(X)
        n_samples = X.shape[0]
        
        sample_weights = np.ones(n_samples) / n_samples
        self.models = []
        self.weights = []
        
        for _ in tqdm(range(self.n_estimators), desc="Training AdaBoost Models"):
            model = self.base_model()
            model.fit(X, y, sample_weight=sample_weights)
            y_pred = model.predict(X)
            
            error = np.sum(sample_weights * (y_pred != y)) / np.sum(sample_weights)
            alpha = 0.5 * np.log((1 - error) / error)
            
            sample_weights *= np.exp(-alpha * y * y_pred)
            sample_weights /= np.sum(sample_weights)
            
            self.models.append(model)
            self.weights.append(alpha)
            
            print(f"Model {_+1}/{self.n_estimators}, Error: {error:.4f}, Alpha: {alpha:.4f}")
        
        return self
    
    def predict(self, X):
        X = self._normalize_data(X)
        predictions = []
        
        for model, weight in tqdm(zip(self.models, self.weights), desc="Making Predictions"):
            pred = model.predict(X)
            predictions.append(weight * pred)
        
        return np.sum(predictions, axis=0)

class StackingEnsemble(BaseModel):
    def __init__(self, base_models, meta_model, normalize=True):
        super().__init__(normalize)
        self.base_models = base_models
        self.meta_model = meta_model
        self.base_predictions = None
        
    def fit(self, X, y):
        X = self._normalize_data(X)
        n_samples = X.shape[0]
        
        self.base_predictions = np.zeros((n_samples, len(self.base_models)))
        
        for i, model in enumerate(tqdm(self.base_models, desc="Training Base Models")):
            model.fit(X, y)
            y_pred = model.predict(X)
            self.base_predictions[:, i] = y_pred
            loss = np.mean((y_pred - y) ** 2)
            print(f"Base Model {i+1}/{len(self.base_models)}, MSE: {loss:.4f}")
        
        print("Training Meta Model...")
        self.meta_model.fit(self.base_predictions, y)
        
        return self
    
    def predict(self, X):
        X = self._normalize_data(X)
        n_samples = X.shape[0]
        
        base_preds = np.zeros((n_samples, len(self.base_models)))
        
        for i, model in enumerate(tqdm(self.base_models, desc="Making Base Predictions")):
            base_preds[:, i] = model.predict(X)
        
        return self.meta_model.predict(base_preds) 