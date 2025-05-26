import numpy as np
from typing import Type, Optional
from ..base import BaseModel
from torch.utils.tensorboard import SummaryWriter

class BaggingEnsemble(BaseModel):
    def __init__(self, base_model_class: Type[BaseModel], n_estimators: int = 10, **base_params):
        super().__init__('BaggingEnsemble')
        self.base_model_class = base_model_class
        self.n_estimators = n_estimators
        self.base_params = base_params
        self.models = []
        
    def fit(self, X: np.ndarray, y: np.ndarray, writer: Optional[SummaryWriter] = None, **kwargs) -> None:
        n_samples = X.shape[0]
        self.models = []
        
        for i in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            model = self.base_model_class(**self.base_params)
            model.fit(X_bootstrap, y_bootstrap, writer=writer, output_idx=i)
            self.models.append(model)
            
            # 记录集成效果
            if writer is not None:
                y_pred = self.predict(X)
                loss = np.mean((y_pred - y) ** 2)
                mae = np.mean(np.abs(y_pred - y))
                mse = np.mean((y_pred - y) ** 2)
                rmse = np.sqrt(mse)
                writer.add_scalar('Loss/ensemble', loss, i)
                writer.add_scalar('MAE/ensemble', mae, i)
                writer.add_scalar('MSE/ensemble', mse, i)
                writer.add_scalar('RMSE/ensemble', rmse, i)
        
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
            
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
            
        return np.mean(predictions, axis=0)
        
    def save(self, path: str) -> None:
        if self.is_fitted:
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(self.models, f)
                
    def load(self, path: str) -> None:
        import pickle
        with open(path, 'rb') as f:
            self.models = pickle.load(f)
        self.is_fitted = True
        
    def get_params(self) -> dict:
        return {
            'base_model_class': self.base_model_class,
            'n_estimators': self.n_estimators,
            **self.base_params
        }
        
    def set_params(self, **params) -> None:
        for key, value in params.items():
            setattr(self, key, value) 