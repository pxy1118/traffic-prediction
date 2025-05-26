import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from typing import Literal, Optional
from ..base import BaseModel
from torch.utils.tensorboard import SummaryWriter

class MultiOutputKNN(BaseModel):
    def __init__(self, k: int = 5):
        super().__init__('MultiOutputKNN')
        self.k = k
        self.models = []
        
    def fit(self, X: np.ndarray, y: np.ndarray, writer: Optional[SummaryWriter] = None, **kwargs) -> None:
        n_outputs = y.shape[1]
        self.models = []
        
        for i in range(n_outputs):
            model = KNN(k=self.k)
            model.fit(X, y[:, i], writer=writer, output_idx=i)
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
        return {'k': self.k}
        
    def set_params(self, **params) -> None:
        for key, value in params.items():
            setattr(self, key, value)

class KNN:
    def __init__(self, k: int = 5):
        self.k = k
        self.X_train = None
        self.y_train = None
        
    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def fit(self, X: np.ndarray, y: np.ndarray, writer: Optional[SummaryWriter] = None, output_idx: int = 0) -> None:
        self.X_train = np.array(X, dtype=np.float64)
        self.y_train = np.array(y, dtype=np.float64)
        
        # 记录训练集性能到TensorBoard
        if writer is not None:
            y_pred = self.predict(X)
            loss = np.mean((y_pred - y) ** 2)
            mae = np.mean(np.abs(y_pred - y))
            mse = np.mean((y_pred - y) ** 2)
            rmse = np.sqrt(mse)
            writer.add_scalar(f'Loss/train/output_{output_idx}', loss, 0)
            writer.add_scalar(f'MAE/train/output_{output_idx}', mae, 0)
            writer.add_scalar(f'MSE/train/output_{output_idx}', mse, 0)
            writer.add_scalar(f'RMSE/train/output_{output_idx}', rmse, 0)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model has not been fitted yet.")
            
        X = np.array(X, dtype=np.float64)
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples, dtype=np.float64)
        
        for i in range(n_samples):
            distances = np.array([self._euclidean_distance(X[i], x) for x in self.X_train])
            k_indices = np.argsort(distances)[:self.k]
            y_pred[i] = np.mean(self.y_train[k_indices])
            
        return y_pred 