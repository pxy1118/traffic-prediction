import numpy as np
from .base import BaseModel
from torch.utils.tensorboard import SummaryWriter

class KNN(BaseModel):
    def __init__(self, k=5, normalize=True):
        super().__init__(normalize)
        self.k = k
        self.X_train = None
        self.y_train = None
        
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def fit(self, X, y, log_dir='runs/KNN'):
        writer = SummaryWriter(log_dir)
        X = self._normalize_data(X)
        self.X_train = np.array(X, dtype=np.float64)
        self.y_train = np.array(y, dtype=np.float64)
        print(f"[DEBUG] KNN fit: X.shape={X.shape}, y.shape={y.shape}")
        y_pred = self.predict(X)
        mae = np.mean(np.abs(y_pred.flatten() - y.flatten()))
        mse = np.mean((y_pred.flatten() - y.flatten()) ** 2)
        rmse = np.sqrt(mse)
        writer.add_scalar('MAE', mae, 0)
        writer.add_scalar('MSE', mse, 0)
        writer.add_scalar('RMSE', rmse, 0)
        writer.close()
        return self
    
    def predict(self, X):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model has not been fitted yet.")
        X = self._normalize_data(X)
        X = np.array(X, dtype=np.float64)
        n_samples = X.shape[0]
        # 多输出自适应
        if self.y_train.ndim == 1:
            y_pred = np.zeros(n_samples, dtype=np.float64)
            for i in range(n_samples):
                distances = np.array([self._euclidean_distance(X[i], x) for x in self.X_train])
                k_indices = np.argsort(distances)[:self.k]
                y_pred[i] = np.mean(self.y_train[k_indices])
        else:
            n_targets = self.y_train.shape[1]
            y_pred = np.zeros((n_samples, n_targets), dtype=np.float64)
            for i in range(n_samples):
                distances = np.array([self._euclidean_distance(X[i], x) for x in self.X_train])
                k_indices = np.argsort(distances)[:self.k]
                y_pred[i, :] = np.mean(self.y_train[k_indices, :], axis=0)
        print(f"[DEBUG] KNN predict: X.shape={X.shape}, y_pred.shape={y_pred.shape}")
        return y_pred 