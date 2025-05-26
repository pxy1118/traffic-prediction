import numpy as np
from ..base import BaseModel
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

class MultiOutputSVM(BaseModel):
    def __init__(self, C: float = 1.0, epsilon: float = 0.1, learning_rate: float = 0.01, max_iter: int = 1000):
        super().__init__('MultiOutputSVM')
        self.C = C
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.models = []
        
    def fit(self, X: np.ndarray, y: np.ndarray, writer: Optional[SummaryWriter] = None, **kwargs) -> None:
        n_outputs = y.shape[1]
        self.models = []
        
        for i in range(n_outputs):
            model = SVM(C=self.C, epsilon=self.epsilon, learning_rate=self.learning_rate, max_iter=self.max_iter)
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
        return {
            'C': self.C,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'max_iter': self.max_iter
        }
        
    def set_params(self, **params) -> None:
        for key, value in params.items():
            setattr(self, key, value)

class SVM:
    def __init__(self, C: float = 1.0, epsilon: float = 0.1, learning_rate: float = 0.01, max_iter: int = 1000):
        self.C = C
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.alpha = None
        self.b = None
        self.X = None
        self.y = None
        
    def _kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.exp(-np.sum((x1 - x2) ** 2) / (2 * 1.0))
    
    def _compute_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples), dtype=np.float64)
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self._kernel(X[i], X[j])
        return K
    
    def fit(self, X: np.ndarray, y: np.ndarray, writer: Optional[SummaryWriter] = None, output_idx: int = 0) -> None:
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        n_samples = X.shape[0]
        self.X = X
        self.y = y
        self.alpha = np.zeros(n_samples, dtype=np.float64)
        self.b = np.float64(0.0)
        K = self._compute_kernel_matrix(X)
        
        for i in range(self.max_iter):
            for j in range(n_samples):
                Ei = self.predict(X[j:j+1])[0] - y[j]
                if ((y[j] * Ei < -self.epsilon and self.alpha[j] < self.C) or 
                    (y[j] * Ei > self.epsilon and self.alpha[j] > 0)):
                    k = np.random.randint(0, n_samples)
                    while k == j:
                        k = np.random.randint(0, n_samples)
                    Ek = self.predict(X[k:k+1])[0] - y[k]
                    old_alpha_j = self.alpha[j]
                    old_alpha_k = self.alpha[k]
                    L = max(0, old_alpha_k + old_alpha_j - self.C)
                    H = min(self.C, old_alpha_k + old_alpha_j)
                    if L == H:
                        continue
                    eta = 2 * K[j, k] - K[j, j] - K[k, k]
                    if eta >= 0:
                        continue
                    self.alpha[k] = old_alpha_k - y[k] * (Ei - Ek) / eta
                    self.alpha[k] = max(L, min(H, self.alpha[k]))
                    if abs(self.alpha[k] - old_alpha_k) < 1e-4:
                        continue
                    self.alpha[j] = old_alpha_j + y[j] * y[k] * (old_alpha_k - self.alpha[k])
                    b1 = self.b - Ei - y[j] * (self.alpha[j] - old_alpha_j) * K[j, j] - y[k] * (self.alpha[k] - old_alpha_k) * K[j, k]
                    b2 = self.b - Ek - y[j] * (self.alpha[j] - old_alpha_j) * K[j, k] - y[k] * (self.alpha[k] - old_alpha_k) * K[k, k]
                    self.b = (b1 + b2) / 2
            
            # 记录指标到TensorBoard
            if writer is not None and i % 10 == 0:
                y_pred = self.predict(X)
                loss = np.mean((y_pred - y) ** 2)
                mae = np.mean(np.abs(y_pred - y))
                mse = np.mean((y_pred - y) ** 2)
                rmse = np.sqrt(mse)
                writer.add_scalar(f'Loss/train/output_{output_idx}', loss, i)
                writer.add_scalar(f'MAE/train/output_{output_idx}', mae, i)
                writer.add_scalar(f'MSE/train/output_{output_idx}', mse, i)
                writer.add_scalar(f'RMSE/train/output_{output_idx}', rmse, i)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.alpha is None or self.b is None or self.X is None or self.y is None:
            raise ValueError("Model has not been fitted yet.")
        X = np.array(X, dtype=np.float64)
        y_pred = np.zeros(X.shape[0], dtype=np.float64)
        for i in range(X.shape[0]):
            kernel_values = np.array([self._kernel(x, X[i]) for x in self.X], dtype=np.float64)
            y_pred[i] = np.sum(self.alpha * self.y * kernel_values) + self.b
        return y_pred 