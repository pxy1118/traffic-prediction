import numpy as np
from ..base import BaseModel
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, cast, Union
from numpy.typing import NDArray

class MultiOutputLinear(BaseModel):
    def __init__(self, learning_rate: float = 0.01, max_iter: int = 1000):
        super().__init__('MultiOutputLinear')
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.models = []
        
    def fit(self, X: np.ndarray, y: np.ndarray, writer: Optional[SummaryWriter] = None, **kwargs) -> None:
        n_outputs = y.shape[1]
        self.models = []
        
        for i in range(n_outputs):
            model = LinearRegression(learning_rate=self.learning_rate, max_iter=self.max_iter)
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
            'learning_rate': self.learning_rate,
            'max_iter': self.max_iter
        }
        
    def set_params(self, **params) -> None:
        for key, value in params.items():
            setattr(self, key, value)

class LinearRegression:
    def __init__(self, learning_rate: float = 0.01, max_iter: int = 1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights: Optional[NDArray[np.float64]] = None
        self.bias: Optional[NDArray[np.float64]] = None
        
    def _initialize_parameters(self, n_features: int) -> None:
        self.weights = np.random.randn(n_features).astype(np.float64) * 0.01
        self.bias = np.array([0.0], dtype=np.float64)
        
    def fit(self, X: np.ndarray, y: np.ndarray, writer: Optional[SummaryWriter] = None, output_idx: int = 0) -> None:
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        n_samples, n_features = X.shape
        
        self._initialize_parameters(n_features)
        assert self.weights is not None and self.bias is not None
        weights = cast(NDArray[np.float64], self.weights)
        bias = cast(NDArray[np.float64], self.bias)
        
        for i in range(self.max_iter):
            # 前向传播
            y_pred = np.dot(X, weights) + bias[0]
            
            # 计算损失和梯度
            loss = np.mean((y_pred - y) ** 2)
            grad_w = 2 * np.dot(X.T, (y_pred - y)) / n_samples
            grad_b = 2 * np.mean(y_pred - y)
            
            # 更新参数
            weights -= self.learning_rate * grad_w
            bias[0] -= self.learning_rate * grad_b
            
            # 记录指标到TensorBoard
            if writer is not None and i % 10 == 0:
                mae = np.mean(np.abs(y_pred - y))
                mse = np.mean((y_pred - y) ** 2)
                rmse = np.sqrt(mse)
                writer.add_scalar(f'Loss/train/output_{output_idx}', loss, i)
                writer.add_scalar(f'MAE/train/output_{output_idx}', mae, i)
                writer.add_scalar(f'MSE/train/output_{output_idx}', mse, i)
                writer.add_scalar(f'RMSE/train/output_{output_idx}', rmse, i)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been fitted yet.")
        X = np.array(X, dtype=np.float64)
        weights = cast(NDArray[np.float64], self.weights)
        bias = cast(NDArray[np.float64], self.bias)
        return np.dot(X, weights) + bias[0] 