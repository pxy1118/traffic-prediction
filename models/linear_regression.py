import numpy as np
from .base import BaseModel
from torch.utils.tensorboard import SummaryWriter

class LinearRegression(BaseModel):
    def __init__(self, learning_rate=0.01, max_iter=1000, normalize=True):
        super().__init__(normalize)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None
        
    def _initialize_parameters(self, n_features, output_dim=1):
        self.weights = np.random.randn(n_features, output_dim).astype(np.float64) * 0.01
        self.bias = np.zeros(output_dim, dtype=np.float64)
        
    def _compute_gradients(self, X, y, y_pred):
        m = X.shape[0]
        dw = np.dot(X.T, (y_pred - y)) / m
        db = np.sum(y_pred - y) / m
        return dw, db
    
    def fit(self, X, y, log_dir='runs/LinearRegression'):
        writer = SummaryWriter(log_dir)
        X = self._normalize_data(X)
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        print(f"[DEBUG] LinearRegression fit: X.shape={X.shape}, y.shape={y.shape}")
        n_samples, n_features = X.shape
        output_dim = y.shape[1] if y.ndim > 1 else 1
        self._initialize_parameters(n_features, output_dim)
        self.weights = np.array(self.weights, dtype=np.float64)
        self.bias = np.array(self.bias, dtype=np.float64)
        for i in range(self.max_iter):
            y_pred = np.dot(X, self.weights) + self.bias
            loss = np.mean((y_pred - y) ** 2)
            mae = np.mean(np.abs(y_pred.flatten() - y.flatten()))
            mse = np.mean((y_pred.flatten() - y.flatten()) ** 2)
            rmse = np.sqrt(mse)
            writer.add_scalar('Loss', loss, i)
            writer.add_scalar('MAE', mae, i)
            writer.add_scalar('MSE', mse, i)
            writer.add_scalar('RMSE', rmse, i)
            grad_w = 2 * np.dot(X.T, (y_pred - y)) / n_samples
            grad_b = 2 * np.mean(y_pred - y, axis=0)
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b
            if i % 100 == 0:
                print(f"epoch {i}, Loss: {loss:.4f}")
        writer.close()
        return self
    
    def predict(self, X):
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been fitted yet.")
        X = self._normalize_data(X)
        X = np.array(X, dtype=np.float64)
        y_pred = np.dot(X, self.weights) + self.bias
        print(f"[DEBUG] LinearRegression predict: X.shape={X.shape}, y_pred.shape={y_pred.shape}")
        return y_pred 