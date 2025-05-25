import numpy as np
from .base import BaseModel
from torch.utils.tensorboard import SummaryWriter

class SVM(BaseModel):
    def __init__(self, C=1.0, epsilon=0.1, learning_rate=0.01, max_iter=1000, normalize=True):
        super().__init__(normalize)
        self.C = C
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.alpha = None
        self.b = None
        self.X = None
        self.y = None
        
    def _kernel(self, x1, x2):
        return np.exp(-np.sum((x1 - x2) ** 2) / (2 * 1.0))
    
    def _compute_kernel_matrix(self, X):
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples), dtype=np.float64)
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self._kernel(X[i], X[j])
        return K
    
    def fit(self, X, y, log_dir='runs/SVM'):
        writer = SummaryWriter(log_dir)
        X = self._normalize_data(X)
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        print(f"[DEBUG] SVM fit: X.shape={X.shape}, y.shape={y.shape}")
        n_samples = X.shape[0]
        self.X = X
        self.y = y
        self.alpha = np.zeros(n_samples, dtype=np.float64)
        self.b = np.float64(0.0)
        K = self._compute_kernel_matrix(X)
        for i in range(self.max_iter):
            for j in range(n_samples):
                Ei = self.predict(X[j:j+1])[0] - y[j]
                if ((y[j] * Ei < -self.epsilon and self.alpha[j] < self.C) or (y[j] * Ei > self.epsilon and self.alpha[j] > 0)):
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
            y_pred = self.predict(X)
            loss = np.mean((y_pred - y) ** 2)
            mae = np.mean(np.abs(y_pred.flatten() - y.flatten()))
            mse = np.mean((y_pred.flatten() - y.flatten()) ** 2)
            rmse = np.sqrt(mse)
            writer.add_scalar('Loss', loss, i)
            writer.add_scalar('MAE', mae, i)
            writer.add_scalar('MSE', mse, i)
            writer.add_scalar('RMSE', rmse, i)
            if i % 100 == 0:
                print(f"Iteration {i}")
        writer.close()
        return self
    
    def predict(self, X):
        if self.alpha is None or self.b is None or self.X is None or self.y is None:
            raise ValueError("Model has not been fitted yet.")
        X = self._normalize_data(X)
        X = np.array(X, dtype=np.float64)
        y_pred = np.zeros(X.shape[0], dtype=np.float64)
        for i in range(X.shape[0]):
            kernel_values = np.array([self._kernel(x, X[i]) for x in self.X], dtype=np.float64)
            y_pred[i] = np.sum(self.alpha * self.y * kernel_values) + self.b
        return y_pred

class MultiOutputSVM:
    def __init__(self, **svm_kwargs):
        self.svms = []
        self.svm_kwargs = svm_kwargs

    def fit(self, X, Y):
        self.svms = []
        for i in range(Y.shape[1]):
            svm = SVM(**self.svm_kwargs)
            svm.fit(X, Y[:, i])
            self.svms.append(svm)

    def predict(self, X):
        preds = [svm.predict(X) for svm in self.svms]
        return np.stack(preds, axis=1)
