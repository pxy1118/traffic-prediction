import numpy as np
from .base import BaseModel
from torch.utils.tensorboard import SummaryWriter

class BPNetwork(BaseModel):
    def __init__(self, hidden_layers=[64, 32], learning_rate=0.01, max_iter=100, normalize=True):
        super().__init__(normalize)
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = []
        self.biases = []
        
    def _initialize_parameters(self, n_features, output_dim=1):
        self.weights = []
        self.biases = []
        self.weights.append(np.random.randn(n_features, self.hidden_layers[0]).astype(np.float64) * 0.01)
        self.biases.append(np.zeros(self.hidden_layers[0], dtype=np.float64))
        
        for i in range(len(self.hidden_layers)-1):
            self.weights.append(np.random.randn(self.hidden_layers[i], self.hidden_layers[i+1]).astype(np.float64) * 0.01)
            self.biases.append(np.zeros(self.hidden_layers[i+1], dtype=np.float64))
        
        self.weights.append(np.random.randn(self.hidden_layers[-1], output_dim).astype(np.float64) * 0.01)
        self.biases.append(np.zeros(output_dim, dtype=np.float64))
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def _forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        for i in range(len(self.hidden_layers)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            activation = self._sigmoid(z)
            self.activations.append(activation)
        
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        output = z
        self.activations.append(output)
        
        return output
    
    def _backward(self, X, y):
        m = X.shape[0]
        delta = self.activations[-1] - y
        dW = []
        db = []
        dW.append(np.dot(self.activations[-2].T, delta) / m)
        db.append(np.sum(delta, axis=0) / m)
        for i in range(len(self.hidden_layers)-1, -1, -1):
            delta = np.dot(delta, self.weights[i+1].T) * self._sigmoid_derivative(self.activations[i+1])
            dW.insert(0, np.dot(self.activations[i].T, delta) / m)
            db.insert(0, np.sum(delta, axis=0) / m)
        return dW, db
    
    def fit(self, X, y, log_dir='runs/BPNetwork'):
        writer = SummaryWriter(log_dir)
        X = self._normalize_data(X)
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        n_samples, n_features = X.shape
        output_dim = y.shape[1] if y.ndim > 1 else 1
        self._initialize_parameters(n_features, output_dim)
        for i in range(self.max_iter):
            y_pred = self._forward(X)
            if y_pred.shape != y.shape:
                y_ = y.reshape(y_pred.shape)
            else:
                y_ = y
            loss = np.mean((y_pred - y_) ** 2)
            mae = np.mean(np.abs(y_pred.flatten() - y_.flatten()))
            mse = np.mean((y_pred.flatten() - y_.flatten()) ** 2)
            rmse = np.sqrt(mse)
            writer.add_scalar('Loss', loss, i)
            writer.add_scalar('MAE', mae, i)
            writer.add_scalar('MSE', mse, i)
            writer.add_scalar('RMSE', rmse, i)
            dW, db = self._backward(X, y_)
            for j in range(len(self.weights)):
                self.weights[j] = self.weights[j] - self.learning_rate * dW[j]
                self.biases[j] = self.biases[j] - self.learning_rate * db[j]
            if i % 100 == 0:
                print(f"epoch {i}, Loss: {loss:.4f}")
        writer.close()
        return self
    
    def predict(self, X):
        if not self.weights or not self.biases:
            raise ValueError("Model has not been fitted yet.")
        X = self._normalize_data(X)
        X = np.array(X, dtype=np.float64)
        return self._forward(X) 