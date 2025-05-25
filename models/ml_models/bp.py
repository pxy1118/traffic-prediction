import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple
from ..base import BaseModel

class BPNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        super().__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class MultiOutputBP(BaseModel):
    def __init__(self, input_size: int, hidden_sizes: List[int] = [64, 32], 
                 learning_rate: float = 0.001, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__('MultiOutputBP')
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.device = device
        self.model = None
        self.optimizer = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32, **kwargs) -> None:
        if self.model is None:
            self.model = BPNetwork(self.input_size, self.hidden_sizes, y.shape[1]).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        self.model.train()
        for epoch in range(epochs):
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                assert self.optimizer is not None
                self.optimizer.zero_grad()
                assert self.model is not None
                output = self.model(batch_X)
                loss = nn.MSELoss()(output, batch_y)
                loss.backward()
                self.optimizer.step()
                
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        assert self.model is not None
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy()
            
    def save(self, path: str) -> None:
        if self.model is not None:
            torch.save(self.model.state_dict(), path)
            
    def load(self, path: str) -> None:
        if self.model is None:
            self.model = BPNetwork(self.input_size, self.hidden_sizes, 1).to(self.device)
        self.model.load_state_dict(torch.load(path))
        self.is_fitted = True
        
    def get_params(self) -> dict:
        return {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'learning_rate': self.learning_rate
        }
        
    def set_params(self, **params) -> None:
        for key, value in params.items():
            setattr(self, key, value)
        if self.model is not None:
            self.model = BPNetwork(self.input_size, self.hidden_sizes, 1).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate) 