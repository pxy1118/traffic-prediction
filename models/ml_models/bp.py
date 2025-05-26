import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional
from ..base import BaseModel
from torch.utils.tensorboard import SummaryWriter

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
        
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32, 
            writer: Optional[SummaryWriter] = None, **kwargs) -> None:
        if self.model is None:
            self.model = BPNetwork(self.input_size, self.hidden_sizes, y.shape[1]).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        assert self.model is not None
        assert self.optimizer is not None
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                output = self.model(batch_X)
                loss = nn.MSELoss()(output, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            if writer is not None:
                writer.add_scalar('Loss/train', avg_loss, epoch)
            
            if epoch % 20 == 0:
                print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
                
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
            self.model = BPNetwork(self.input_size, self.hidden_sizes, 12).to(self.device)
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