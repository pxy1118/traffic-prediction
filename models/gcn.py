import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Tuple, Optional
import numpy as np

class GCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

class GCNFeatureExtractor:
    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 32, 
                 num_layers: int = 2, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = GCN(in_channels, hidden_channels, out_channels, num_layers).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
    def fit(self, x: np.ndarray, edge_index: np.ndarray, epochs: int = 100) -> None:
        x_tensor = torch.FloatTensor(x).to(self.device)
        edge_index_tensor = torch.LongTensor(edge_index).to(self.device)
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            out = self.model(x_tensor, edge_index_tensor)
            loss = F.mse_loss(out, x_tensor)
            loss.backward()
            self.optimizer.step()
            
    def transform(self, x, edge_index: torch.Tensor) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(x).to(self.device)
            edge_index = edge_index.to(self.device)
            embeddings = self.model(x, edge_index)
            return embeddings.cpu().numpy()
            
    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
        
    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path)) 