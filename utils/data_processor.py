import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Any
from models.gcn import GCNFeatureExtractor
import torch

class DataProcessor:
    def __init__(self, data_path: str, seq_len: int = 12, pred_len: int = 12, adj_path: str = "data/PEMS-08/adj_mx.npy"):
        self.data_path = data_path
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.adj_path = adj_path
        self.scaler = StandardScaler()
        self.gcn_extractor = None

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        data = np.load(self.data_path)
        x = data['data']  # (样本数, 节点数, 特征数)
        adj = np.load(self.adj_path)
        return x, adj

    def normalize(self, x: np.ndarray) -> np.ndarray:
        n_samples, n_nodes, n_features = x.shape
        x_reshaped = x.reshape(-1, n_features)
        x_normalized = self.scaler.fit_transform(x_reshaped)
        return x_normalized.reshape(n_samples, n_nodes, n_features)

    def inverse_normalize(self, x: Any) -> np.ndarray:
        n_samples, n_nodes, n_features = x.shape
        x_reshaped = x.reshape(-1, n_features)
        x_original = self.scaler.inverse_transform(x_reshaped)
        return x_original.reshape(n_samples, n_nodes, n_features)

    def inverse_normalize_target(self, x: Any) -> np.ndarray:
        assert self.scaler.scale_ is not None and self.scaler.mean_ is not None, "scaler未fit，不能反归一化"
        return x * (self.scaler.scale_[0] + 1e-8) + self.scaler.mean_[0]

    def create_sequences(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = len(x) - self.seq_len - self.pred_len + 1
        n_nodes = x.shape[1]
        n_features = x.shape[2]
        X = np.zeros((n_samples, self.seq_len, n_nodes, n_features))
        Y = np.zeros((n_samples, n_nodes, self.pred_len))
        for i in range(n_samples):
            X[i] = x[i:i+self.seq_len]
            Y[i] = x[i+self.seq_len:i+self.seq_len+self.pred_len, :, 0].T  # (pred_len, n_nodes) -> (n_nodes, pred_len)
        return X, Y

    def prepare_data(self, train_ratio: float = 0.7, val_ratio: float = 0.15):
        x, adj = self.load_data()
        x = self.normalize(x)
        X, Y = self.create_sequences(x)
        n_samples = len(X)
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        X_train, Y_train = X[:train_size], Y[:train_size]
        X_val, Y_val = X[train_size:train_size+val_size], Y[train_size:train_size+val_size]
        X_test, Y_test = X[train_size+val_size:], Y[train_size+val_size:]
        return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), adj

    def extract_gcn_features(self, x: np.ndarray, edge_index: 'torch.Tensor', hidden_channels: int = 64, out_channels: int = 32) -> np.ndarray:
        if self.gcn_extractor is None:
            self.gcn_extractor = GCNFeatureExtractor(
                in_channels=x.shape[-1],
                hidden_channels=hidden_channels,
                out_channels=out_channels
            )
        n_samples, n_nodes, n_features = x.shape
        x_reshaped = x.reshape(-1, n_features)
        features = self.gcn_extractor.transform(x_reshaped, edge_index)
        return features.reshape(n_samples, n_nodes, -1)

    @staticmethod
    def load_adj(adj_path: str) -> sp.csr_matrix:
        adj_matrix = np.load(adj_path)
        return sp.csr_matrix(adj_matrix)

    @staticmethod
    def normalize_adj(adj):
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        return normalized_adj 