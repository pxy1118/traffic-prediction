import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
from torch.utils.tensorboard import SummaryWriter

class BaseModel(ABC):
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, writer: Optional[SummaryWriter] = None, **kwargs) -> None:
        """训练模型"""
        pass
        
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
        
    def save(self, path: str) -> None:
        pass
        
    def load(self, path: str) -> None:
        pass
        
    def get_params(self) -> Dict[str, Any]:
        return {}
        
    def set_params(self, **params) -> None:
        pass 