import numpy as np
from typing import Dict

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse
    }
    
def calculate_node_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    mse = np.mean((y_true - y_pred) ** 2, axis=0)
    rmse = np.sqrt(mse)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse
    }
    
def calculate_horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    mae = np.mean(np.abs(y_true - y_pred), axis=(0, 1))
    mse = np.mean((y_true - y_pred) ** 2, axis=(0, 1))
    rmse = np.sqrt(mse)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse
    } 