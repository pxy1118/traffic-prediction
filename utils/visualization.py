import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List
import torch
from torch.utils.tensorboard import SummaryWriter
   # pylint: disable=no-value-for-parameter

class Visualizer:
    def __init__(self, log_dir: str = 'runs'):
        self.writer = SummaryWriter(log_dir)
        
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        for name, value in metrics.items():
            self.writer.add_scalar(f'metrics/{name}', value, step)
            
    def log_node_metrics(self, metrics: Dict[str, np.ndarray], step: int) -> None:
        for name, values in metrics.items():
            for i, value in enumerate(values):
                if isinstance(value, np.ndarray) and value.ndim > 0:
                    scalar = float(np.mean(value))
                else:
                    scalar = float(value)
                self.writer.add_scalar(f'node_{i}/{name}', scalar, step)
                
    def log_horizon_metrics(self, metrics: Dict[str, np.ndarray], step: int) -> None:
        for name, values in metrics.items():
            for i, value in enumerate(values):
                self.writer.add_scalar(f'horizon_{i}/{name}', value, step)
      
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        node_idx: int = 0, horizon_idx: int = 0) -> None:
        fig = go.Figure()   # type: ignore
        fig.add_trace(go.Scatter(y=y_true[:, node_idx, horizon_idx], name='True'))   # type: ignore
        fig.add_trace(go.Scatter(y=y_pred[:, node_idx, horizon_idx], name='Predicted'))   # type: ignore
        fig.update_layout(title=f'Node {node_idx} - Horizon {horizon_idx}')
        fig.show(renderer="browser")
        
    def plot_metrics(self, metrics_history: Dict[str, List[float]]) -> None:
        fig = make_subplots(rows=3, cols=1, subplot_titles=['MAE', 'MSE', 'RMSE'])
        
        for i, (name, values) in enumerate(metrics_history.items()):
            fig.add_trace(
                go.Scatter(y=values, name=name),   # type: ignore
                row=i+1, col=1
            )
            
        fig.update_layout(height=900, title_text='Training Metrics')
        fig.show()
        
    def close(self) -> None:
        self.writer.close() 