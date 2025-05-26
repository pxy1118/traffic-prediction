import numpy as np
from typing import Type, Optional
from ..base import BaseModel
from torch.utils.tensorboard import SummaryWriter

class AdaBoostEnsemble(BaseModel):
    def __init__(self, base_model_class: Type[BaseModel], n_estimators: int = 10, learning_rate: float = 1.0, **base_params):
        super().__init__('AdaBoostEnsemble')
        self.base_model_class = base_model_class
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_params = base_params
        self.models = []
        self.weights = []
        
    def fit(self, X: np.ndarray, y: np.ndarray, writer: Optional[SummaryWriter] = None, **kwargs) -> None:
        n_samples = X.shape[0]
        self.models = []
        self.weights = []
        
        # 初始化样本权重
        sample_weights = np.ones(n_samples) / n_samples
        
        for i in range(self.n_estimators):
            # 训练基模型
            model = self.base_model_class(**self.base_params)
            model.fit(X, y, sample_weight=sample_weights, writer=writer, output_idx=i)
            self.models.append(model)
            
            # 计算预测误差
            predictions = model.predict(X)
            errors = np.abs(predictions - y)
            if errors.ndim > 1:
                errors = np.mean(errors, axis=1)
            weighted_error = np.sum(sample_weights * errors) / np.sum(sample_weights)
            
            if weighted_error >= 0.5:
                break
                
            alpha = self.learning_rate * 0.5 * np.log((1 - weighted_error) / weighted_error)
            self.weights.append(alpha)
            
            # 更新样本权重
            sample_weights *= np.exp(alpha * errors)
            sample_weights /= np.sum(sample_weights)  # 归一化
            
            # 记录集成效果
            if writer is not None:
                y_pred = self.predict(X)
                loss = np.mean((y_pred - y) ** 2)
                mae = np.mean(np.abs(y_pred - y))
                mse = np.mean((y_pred - y) ** 2)
                rmse = np.sqrt(mse)
                writer.add_scalar('Loss/ensemble', loss, i)
                writer.add_scalar('MAE/ensemble', mae, i)
                writer.add_scalar('MSE/ensemble', mse, i)
                writer.add_scalar('RMSE/ensemble', rmse, i)
        
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
            
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(X)
            predictions.append(pred * weight)
            
        return np.sum(predictions, axis=0) / np.sum(self.weights)
        
    def save(self, path: str) -> None:
        if self.is_fitted:
            import pickle
            with open(path, 'wb') as f:
                pickle.dump((self.models, self.weights), f)
                
    def load(self, path: str) -> None:
        import pickle
        with open(path, 'rb') as f:
            self.models, self.weights = pickle.load(f)
        self.is_fitted = True
        
    def get_params(self) -> dict:
        return {
            'base_model_class': self.base_model_class,
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            **self.base_params
        }
        
    def set_params(self, **params) -> None:
        for key, value in params.items():
            setattr(self, key, value) 