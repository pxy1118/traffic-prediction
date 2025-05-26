import numpy as np
from typing import List, Type, Optional
from ..base import BaseModel
from torch.utils.tensorboard import SummaryWriter

class StackingEnsemble(BaseModel):
    def __init__(self, base_models: List[Type[BaseModel]], meta_model: Type[BaseModel], **meta_params):
        super().__init__('StackingEnsemble')
        self.base_models = base_models
        self.meta_model = meta_model
        self.meta_params = meta_params
        self.trained_base_models = []
        self.trained_meta_model = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, writer: Optional[SummaryWriter] = None, **kwargs) -> None:
        n_samples = X.shape[0]
        self.trained_base_models = []
        
        # 第一层：训练基模型
        meta_features = []
        for i, model_class in enumerate(self.base_models):
            model = model_class(name=f'base_model_{i}')
            model.fit(X, y, writer=writer, output_idx=i)
            self.trained_base_models.append(model)
            
            # 生成元特征
            predictions = model.predict(X)
            meta_features.append(predictions)
            
            # 记录基模型性能
            if writer is not None:
                loss = np.mean((predictions - y) ** 2)
                mae = np.mean(np.abs(predictions - y))
                mse = np.mean((predictions - y) ** 2)
                rmse = np.sqrt(mse)
                writer.add_scalar(f'Loss/base_model_{i}', loss, 0)
                writer.add_scalar(f'MAE/base_model_{i}', mae, 0)
                writer.add_scalar(f'MSE/base_model_{i}', mse, 0)
                writer.add_scalar(f'RMSE/base_model_{i}', rmse, 0)
        
        # 拼接元特征
        meta_X = np.column_stack(meta_features)
        
        # 第二层：训练元模型
        self.trained_meta_model = self.meta_model(**self.meta_params)
        self.trained_meta_model.fit(meta_X, y, writer=writer, output_idx=len(self.base_models))
        
        # 记录最终集成效果
        if writer is not None:
            y_pred = self.predict(X)
            loss = np.mean((y_pred - y) ** 2)
            mae = np.mean(np.abs(y_pred - y))
            mse = np.mean((y_pred - y) ** 2)
            rmse = np.sqrt(mse)
            writer.add_scalar('Loss/ensemble', loss, 0)
            writer.add_scalar('MAE/ensemble', mae, 0)
            writer.add_scalar('MSE/ensemble', mse, 0)
            writer.add_scalar('RMSE/ensemble', rmse, 0)
        
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.trained_meta_model is None:
            raise ValueError("Model not fitted yet")
            
        # 生成元特征
        meta_features = []
        for model in self.trained_base_models:
            predictions = model.predict(X)
            meta_features.append(predictions)
            
        # 拼接元特征
        meta_X = np.column_stack(meta_features)
        
        return self.trained_meta_model.predict(meta_X)
        
    def save(self, path: str) -> None:
        if self.is_fitted:
            import pickle
            with open(path, 'wb') as f:
                pickle.dump((self.trained_base_models, self.trained_meta_model), f)
                
    def load(self, path: str) -> None:
        import pickle
        with open(path, 'rb') as f:
            self.trained_base_models, self.trained_meta_model = pickle.load(f)
        self.is_fitted = True
        
    def get_params(self) -> dict:
        return {
            'base_models': self.base_models,
            'meta_model': self.meta_model,
            **self.meta_params
        }
        
    def set_params(self, **params) -> None:
        for key, value in params.items():
            setattr(self, key, value) 