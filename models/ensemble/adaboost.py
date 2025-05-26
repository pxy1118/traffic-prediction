import numpy as np
from typing import List, Type, Optional, Dict
from ..base import BaseModel
from torch.utils.tensorboard import SummaryWriter
from ...utils.metrics import calculate_metrics

class AdaBoostEnsemble(BaseModel):
    def __init__(self, base_model_class: Type[BaseModel], n_estimators: int = 10, learning_rate: float = 1.0, **base_params):
        super().__init__('AdaBoostEnsemble')
        self.base_model_class = base_model_class
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_params = base_params
        self.models = []
        self.weights = []
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """评估所有基学习器和集成模型的性能
        
        Args:
            X: 输入特征
            y: 真实标签
            
        Returns:
            Dict: 包含所有模型评估指标的字典
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
            
        results = {}
        
        # 评估每个基学习器
        for i, (model, weight) in enumerate(zip(self.models, self.weights)):
            predictions = model.predict(X)
            metrics = calculate_metrics(y, predictions)
            metrics['weight'] = weight  # 添加权重信息
            results[f'base_model_{i}'] = metrics
            
        # 评估集成模型
        ensemble_predictions = self.predict(X)
        ensemble_metrics = calculate_metrics(y, ensemble_predictions)
        results['ensemble'] = ensemble_metrics
        
        return results
        
    def print_evaluation_table(self, X: np.ndarray, y: np.ndarray) -> None:
        """打印评估结果表格
        
        Args:
            X: 输入特征
            y: 真实标签
        """
        results = self.evaluate(X, y)
        
        # 打印表头
        print("\nAdaBoost集成模型评估结果:")
        print("-" * 90)
        print(f"{'模型名称':<15} {'MAE':<10} {'MSE':<12} {'RMSE':<10} {'MAPE':<10} {'权重':<10}")
        print("-" * 90)
        
        # 打印每个基学习器的结果
        for i in range(len(self.models)):
            metrics = results[f'base_model_{i}']
            print(f"{f'基学习器 {i}':<15} {metrics['mae']:<10.4f} {metrics['mse']:<12.4f} "
                  f"{metrics['rmse']:<10.4f} {metrics['mape']:<10.4f}% {metrics['weight']:<10.4f}")
        
        # 打印集成模型的结果
        ensemble_metrics = results['ensemble']
        print("-" * 90)
        print(f"{'集成模型':<15} {ensemble_metrics['mae']:<10.4f} {ensemble_metrics['mse']:<12.4f} "
              f"{ensemble_metrics['rmse']:<10.4f} {ensemble_metrics['mape']:<10.4f}%")
        print("-" * 90)
        
    def fit(self, X: np.ndarray, y: np.ndarray, writer: Optional[SummaryWriter] = None, **kwargs) -> None:
        n_samples = X.shape[0]
        self.models = []
        self.weights = []
        
        # 初始化样本权重
        sample_weights = np.ones(n_samples) / n_samples
        
        for i in range(self.n_estimators):
            # 训练基模型
            model = self.base_model_class(**self.base_params)
            model.fit(X, y, sample_weight=sample_weights)
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
            
            # 记录基模型性能
            if writer is not None:
                metrics = calculate_metrics(y, predictions)
                for metric_name, value in metrics.items():
                    writer.add_scalar(f'{metric_name}/base_model_{i}', value, 0)
                writer.add_scalar('weight/base_model_{i}', alpha, 0)
        
        # 记录最终集成效果
        if writer is not None:
            y_pred = self.predict(X)
            metrics = calculate_metrics(y, y_pred)
            for metric_name, value in metrics.items():
                writer.add_scalar(f'{metric_name}/ensemble', value, 0)
        
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