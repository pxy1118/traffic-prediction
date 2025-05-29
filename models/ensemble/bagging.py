import numpy as np
from typing import List, Type, Optional, Dict
from models.base import BaseModel
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import calculate_metrics

class BaggingEnsemble(BaseModel):
    def __init__(self, base_model_class: Type[BaseModel], n_estimators: int = 10, processor=None, **base_params):
        super().__init__('BaggingEnsemble')
        self.base_model_class = base_model_class
        self.n_estimators = n_estimators
        self.base_params = base_params
        self.models = []
        self.processor = processor
        
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
        for i, model in enumerate(self.models):
            predictions = model.predict(X)
            metrics = calculate_metrics(y, predictions)
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
        print("\nBagging集成模型评估结果:")
        print("-" * 80)
        print(f"{'模型名称':<15} {'MAE':<10} {'MSE':<12} {'RMSE':<10} {'MAPE':<10}")
        print("-" * 80)
        
        # 打印每个基学习器的结果
        for i in range(len(self.models)):
            metrics = results[f'base_model_{i}']
            print(f"{f'基学习器 {i}':<15} {metrics['mae']:<10.4f} {metrics['mse']:<12.4f} "
                  f"{metrics['rmse']:<10.4f} {metrics['mape']:<10.4f}%")
        
        # 打印集成模型的结果
        ensemble_metrics = results['ensemble']
        print("-" * 80)
        print(f"{'集成模型':<15} {ensemble_metrics['mae']:<10.4f} {ensemble_metrics['mse']:<12.4f} "
              f"{ensemble_metrics['rmse']:<10.4f} {ensemble_metrics['mape']:<10.4f}%")
        print("-" * 80)
        
    def fit(self, X: np.ndarray, y: np.ndarray, writer: Optional[SummaryWriter] = None, **kwargs) -> None:
        n_samples = X.shape[0]
        self.models = []
        
        print(f"\n开始训练Bagging集成模型 (n_estimators={self.n_estimators})...")
        for i in range(self.n_estimators):
            print(f"\n训练基模型 {i+1}/{self.n_estimators}")
            # 随机采样训练数据
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # 训练基模型
            model = self.base_model_class(**self.base_params)
            model.fit(X_bootstrap, y_bootstrap)
            self.models.append(model)
            
            # 记录基模型性能
            if writer is not None:
                predictions = model.predict(X)
                # 反归一化预测结果和真实值
                y_denorm = self.processor.inverse_normalize_target(y)
                pred_denorm = self.processor.inverse_normalize_target(predictions)
                metrics = calculate_metrics(y_denorm, pred_denorm)
                for metric_name, value in metrics.items():
                    writer.add_scalar(f'{metric_name}/base_model_{i}', value, 0)
                print(f"基模型 {i+1} 训练完成，MAE: {metrics['MAE']:.4f}")
        
        # 记录最终集成效果
        if writer is not None:
            print("\n计算最终集成效果...")
            y_pred = self.predict(X)
            # 反归一化预测结果和真实值
            y_denorm = self.processor.inverse_normalize_target(y)
            pred_denorm = self.processor.inverse_normalize_target(y_pred)
            metrics = calculate_metrics(y_denorm, pred_denorm)
            for metric_name, value in metrics.items():
                writer.add_scalar(f'{metric_name}/ensemble', value, 0)
            print(f"集成模型 MAE: {metrics['MAE']:.4f}")
            
        self.is_fitted = True
        print("\nBagging集成模型训练完成！")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
            
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
            
        # 对所有基模型的预测结果取平均
        return np.mean(predictions, axis=0)
        
    def save(self, path: str) -> None:
        if self.is_fitted:
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(self.models, f)
                
    def load(self, path: str) -> None:
        import pickle
        with open(path, 'rb') as f:
            self.models = pickle.load(f)
        self.is_fitted = True
        
    def get_params(self) -> dict:
        return {
            'base_model_class': self.base_model_class,
            'n_estimators': self.n_estimators,
            **self.base_params
        }
        
    def set_params(self, **params) -> None:
        for key, value in params.items():
            setattr(self, key, value) 