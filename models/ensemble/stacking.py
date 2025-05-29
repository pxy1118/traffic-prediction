import numpy as np
from typing import List, Type, Optional, Dict
from models.base import BaseModel
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import calculate_metrics

class StackingEnsemble(BaseModel):
    def __init__(self, base_models: List[Type[BaseModel]], meta_model: Type[BaseModel], processor=None, **meta_params):
        super().__init__('StackingEnsemble')
        self.base_models = base_models
        self.meta_model = meta_model
        self.meta_params = meta_params
        self.base_params = {}  
        self.trained_base_models = []
        self.trained_meta_model = None
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
        for i, model in enumerate(self.trained_base_models):
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
        print("\n模型评估结果:")
        print("-" * 80)
        print(f"{'模型名称':<15} {'MAE':<10} {'MSE':<12} {'RMSE':<10} {'MAPE':<10}")
        print("-" * 80)
        
        # 打印每个基学习器的结果
        for i, model in enumerate(self.trained_base_models):
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
        self.trained_base_models = []
        
        # 第一层：训练基模型
        meta_features = []
        for i, model_class in enumerate(self.base_models):
            print(f"\n训练基模型 {i+1}/{len(self.base_models)}: {model_class.__name__}")
            model = model_class(**self.base_params)
            model.fit(X, y, writer=writer, output_idx=i)
            self.trained_base_models.append(model)
            
            # 生成元特征
            predictions = model.predict(X)
            meta_features.append(predictions)
            
            # 记录基模型性能
            if writer is not None:
                # 反归一化预测结果和真实值
                y_denorm = self.processor.inverse_normalize_target(y)
                pred_denorm = self.processor.inverse_normalize_target(predictions)
                metrics = calculate_metrics(y_denorm, pred_denorm)
                for metric_name, value in metrics.items():
                    writer.add_scalar(f'{metric_name}/base_model_{i}', value, 0)
                print(f"基模型 {i+1} 训练完成，MAE: {metrics['MAE']:.4f}")
        
        print("\n开始训练元模型...")
        # 拼接元特征
        meta_X = np.column_stack(meta_features)
        
        # 第二层：训练元模型
        self.trained_meta_model = self.meta_model(**self.meta_params)
        self.trained_meta_model.fit(meta_X, y, writer=writer, output_idx=len(self.base_models))
        print("元模型训练完成！")
        
        # 记录最终集成效果
        if writer is not None:
            print("\n计算最终集成效果...")
            # 使用训练好的模型进行预测
            meta_features = []
            for model in self.trained_base_models:
                predictions = model.predict(X)
                meta_features.append(predictions)
            meta_X = np.column_stack(meta_features)
            y_pred = self.trained_meta_model.predict(meta_X)
            
            # 反归一化预测结果和真实值
            y_denorm = self.processor.inverse_normalize_target(y)
            pred_denorm = self.processor.inverse_normalize_target(y_pred)
            metrics = calculate_metrics(y_denorm, pred_denorm)
            for metric_name, value in metrics.items():
                writer.add_scalar(f'{metric_name}/ensemble', value, 0)
            print(f"集成模型 MAE: {metrics['MAE']:.4f}")
        
        self.is_fitted = True
        print("\nStacking集成模型训练完成！")
        
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