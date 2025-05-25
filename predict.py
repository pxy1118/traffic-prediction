import os
import numpy as np
import joblib
from utils.data_processor import DataProcessor

def load_models():
    """加载所有训练好的模型"""
    models = {}
    model_dir = 'saved_models'
    
    for filename in os.listdir(model_dir):
        if filename.endswith('_model.joblib'):
            name = filename.replace('_model.joblib', '')
            model_path = os.path.join(model_dir, filename)
            models[name] = joblib.load(model_path)
    
    return models

def ensemble_predict(models, X, weights=None):
    """使用集成方法进行预测
    
    Args:
        models: 字典，包含所有模型
        X: 输入数据
        weights: 各模型的权重，如果为None则使用相等权重
        
    Returns:
        y_pred: 集成预测结果
    """
    if weights is None:
        weights = {name: 1.0/len(models) for name in models.keys()}
    
    # 重塑数据为2D
    n_samples = X.shape[0]
    X_2d = X.reshape(n_samples, -1)
    
    # 获取每个模型的预测
    predictions = {}
    for name, model in models.items():
        pred = model.predict(X_2d)
        predictions[name] = pred.reshape(n_samples, -1)
    
    # 计算加权平均
    y_pred = np.zeros_like(predictions[list(models.keys())[0]])
    for name, pred in predictions.items():
        y_pred += weights[name] * pred
    
    return y_pred

def main():
    # 加载数据
    data_path = 'data/PEMS-08/pems08.npz'
    adj_path = 'data/PEMS-08/adj_mx.npy'
    
    print("Loading data...")
    processor = DataProcessor(data_path)
    x, adj = processor.load_data()
    x = processor.normalize(x)
    
    # 准备时间序列数据
    X, y = processor.create_sequences(x)
    
    # 加载模型
    print("Loading models...")
    models = load_models()
    
    # 设置模型权重
    weights = {name: 1.0/len(models) for name in models.keys()}
    
    # 进行预测
    print("Making predictions...")
    y_pred = ensemble_predict(models, X[:, -1, :, :], weights)  # 只用最后一帧做预测
    
    # 保存预测结果
    np.save('predictions.npy', y_pred)
    print("Predictions saved to predictions.npy")

if __name__ == '__main__':
    main() 