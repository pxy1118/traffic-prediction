import os
import argparse
import numpy as np
import subprocess
import sys
from typing import Dict, Any, Optional
from models.ml_models.svm import MultiOutputSVM
from models.ml_models.knn import MultiOutputKNN
from models.ml_models.linear import MultiOutputLinear
from models.ml_models.bp import MultiOutputBP
from models.ensemble.bagging import BaggingEnsemble
from models.ensemble.adaboost import AdaBoostEnsemble
from models.ensemble.stacking import StackingEnsemble
from utils.data_processor import DataProcessor
from utils.metrics import calculate_metrics, calculate_node_metrics, calculate_horizon_metrics
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def adj_to_edge_index(adj: np.ndarray) -> torch.Tensor:
    src, dst = np.nonzero(adj)
    edge_index = np.vstack((src, dst))
    return torch.LongTensor(edge_index)

def get_model(model_name: str, input_size: Optional[int] = None, **kwargs) -> Any:
    models = {
        'svm': MultiOutputSVM,
        'knn': MultiOutputKNN,
        'linear': MultiOutputLinear,
        'bp': MultiOutputBP,
        'bagging': BaggingEnsemble,
        'adaboost': AdaBoostEnsemble,
        'stacking': StackingEnsemble
    }
    
    if model_name == 'bp' and input_size is not None:
        kwargs['input_size'] = input_size
    
    if model_name in ['bagging', 'adaboost'] and 'base_model_class' not in kwargs:
        if model_name == 'bagging':
            kwargs['base_model_class'] = MultiOutputBP
            kwargs['n_estimators'] = 15
            if input_size is not None:
                kwargs['input_size'] = input_size
                kwargs['hidden_sizes'] = [64, 32]
        else:
            kwargs['base_model_class'] = MultiOutputLinear
    
    if model_name == 'stacking' and 'base_models' not in kwargs:
        kwargs['base_models'] = [MultiOutputLinear, MultiOutputKNN]
        kwargs['meta_model'] = MultiOutputLinear
    
    return models[model_name](**kwargs)

def train(args):
    # 创建TensorBoard日志目录
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('runs', f'{args.model}_{current_time}')
    writer = SummaryWriter(log_dir)
    
    # 数据预处理
    processor = DataProcessor(args.data_path, args.seq_len, args.pred_len)
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), adj = processor.prepare_data()
    
    edge_index = adj_to_edge_index(adj)
    # GCN特征提取，节点级一一对应
    X_train_gcn = processor.extract_gcn_features(X_train[:, -1, :, :], edge_index)
    X_val_gcn = processor.extract_gcn_features(X_val[:, -1, :, :], edge_index)
    X_test_gcn = processor.extract_gcn_features(X_test[:, -1, :, :], edge_index)
    
    n_train_samples, n_nodes, gcn_dim = X_train_gcn.shape
    n_val_samples = X_val_gcn.shape[0]
    n_test_samples = X_test_gcn.shape[0]
    
    X_train = X_train_gcn.reshape(n_train_samples * n_nodes, gcn_dim)
    X_val = X_val_gcn.reshape(n_val_samples * n_nodes, gcn_dim)
    X_test = X_test_gcn.reshape(n_test_samples * n_nodes, gcn_dim)
    
    Y_train = Y_train.reshape(n_train_samples * n_nodes, args.pred_len)
    Y_val = Y_val.reshape(n_val_samples * n_nodes, args.pred_len)
    Y_test = Y_test.reshape(n_test_samples * n_nodes, args.pred_len)
    
    # 限制训练集规模，加速调试
    max_train = 200
    X_train, Y_train = X_train[:max_train], Y_train[:max_train]    
    
    # 初始化模型
    model = get_model(args.model, input_size=gcn_dim, **args.model_params)
    print("================ 训练任务信息 ================")
    print(f"模型类型: {args.model}")
    print(f"模型参数: {args.model_params}")
    print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
    print(f"X_val shape: {X_val.shape}, Y_val shape: {Y_val.shape}")
    print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")
    print(f"节点数: {n_nodes}, 输入特征数: {gcn_dim}, 预测步数: {args.pred_len}")
    print("=============================================")
    
    # 训练模型
    model.fit(X_train, Y_train, writer=writer)
    
    val_pred = model.predict(X_val)
    val_pred = val_pred.reshape(n_val_samples, n_nodes, args.pred_len)
    Y_val = Y_val.reshape(n_val_samples, n_nodes, args.pred_len)
    val_pred = processor.inverse_normalize_target(val_pred)
    y_val = processor.inverse_normalize_target(Y_val)
    val_metrics = calculate_metrics(y_val, val_pred)
    
    y_pred = model.predict(X_test)
    y_pred = y_pred.reshape(n_test_samples, n_nodes, args.pred_len)
    Y_test = Y_test.reshape(n_test_samples, n_nodes, args.pred_len)
    y_pred = processor.inverse_normalize_target(y_pred)
    y_test = processor.inverse_normalize_target(Y_test)
    
    metrics = calculate_metrics(y_test, y_pred)
    node_metrics = calculate_node_metrics(y_test, y_pred)
    horizon_metrics = calculate_horizon_metrics(y_test, y_pred)
    
    for name, value in metrics.items():
        writer.add_scalar(f'final_test/{name}', value, 0)
    for name, value in val_metrics.items():
        writer.add_scalar(f'final_val/{name}', value, 0)
    
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    model.save(os.path.join('saved_models', f'{args.model}.pkl'))
    
    print(f'Model: {args.model}')
    print('Overall metrics:')
    for name, value in metrics.items():
        print(f'{name}: {value:.4f}')
    
    os.makedirs('app/data', exist_ok=True)
    np.save('app/data/predictions.npy', y_pred)
    np.save('app/data/y_true.npy', y_test)
    
    print("\n================ 训练完成 ================")
    print(f"TensorBoard日志保存在: {log_dir}")
    print("使用以下命令查看训练过程:")
    print(f"tensorboard --logdir={log_dir}")
    print("正在启动可视化应用...")
    print("请在浏览器中访问: http://127.0.0.1:8050")
    print("========================================")
    
    # 自动启动app.py
    try:
        os.chdir('app')
        subprocess.run([sys.executable, 'app.py'], check=True)
    except KeyboardInterrupt:
        print("\n可视化应用已停止")
    except Exception as e:
        print(f"启动可视化应用时出错: {e}")
        print("请手动运行: cd app && python app.py")
    
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/PEMS-08/pems08.npz')
    parser.add_argument('--model', type=str, default='stacking', choices=['svm', 'knn', 'linear', 'bp', 'bagging', 'adaboost', 'stacking'])
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--model_params', type=str, default='{}')
    args = parser.parse_args()
    
    import json
    args.model_params = json.loads(args.model_params)
    
    train(args) 