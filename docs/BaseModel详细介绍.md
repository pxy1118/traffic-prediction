# 第3章  系统架构与BaseModel设计

## 3.1 系统整体架构

### 3.1.1 架构设计理念

本交通流预测系统采用模块化设计思想，将复杂的预测任务分解为数据处理、特征提取、模型训练、集成学习和结果可视化等独立模块。系统架构的核心是BaseModel抽象基类，它为所有学习器提供了统一的接口规范，实现了算法的可插拔性和扩展性。

**设计原则：**
1) **统一接口**：所有模型遵循相同的训练和预测接口
2) **模块解耦**：各功能模块相互独立，降低系统复杂度
3) **易于扩展**：新算法可以轻松集成到现有框架中
4) **配置驱动**：支持通过配置文件灵活调整模型参数

### 3.1.2 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    交通流预测系统架构                              │
├─────────────────────────────────────────────────────────────────┤
│  数据层 (Data Layer)                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ PEMS-08数据 │  │ 邻接矩阵    │  │ 节点坐标    │            │
│  │ 17856×170×3 │  │ 170×170     │  │ GPS信息     │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
├─────────────────────────────────────────────────────────────────┤
│  处理层 (Processing Layer)                                      │
│  ┌─────────────────┐  ┌─────────────────┐                     │
│  │ DataProcessor   │  │ GCNFeatureExtractor │                │
│  │ - 数据加载      │  │ - 图特征提取        │                │
│  │ - 标准化        │  │ - 空间建模          │                │
│  │ - 序列构造      │  │ - 降维表示          │                │
│  └─────────────────┘  └─────────────────┘                     │
├─────────────────────────────────────────────────────────────────┤
│  模型层 (Model Layer)                                           │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                BaseModel (抽象基类)                          ││
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐         ││
│  │  │ fit()       │ │ predict()   │ │ save/load() │         ││
│  │  └─────────────┘ └─────────────┘ └─────────────┘         ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐              │
│  │   SVM   │ │   KNN   │ │ Linear  │ │   BP    │              │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│  │  Bagging    │ │  AdaBoost   │ │  Stacking   │            │
│  └─────────────┘ └─────────────┘ └─────────────┘            │
├─────────────────────────────────────────────────────────────────┤
│  评估层 (Evaluation Layer)                                      │
│  ┌─────────────────┐  ┌─────────────────┐                     │
│  │ Metrics计算     │  │ 可视化展示      │                     │
│  │ - MAE/MSE/RMSE  │  │ - Dash Web界面  │                     │
│  │ - 节点级评估    │  │ - 交互式图表    │                     │
│  │ - 时间步评估    │  │ - 模型对比      │                     │
│  └─────────────────┘  └─────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

## 3.2 BaseModel抽象基类设计

### 3.2.1 设计目标与原则

BaseModel抽象基类是整个系统的核心组件，它定义了所有机器学习模型必须遵循的统一接口。设计目标包括：

**统一性目标：**
- 为所有算法提供一致的训练和预测接口
- 统一模型参数管理和持久化机制
- 标准化模型状态管理和错误处理

**扩展性目标：**
- 支持新算法的快速集成
- 允许算法特定的参数定制
- 便于集成学习方法的组合

**可维护性目标：**
- 清晰的抽象层次和职责分离
- 完善的文档和类型注解
- 便于单元测试和调试

### 3.2.2 类结构设计

```python
from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

class BaseModel(ABC):
    """
    交通流预测模型的抽象基类
    
    所有预测模型必须继承此基类并实现其抽象方法，
    确保系统中所有模型具有统一的接口和行为。
    """
    
    def __init__(self, name: str):
        """
        初始化基础模型
        
        Args:
            name: 模型名称，用于标识和日志记录
        """
        self.name = name
        self.is_fitted = False  # 模型训练状态标记
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        训练模型的抽象方法
        
        Args:
            X: 输入特征矩阵，形状为 (n_samples, n_features)
            y: 目标值矩阵，形状为 (n_samples, n_outputs)
            **kwargs: 算法特定的训练参数
        """
        pass
        
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测方法的抽象接口
        
        Args:
            X: 输入特征矩阵，形状为 (n_samples, n_features)
            
        Returns:
            预测结果矩阵，形状为 (n_samples, n_outputs)
        """
        pass
```

### 3.2.3 核心方法详解

#### 3.2.3.1 训练方法 (fit)

`fit` 方法是模型训练的核心接口，所有子类必须实现此方法：

**接口规范：**
- **输入格式**：X为32维GCN特征，y为12维预测目标
- **状态管理**：训练完成后设置 `is_fitted = True`
- **参数传递**：通过 `**kwargs` 支持算法特定参数
- **异常处理**：捕获并处理训练过程中的错误

**实现示例：**
```python
def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
    """SVR模型的训练实现"""
    n_outputs = y.shape[1]  # 12个输出
    self.models = []
    
    # 为每个输出维度训练独立的SVR
    for i in range(n_outputs):
        model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon)
        model.fit(X, y[:, i])
        self.models.append(model)
        
    self.is_fitted = True
```

#### 3.2.3.2 预测方法 (predict)

`predict` 方法执行模型推理，返回预测结果：

**安全检查：**
```python
def predict(self, X: np.ndarray) -> np.ndarray:
    if not self.is_fitted:
        raise ValueError("Model not fitted yet")
    # 执行预测逻辑
```

**输出规范：**
- 返回形状为 `(n_samples, 12)` 的预测矩阵
- 对应未来12个时间步的交通流量预测
- 数值类型为 `float64`，确保精度

#### 3.2.3.3 模型持久化

BaseModel提供了模型保存和加载的默认实现：

```python
def save(self, path: str) -> None:
    """保存模型到指定路径"""
    import joblib
    if hasattr(self, 'model'):
        joblib.dump(self.model, path)
    elif hasattr(self, 'models'):
        joblib.dump(self.models, path)
        
def load(self, path: str) -> None:
    """从指定路径加载模型"""
    import joblib
    loaded_obj = joblib.load(path)
    if isinstance(loaded_obj, list):
        self.models = loaded_obj
    else:
        self.model = loaded_obj
    self.is_fitted = True
```

#### 3.2.3.4 参数管理

统一的参数获取和设置接口：

```python
def get_params(self) -> Dict[str, Any]:
    """获取模型参数字典"""
    return {
        'name': self.name,
        'is_fitted': self.is_fitted
    }
    
def set_params(self, **params) -> None:
    """设置模型参数"""
    for key, value in params.items():
        if hasattr(self, key):
            setattr(self, key, value)
```

## 3.3 数据处理与特征工程

### 3.3.1 DataProcessor设计

DataProcessor类负责原始交通数据的加载、预处理和特征工程：

**核心功能：**
1) **数据加载**：从PEMS-08数据集加载交通流数据
2) **数据标准化**：使用StandardScaler进行Z-score标准化
3) **序列构造**：构建时间序列输入-输出对
4) **数据分割**：按7:1.5:1.5比例分割训练/验证/测试集

**关键方法实现：**

```python
def create_sequences(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    构建时间序列数据
    
    输入：(时间步, 节点数, 特征数)
    输出：X(样本数, 12, 节点数, 特征数), Y(样本数, 节点数, 12)
    """
    n_samples = len(x) - self.seq_len - self.pred_len + 1
    n_nodes, n_features = x.shape[1], x.shape[2]
    
    X = np.zeros((n_samples, self.seq_len, n_nodes, n_features))
    Y = np.zeros((n_samples, n_nodes, self.pred_len))
    
    for i in range(n_samples):
        X[i] = x[i:i+self.seq_len]  # 历史12步
        Y[i] = x[i+self.seq_len:i+self.seq_len+self.pred_len, :, 0].T
        
    return X, Y
```

### 3.3.2 GCN特征提取集成

DataProcessor集成了GCN特征提取功能：

```python
def extract_gcn_features(self, x: np.ndarray, edge_index: torch.Tensor) -> np.ndarray:
    """
    使用GCN提取空间特征
    
    Args:
        x: 原始节点特征 (n_samples, n_nodes, n_features)
        edge_index: 图的边索引
        
    Returns:
        GCN特征 (n_samples, n_nodes, 32)
    """
    if self.gcn_extractor is None:
        self.gcn_extractor = GCNFeatureExtractor(
            in_channels=x.shape[-1],
            hidden_channels=64,
            out_channels=32
        )
    
    # 重塑数据并提取特征
    n_samples, n_nodes, n_features = x.shape
    x_reshaped = x.reshape(-1, n_features)
    features = self.gcn_extractor.transform(x_reshaped, edge_index)
    
    return features.reshape(n_samples, n_nodes, -1)
```

## 3.4 评估体系设计

### 3.4.1 多层次评估指标

系统设计了三个层次的评估指标：

**1) 全局评估 (calculate_metrics)**
```python
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """计算整体预测性能"""
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}
```

**2) 节点级评估 (calculate_node_metrics)**
```python
def calculate_node_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    """计算每个传感器节点的预测性能"""
    # 在时间维度上聚合，保留节点维度
    mae = np.mean(np.abs(y_true - y_pred), axis=0)  # (n_nodes, n_steps)
    mse = np.mean((y_true - y_pred) ** 2, axis=0)
    rmse = np.sqrt(mse)
    
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}
```

**3) 时间步评估 (calculate_horizon_metrics)**
```python
def calculate_horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    """计算每个预测时间步的性能"""
    # 在样本和节点维度上聚合，保留时间维度
    mae = np.mean(np.abs(y_true - y_pred), axis=(0, 1))  # (n_steps,)
    mse = np.mean((y_true - y_pred) ** 2, axis=(0, 1))
    rmse = np.sqrt(mse)
    
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}
```

### 3.4.2 评估指标解释

**平均绝对误差 (MAE)：**
$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

- 直观反映预测值与真实值的平均偏差
- 对异常值不敏感，稳定性好
- 单位与原始数据相同，易于理解

**均方误差 (MSE)：**
$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

- 对大误差敏感，能突出严重的预测错误
- 数学性质良好，便于优化
- 单位为原始数据的平方

**均方根误差 (RMSE)：**
$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

- 兼具MSE对大误差的敏感性和MAE的直观性
- 单位与原始数据相同
- 在回归任务中广泛使用

## 3.5 训练流程设计

### 3.5.1 统一训练框架

系统设计了标准化的训练流程，确保所有模型的训练过程一致：

```python
def train_model_pipeline(model_name: str, config: Dict[str, Any]):
    """统一的模型训练流程"""
    
    # 1. 数据准备
    processor = DataProcessor(config['data_path'])
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), adj = processor.prepare_data()
    
    # 2. 特征提取
    edge_index = adj_to_edge_index(adj)
    X_train_gcn = processor.extract_gcn_features(X_train[:, -1, :, :], edge_index)
    X_test_gcn = processor.extract_gcn_features(X_test[:, -1, :, :], edge_index)
    
    # 3. 数据重塑
    n_samples, n_nodes, gcn_dim = X_train_gcn.shape
    X_train = X_train_gcn.reshape(n_samples * n_nodes, gcn_dim)
    X_test = X_test_gcn.reshape(len(X_test) * n_nodes, gcn_dim)
    
    # 4. 模型训练
    model = get_model(model_name, input_size=gcn_dim, **config.get('model_params', {}))
    model.fit(X_train, Y_train.reshape(-1, 12))
    
    # 5. 预测评估
    y_pred = model.predict(X_test)
    y_pred = y_pred.reshape(-1, n_nodes, 12)
    Y_test = Y_test.reshape(-1, n_nodes, 12)
    
    # 6. 反标准化
    y_pred = processor.inverse_normalize_target(y_pred)
    y_test = processor.inverse_normalize_target(Y_test)
    
    # 7. 性能评估
    metrics = calculate_metrics(y_test, y_pred)
    
    return model, metrics
```

### 3.5.2 模型工厂模式

使用工厂模式统一模型创建过程：

```python
def get_model(model_name: str, input_size: Optional[int] = None, **kwargs) -> BaseModel:
    """模型工厂函数"""
    models = {
        'svm': MultiOutputSVM,
        'knn': MultiOutputKNN,
        'linear': MultiOutputLinear,
        'bp': MultiOutputBP,
        'bagging': BaggingEnsemble,
        'adaboost': AdaBoostEnsemble,
        'stacking': StackingEnsemble
    }
    
    # 处理算法特定的参数
    if model_name == 'bp' and input_size is not None:
        kwargs['input_size'] = input_size
        
    # 配置集成学习参数
    if model_name == 'bagging':
        kwargs.setdefault('base_model_class', MultiOutputBP)
        kwargs.setdefault('n_estimators', 15)
        
    return models[model_name](**kwargs)
```

## 3.6 系统可扩展性设计

### 3.6.1 新算法集成步骤

得益于BaseModel的统一接口设计，集成新算法只需要三个步骤：

**步骤1：实现BaseModel接口**
```python
class NewAlgorithm(BaseModel):
    def __init__(self, **params):
        super().__init__('NewAlgorithm')
        # 初始化算法特定参数
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        # 实现训练逻辑
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        # 实现预测逻辑
        return predictions
```

**步骤2：注册到模型工厂**
```python
# 在get_model函数中添加
models['new_algorithm'] = NewAlgorithm
```

**步骤3：配置训练参数**
```python
# 在配置文件中添加
{
    "new_algorithm": {
        "param1": value1,
        "param2": value2
    }
}
```

### 3.6.2 模块化设计优势

**松耦合架构：**
- 数据处理与模型训练相互独立
- 特征提取与预测算法解耦
- 评估体系独立于具体算法

**配置驱动：**
- 通过配置文件控制实验参数
- 支持批量实验和参数网格搜索
- 便于复现和比较实验结果

**接口标准化：**
- 统一的数据格式和接口规范
- 一致的错误处理和日志记录
- 标准化的性能评估流程

这种设计使得系统具有良好的可维护性和可扩展性，为未来引入新算法和功能提供了坚实的基础。