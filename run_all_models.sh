#!/bin/bash

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 创建日志目录
mkdir -p logs

# 定义模型列表
models=("bp" "svm" "knn" "linear" "bagging" "adaboost" "stacking")

# 运行每个模型
for model in "${models[@]}"
do
    echo "=========================================="
    echo "开始训练模型: $model"
    echo "=========================================="
    
    # 运行模型并记录日志
    python train.py --model $model 2>&1 | tee "logs/${model}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "=========================================="
    echo "模型 $model 训练完成"
    echo "=========================================="
    
    # 等待5秒，确保上一个模型完全结束
    sleep 5
done

echo "所有模型训练完成！" 