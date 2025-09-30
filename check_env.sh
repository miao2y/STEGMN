#!/bin/bash

echo "========================================"
echo "      实验环境一键查询脚本"
echo "========================================"

# 操作系统
echo "【操作系统】"
uname -srm
cat /etc/os-release 2>/dev/null | grep "PRETTY_NAME" | cut -d'"' -f2
echo

# CPU
echo "【CPU】"
lscpu | grep "Model name" | sed 's/  */ /g'
echo

# 内存
echo "【内存】"
free -h | grep "Mem:" | awk '{print $2}'" RAM"
echo

# GPU
echo "【GPU】"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,cuda_version,memory.total --format=csv,noheader,nounits
else
    echo "NVIDIA 驱动未安装或无 GPU"
fi
echo

# CUDA
echo "【CUDA 版本】"
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release" | awk '{print $6}' | sed 's/,//'
else
    echo "CUDA Toolkit 未安装"
fi
echo

# Python
echo "【Python 版本】"
python3 --version
echo

# PyTorch
echo "【PyTorch 信息】"
python3 -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
print(f'CUDA 版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')
print(f'GPU 数量: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'当前 GPU: {torch.cuda.get_device_name(0)}')
"
echo

# 主要依赖库
echo "【关键依赖库版本】"
pip list | grep -i "torch\|numpy\|scipy\|matplotlib\|mdanalysis\|pyg\|dgl\|tensorboard" || echo "pip list 查询失败（可能未激活环境）"
echo

echo "========================================"
echo "请将以上全部内容复制并发送给论文助手"
echo "========================================"
