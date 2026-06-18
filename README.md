# EuroSAT 卫星影像分类系统

基于 ResNet18 迁移学习的 Sentinel-2 卫星影像土地覆盖分类系统。在 GTX 1050 Ti (4 GB 显存) 上实现 10 类土地覆盖分类，测试集准确率 ≥ 93%。

## 环境要求

| 项目 | 规格 |
|------|------|
| GPU | NVIDIA GTX 1050 Ti 4 GB+（支持纯 CPU） |
| 操作系统 | Windows 10/11 64-bit，兼容 Linux |
| Python | 3.8.x |
| CUDA | 11.3 |

## 快速开始

```bash
# 1. 创建环境并安装依赖
conda create -n eurosat python=3.8 -y && conda activate eurosat
# torch/torchvision 从本地 .whl 安装或在线下载（见 INSTALL.md）
pip install -r requirements.txt

# 2. 配置数据路径
# 编辑 configs/config.yaml，修改 data.root_dir 为你的 EuroSAT 路径

# 3. 环境检测
python main.py --mode check

# 4. 训练（50 epoch 快速验证: --epochs 30）
python main.py --mode train

# 5. 评估
python main.py --mode evaluate --model checkpoints/best_model.pth

# 6. 推理
python main.py --mode predict --input test.jpg --model checkpoints/best_model.pth
```

## 性能指标

| 指标 | 数值 |
|------|------|
| 测试集 Top-1 Accuracy | 93.07% |
| 训练显存 (batch=256) | ~1.8 GB |
| 单张推理耗时 | ~30 ms |
| 模型参数量 | 11.2M |

## 目录结构

```
├── main.py              # CLI 统一入口
├── config.py            # 配置管理
├── data.py              # 数据加载与增强
├── model.py             # 模型构建
├── train.py             # 训练模块
├── evaluate.py          # 评估模块
├── predict.py           # 推理模块
├── utils.py             # 工具函数
├── check_env.py         # 环境检测
├── configs/config.yaml  # 配置文件
├── tests/               # 测试
├── docs/                # 文档
├── checkpoints/         # 模型权重
├── logs/                # 训练日志
└── outputs/             # 输出结果
```
