# EuroSAT 卫星影像分类系统

基于 **ResNet18 迁移学习** 的 Sentinel-2 卫星影像土地覆盖分类系统。在 GTX 1050 Ti (4 GB) 上实现 10 类土地覆盖分类，**测试集准确率 93.07%**。

## 快速开始

```bash
# 1. 环境
conda create -n eurosat python=3.8 -y && conda activate eurosat
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

# 2. 配置数据路径 — 编辑 configs/config.yaml 中的 data.root_dir

# 3. 环境检测
python main.py --mode check

# 4. 训练（可选 --epochs 30 快速验证）
python main.py --mode train

# 5. 评估
python main.py --mode evaluate --model checkpoints/best_model.pth

# 6. 推理
python main.py --mode predict --input test.jpg --model checkpoints/best_model.pth
```

## 环境要求

| 项目 | 规格 |
|------|------|
| GPU | NVIDIA GTX 1050 Ti 4 GB+（支持纯 CPU） |
| 操作系统 | Windows 10/11 64-bit，兼容 Linux |
| Python | 3.8.x |
| PyTorch | 1.12.1 + CUDA 11.3 |
| 磁盘 | ≥ 10 GB（含 EuroSAT 数据集约 2 GB） |

## 性能

| 指标 | 数值 |
|------|------|
| 测试集 Top-1 Accuracy | **93.07%** |
| 训练显存 (batch=256) | ~1.81 GB |
| 单张推理耗时 (GPU) | ~30 ms |
| 模型参数量 | 11.18M |
| 训练耗时 (70 epoch) | ~75 分钟 |

## 目录结构

```
├── main.py              # CLI 统一入口（train/evaluate/predict/check）
├── config.py            # 配置管理（7 个 frozen dataclass）
├── data.py              # 数据加载 + 7:1:2 划分 + 增强 + DataLoader
├── model.py             # ResNet18 构建 + 预训练 + 冻结/解冻 + checkpoint
├── train.py             # 训练循环 + 验证 + 早停 + OOM 恢复
├── evaluate.py          # 评估指标 + 混淆矩阵 + 报告生成
├── predict.py           # 单张/批量推理 + 通道自适应
├── utils.py             # TensorBoard + 可视化 + GPU 监控 + 随机种子
├── check_env.py         # 一键环境检测（6 项）
├── configs/config.yaml  # 配置文件（唯一配置入口）
├── tests/               # 单元测试（29 项）+ E2E 测试
├── docs/                # 用户文档
├── checkpoints/         # 模型权重（运行时生成）
├── logs/                # 训练日志 + TensorBoard（运行时生成）
└── outputs/             # 评估/推理结果（运行时生成）
```

## 文档

| 文档 | 说明 |
|------|------|
| [INSTALL.md](INSTALL.md) | 详细安装指南 |
| [USER_GUIDE.md](USER_GUIDE.md) | 使用指南（4 种模式详解） |
| [CONFIG_GUIDE.md](CONFIG_GUIDE.md) | 配置参数说明 + 典型场景 |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | 常见问题排查 |
| [CHANGELOG.md](CHANGELOG.md) | 版本更新记录 |

## 许可

本项目仅用于研究和学习目的。EuroSAT 数据集由 Helber et al. (2019) 发布。
