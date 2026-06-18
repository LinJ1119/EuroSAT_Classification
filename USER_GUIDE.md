# 用户使用指南

## 一、CLI 总览

```bash
python main.py --mode <train|evaluate|predict|check> [选项...]
```

## 二、四种模式

### 训练模式

```bash
# 使用默认配置完整训练 (50 epoch)
python main.py --mode train

# 快速验证 (2 epoch)
python main.py --mode train --epochs 2

# 自定义数据路径
python main.py --mode train --data-dir "D:/MyData/EuroSAT"

# 从检查点恢复/微调（基于之前训练的模型继续）
python main.py --mode train --epochs 70 --lr 0.00001 --resume checkpoints/best_model.pth
```

**常用 CLI 参数**：

| 参数 | 说明 | 示例 |
|------|------|------|
| `--epochs` | 训练轮数 | `--epochs 30` |
| `--batch-size` | 批大小 | `--batch-size 128` |
| `--lr` | 学习率 | `--lr 0.0001` |
| `--resume` | 继续训练 | `--resume checkpoints/best_model.pth` |
| `--data-dir` | 数据路径 | `--data-dir "D:/Data/EuroSAT"` |

**训练期间监控**：
```bash
# 另开终端
tensorboard --logdir logs/
# 浏览器打开 http://localhost:6006
```

**输出**：
- `checkpoints/best_model.pth` — 最佳模型
- `checkpoints/checkpoint_epoch_XXX.pth` — 定期检查点
- `logs/` — TensorBoard 事件文件
- `runs/train_*/` — 训练配置快照

### 评估模式

```bash
python main.py --mode evaluate --model checkpoints/best_model.pth
```

**输出**（均在 `outputs/evaluation/`）：
- `eval_results.json` — 全部指标的 JSON
- `eval_report.md` — Markdown 评估报告
- `confusion_matrix.png` — 混淆矩阵热力图
- `class_accuracy.png` — 各类别准确率柱状图
- `misclassified.csv` — 误分类样本列表

### 推理模式

```bash
# 单张图像
python main.py --mode predict --input test.jpg --model checkpoints/best_model.pth
# 输出: 类别: Forest, 置信度: 0.9521

# 批量（文件夹）
python main.py --mode predict --input "D:/TestImages/" --model checkpoints/best_model.pth

# 返回 Top-5 预测
python main.py --mode predict --input test.jpg --top-k 5
```

**输出**（均在 `outputs/predictions/`）：
- `predictions.csv` — 预测结果汇总
- `skipped_files.txt` — 跳过的损坏文件

### 环境检测模式

```bash
python main.py --mode check
```

检测 6 项：Python 版本、PyTorch+CUDA+GPU、TorchVision、数据路径、磁盘空间、依赖包版本。

## 三、配置文件

所有参数通过 `configs/config.yaml` 管理，CLI 参数可覆盖。详见 `CONFIG_GUIDE.md`。

## 四、典型使用场景

### 场景 1：快速体验（有预训练模型）

```bash
python main.py --mode predict --input "D:/DataDownload/EuroSat_Dataset/EuroSAT/Forest/Forest_1.jpg"
```

### 场景 2：在自己的 EuroSAT 数据上训练

```bash
# 1. 修改 configs/config.yaml 中的 data.root_dir
# 2. 环境检测
python main.py --mode check
# 3. 训练
python main.py --mode train
# 4. 评估
python main.py --mode evaluate
```

### 场景 3：精度不足，微调提升

```bash
# 1. 修改 configs/config.yaml: model.unfreeze_layers: 1
# 2. 微调 20 epoch
python main.py --mode train --epochs 70 --lr 0.00001 --resume checkpoints/best_model.pth
# 3. 重新评估
python main.py --mode evaluate
```
