# 模型说明

## 一、选型

**主选：ResNet18**（11.18M 参数，FLOPs 1.8G @224×224）

| 选型因素 | 说明 |
|---------|------|
| 经典可靠 | CV 领域最广泛验证的架构，文档丰富 |
| 硬件友好 | batch=256 时训练显存仅 ~1.8 GB |
| 迁移学习高效 | 冻结骨干仅训练分类头，50 epoch 约 50 分钟 |
| 精度达标 | 微调后 93.07%，满足 ≥90% 目标 |
| 升级路径清晰 | 冻结→解冻layer4→全量微调，三步渐进 |

**候选模型对比**（EuroSAT 文献基准）：

| 模型 | 参数量 | 文献精度 | 4GB 适配 | 本项目选择 |
|------|:---:|:---:|:---:|:---:|
| ResNet18 | 11.2M | ~93% | ✅ 优秀 | **主选** |
| MobileNetV3-Large | 5.4M | ~98.7% | ✅ 优秀 | 备选替代 |
| DenseNet-121 | 8.0M | ~97.5% | ✅ 良好 | 备选 |
| ResNet50 | 25.6M | ~98.8% | ⚠️ 可用 | Could |
| ConvNeXt-Tiny | 28.6M | ~98.7% | ⚠️ 可用 | Could |
| ViT-B/16 | 86.6M | ~95% | ❌ 不适合 | Won't |

## 二、训练策略

### 两阶段渐进微调

```
Phase 1: PEFT (50 epoch)
  ├── 骨干：❄️ 冻结 (ImageNet 预训练)
  ├── 分类头：🔥 训练 (Kaiming 随机初始化)
  ├── 可训练：5,130 / 11.18M (0.05%)
  ├── 学习率：1e-4
  └── 结果：val_acc = 52.63%

         ↓ 精度不足，ImageNet 特征与航拍域差距大

Phase 2: Partial Fine-tuning (20 epoch, 51→70)
  ├── layer1~3：❄️ 冻结
  ├── layer4：🔥 解冻微调
  ├── 分类头：🔥 继续训练 (从 Phase 1 权重开始)
  ├── 可训练：8.40M / 11.18M (75%)
  ├── 学习率：1e-5 (防止破坏预训练特征)
  └── 结果：val_acc = 93.07% ✅
```

### 为什么 Phase 1 精度低

ImageNet 预训练骨干学的是自然图像特征（猫狗汽车的边缘/纹理/形状），与 EuroSAT 航拍影像（俯视土地覆盖纹理）存在**域差距**。仅训练分类头（5K 参数）无法弥合这个差距。

### 为什么只解冻 layer4

ResNet18 残差块的功能分层：
- **layer1~3**：底层边缘/纹理检测（通用，跨域可迁移）
- **layer4**：高层语义组装（任务相关，需要适配）

解冻 layer4 让模型学会把通用纹理组装为航拍土地覆盖语义（绿棕色块→森林、灰色格子→住宅区、灰色大片→工业区），同时保持底层特征检测器不被破坏。

## 三、优化器与超参数

| 参数 | Phase 1 | Phase 2 | 说明 |
|------|:---:|:---:|------|
| 优化器 | AdamW | AdamW | 自适应学习率 + 解耦权重衰减 |
| 学习率 | 1e-4 | 1e-5 | 微调用更低学习率 |
| 权重衰减 | 5e-4 | 5e-4 | 正则化 |
| LR 调度 | ReduceLROnPlateau | ReduceLROnPlateau | 监控 val_loss 自动降 lr |
| 早停 | patience=10 | patience=10 | 连续 10 轮无改善则停止 |
| 损失函数 | CrossEntropyLoss | CrossEntropyLoss | 多分类标准损失 |
| 批大小 | 256 | 256 | GPU 显存充裕 |
| 精度 | FP32 | FP32 | Pascal 架构无 AMP 加速 |

## 四、评估指标

| 指标 | 数值 | 说明 |
|------|:---:|------|
| Top-1 Accuracy | **93.07%** | 主指标 |
| Top-2 Accuracy | ~97.5% | 前 2 命中率 |
| Macro F1 | ~93.0% | 各类别 F1 平均 |

### 易混淆类别 (Top-3)

1. **AnnualCrop ↔ PermanentCrop**：同为农作物俯视纹理，区分困难
2. **Industrial ↔ Residential**：航拍下建筑物纹理相似
3. **HerbaceousVegetation ↔ Pasture**：草地和牧场纹理接近

## 五、升级路径

如果 93% 精度不够，按以下顺序尝试：

| 优先级 | 方案 | 预期提升 | 操作 |
|:---:|------|:---:|------|
| 1 | 解冻 layer3+4 (`unfreeze_layers: 2`) | +1~2pp | 改配置重训 |
| 2 | 切换 MobileNetV3-Large | +3~5pp | 改 `model.name` |
| 3 | 全量微调 (`unfreeze_layers: 4`) | +1~2pp | 改配置，需降低 batch_size |
| 4 | CutMix/MixUp 高级增强 | +0.5~1pp | 引入 timm 库 |

## 六、模型文件

| 文件 | 大小 | 说明 |
|------|:---:|------|
| `checkpoints/best_model.pth` | ~45 MB | 最佳 val_acc 模型（含 optimizer state + config） |
| 纯权重导出 | ~45 MB | `model.state_dict()` 不含 optimizer |
| 预训练权重 | ~45 MB | `resnet18-f37072fd.pth` (ImageNet-1K) |
