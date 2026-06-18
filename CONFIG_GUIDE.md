# 配置指南

`configs/config.yaml` 是唯一配置入口。所有参数有默认值，**只需修改 `data.root_dir` 即可开始训练**。命令行参数优先级高于配置文件。

---

## 一、顶层

```yaml
experiment_name: eurosat_classification   # 实验名称（影响输出目录命名）
seed: 42                                  # 全局随机种子（确保可复现）
```

---

## 二、system — 系统配置

```yaml
system:
  gpu_id: 0                               # GPU 编号，-1 = 纯 CPU（速度慢 5-10 倍）
  gpu_memory_fraction: 0.75               # 进程显存占比上限，(0, 1]，设为 1.0 则不限制
  num_workers: 4                          # DataLoader 子进程数。内存不足时降为 2
  log_level: INFO                         # DEBUG / INFO / WARNING / ERROR
  output_dir: "./runs"                    # 训练配置快照输出目录
```

---

## 三、data — 数据配置

```yaml
data:
  root_dir: "D:/DataDownload/EuroSat_Dataset/EuroSAT"   # ⚠️ 必填，EuroSAT 根目录
  num_classes: 10                         # 固定值
  class_names:                            # 类别名，按字母序映射为标签 0~9
    - AnnualCrop       # 年度作物
    - Forest           # 森林
    - HerbaceousVegetation  # 草本植被
    - Highway          # 高速公路
    - Industrial       # 工业区
    - Pasture          # 牧场
    - PermanentCrop    # 永久作物
    - Residential      # 住宅区
    - River            # 河流
    - SeaLake          # 海湖
  train_ratio: 0.7                        # 训练集比例
  val_ratio: 0.1                          # 验证集比例（测试集 = 1 - 0.7 - 0.1 = 0.2）
  input_size: 224                         # ResNet18 输入尺寸（固定，勿改）
  original_size: 64                       # EuroSAT 原始尺寸（仅日志输出用）
```

> 数据集目录结构：`root_dir/` 下应有 10 个类别子文件夹，每个含 .jpg/.png 图像。

---

## 四、augmentation — 数据增强

```yaml
augmentation:
  horizontal_flip: 0.5                    # 随机水平翻转概率（0 = 关闭）
  rotation_degrees: 15.0                  # 随机旋转最大角度（0 = 关闭）
  brightness: 0.2                         # 亮度抖动幅度 ±20%（0 = 关闭）
  contrast: 0.2                           # 对比度抖动幅度 ±20%（0 = 关闭）
```

> 增强仅在训练集生效，验证/测试集无随机变换，确保评估结果可比。

---

## 五、model — 模型配置

```yaml
model:
  name: "resnet18"                        # resnet18 / mobilenet_v3_large
  pretrained: true                        # 是否加载预训练权重
  pretrained_weights: "IMAGENET1K_V1"    # 在线权重版本
  pretrained_path: "D:/DataDownload/Models/resnet18-f37072fd.pth"  # ⚠️ 本地权重路径
  freeze_backbone: true                   # true = 仅训练分类头；false = 全量训练
  unfreeze_layers: 1                      # 解冻最后 N 个残差块（0=PEFT, 1=layer4, 4=全量）
```

**`pretrained_path` 加载优先级**：
1. 指向的文件存在 → 从本地 `.pth` 加载
2. 本地不存在 → 回退在线下载 `pretrained_weights`
3. 在线下载失败 → Kaiming 随机初始化 + 警告

**`unfreeze_layers` 说明**：

| 值 | 含义 | 可训练参数 | 适用场景 |
|:---:|------|:---:|------|
| 0 | PEFT，仅训练分类头 | ~5K (0.05%) | 快速收敛，但精度上限低 |
| 1 | 解冻 layer4 | ~8.4M (75%) | **推荐**，精度最高 |
| 2 | 解冻 layer3+4 | ~10M | 更多调参空间 |
| 4 | 全量微调 | ~11.2M (100%) | 最慢，可能过拟合 |

---

## 六、train — 训练配置

```yaml
train:
  batch_size: 256                         # 批大小。显存不足降为 128/64
  epochs: 50                              # 总训练轮数。快速验证用 30，微调用 70
  learning_rate: 0.0001                   # 1e-4（PEFT）；微调时降为 1e-5
  optimizer: "adamw"                      # adamw / adam / sgd
  weight_decay: 0.0005                    # 权重衰减（5e-4）
  momentum: 0.9                           # SGD 动量（仅 optimizer=sgd 时生效）
  loss: "cross_entropy"                   # 损失函数
  lr_scheduler: "plateau"                 # plateau（监控val_loss） / cosine / none
  lr_patience: 5                          # ReduceLROnPlateau 等待轮数
  lr_factor: 0.5                          # 衰减因子（lr × 0.5）
  early_stop_patience: 10                 # 早停耐心值（连续N轮无改善则停止，0=关闭）
  early_stop_min_delta: 0.001             # 早停最小改善阈值
  checkpoint_dir: "checkpoints"           # 检查点保存目录
  log_dir: "logs"                         # TensorBoard 日志目录
  save_best_only: false                   # true = 仅保存 best_model；false = 外加定期检查点
  save_interval: 10                       # 每隔 N epoch 保存定期检查点
  resume: ""                              # 断点续训路径（空 = 从头训练）
```

---

## 七、inference — 推理配置

```yaml
inference:
  batch_size: 256                         # 推理批大小
  top_k: 2                                # 返回 Top-K 预测
  conf_warning_threshold: 0.3             # 置信度低于此值告警
  output_dir: "./outputs/predictions"     # 预测结果输出目录
  save_format: "csv"                      # csv / json
```

---

## 八、visualization — 可视化配置

```yaml
visualization:
  dpi: 200                                # 输出图像分辨率
  figsize: [10, 8]                        # 图像尺寸（英寸，[宽, 高]）
  cmap: "Blues"                           # 混淆矩阵配色方案
  show_values: true                       # 混淆矩阵格子中显示数值
  plot_format: "png"                      # png / pdf / svg
```

---

## 九、三种典型配置场景

### 场景 1：快速验证（~30 分钟，精度 ~50%）

```yaml
train:
  epochs: 30
model:
  freeze_backbone: true
  unfreeze_layers: 0
```

### 场景 2：精度优先（推荐，~75 分钟，精度 ~93%）

```yaml
# 第一步：PEFT 50 epoch（训练分类头）
train:
  epochs: 50
  learning_rate: 0.0001
model:
  freeze_backbone: true
  unfreeze_layers: 0

# 第二步：修改 config.yaml，解冻 layer4 微调 20 epoch
train:
  epochs: 70           # 总目标 70，实际跑 51→70
  learning_rate: 0.00001
  resume: "checkpoints/best_model.pth"
model:
  unfreeze_layers: 1
```

### 场景 3：显存受限（≤ 2GB）

```yaml
train:
  batch_size: 64
system:
  num_workers: 2
```
