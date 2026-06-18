# 基于深度学习的卫星影像分类系统 — 详细设计说明书

> **文档编号**：DDS-EuroSAT-001  
> **版本**：v1.0.0  
> **编制日期**：2026-06-16  
> **编制依据**：GB/T 8567-2006、《卫星影像分类需求规格说明书V1.0》、《卫星影像分类技术选型方案V1.0》、《EuroSAT卫星影像分类系统_概要设计说明书V1.0》  
> **项目版本**：EuroSAT_Classification v1.0.0

---

## 修改记录

| 版本 | 日期 | 修改内容 | 修改人 |
|------|------|---------|--------|
| v1.0.0 | 2026-06-16 | 初稿：基于ADS V1.0完成8模块详细实现设计，含数据结构定义、算法伪代码、异常处理逻辑 | — |

---

## 目录

- [1. 引言](#1-引言)
- [2. config.py — 配置管理模块](#2-configpy--配置管理模块)
- [3. data.py — 数据加载与增强模块](#3-datapy--数据加载与增强模块)
- [4. model.py — 模型构建与迁移学习模块](#4-modelpy--模型构建与迁移学习模块)
- [5. train.py — 模型训练模块](#5-trainpy--模型训练模块)
- [6. evaluate.py — 模型评估模块](#6-evaluatepy--模型评估模块)
- [7. predict.py — 推理与预测模块](#7-predictpy--推理与预测模块)
- [8. utils.py — 工具函数模块](#8-utilspy--工具函数模块)
- [9. main.py — CLI 统一入口模块](#9-mainpy--cli-统一入口模块)
- [附录A 异常码定义](#附录a-异常码定义)
- [附录B 需求追溯矩阵](#附录b-需求追溯矩阵)

---

## 1. 引言

### 1.1 编写目的

本文档是 EuroSAT_Classification 卫星影像分类系统的详细设计说明书（Detailed Design Specification, DDS），依据 GB/T 8567-2006 编制。用于：

1. 描述每个模块的详细实现逻辑、数据结构和算法
2. 作为编码实现的直接依据（可直接翻译为 Python 代码）
3. 为单元测试用例编写提供输入

### 1.2 适用范围

- **开发人员**：按本文档进行编码实现
- **代码审查者**：对照本文档审查实现正确性
- **测试人员**：理解函数内部逻辑，设计白盒测试用例

### 1.3 参考文档

| 文档 | 路径 | 说明 |
|------|------|------|
| 需求规格说明书 V1.0 | `卫星影像分类需求规格说明书V1.0.md` | 功能与非功能需求 |
| 技术选型方案 V1.0 | `卫星影像分类技术选型方案V1.0.md` | 技术栈与模型决策 |
| 概要设计说明书 V1.0 | `EuroSAT卫星影像分类系统_概要设计说明书V1.0.md` | 架构、模块划分、接口契约 |

### 1.4 文档约定

- **Python 版本**：3.8.20
- **类型注解**：使用 `typing` 模块（`List`, `Dict`, `Tuple`, `Optional`, `Union`）
- **命名规范**：函数 `snake_case`、类 `PascalCase`、常量 `UPPER_SNAKE_CASE`
- **数据路径**：`D:\DataDownload\EuroSat_Dataset\EuroSAT`
- **代码文件均位于项目根目录**

---

## 2. config.py — 配置管理模块

### 2.1 模块概述

| 属性 | 说明 |
|------|------|
| 行数估算 | ~200 行 |
| 依赖 | `pyyaml`, `dataclasses`, `pathlib`, `argparse`, `logging` |
| 核心职责 | YAML 配置加载/校验/后处理 → 不可变 Config 对象；配置快照保存；CLI 参数合并 |
| 对应 ADS | §4.6, 接口 I-10 |

### 2.2 数据结构定义

```python
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

@dataclass(frozen=True)
class SystemConfig:
    """系统配置（不可变）"""
    gpu_id: int = 0                          # GPU 编号，-1 表示 CPU
    gpu_memory_fraction: float = 0.75        # 显存使用比例上限 (0, 1]
    num_workers: int = 4                     # DataLoader 子进程数
    log_level: str = "INFO"                  # DEBUG / INFO / WARNING / ERROR
    output_dir: str = "./runs"               # 输出根目录

@dataclass(frozen=True)
class DataConfig:
    """数据配置（不可变）"""
    root_dir: str = ""                       # EuroSAT 根目录（必填，含10个类别子文件夹）
    num_classes: int = 10                    # 类别数
    class_names: Tuple[str, ...] = (         # 类别名（按字母序映射 0~9）
        "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
        "Industrial", "Pasture", "PermanentCrop", "Residential",
        "River", "SeaLake"
    )
    train_ratio: float = 0.7                # 训练集比例
    val_ratio: float = 0.1                  # 验证集比例
    input_size: int = 224                    # ResNet18 输入尺寸
    original_size: int = 64                  # EuroSAT 原始尺寸

@dataclass(frozen=True)
class AugmentationConfig:
    """数据增强配置（不可变）"""
    horizontal_flip: float = 0.5             # 水平翻转概率
    rotation_degrees: float = 15.0           # 随机旋转最大角度
    brightness: float = 0.2                  # 亮度调整幅度
    contrast: float = 0.2                    # 对比度调整幅度

@dataclass(frozen=True)
class ModelConfig:
    """模型配置（不可变）"""
    name: str = "resnet18"                   # 模型名称
    pretrained: bool = True                  # 是否加载预训练权重
    pretrained_weights: str = "IMAGENET1K_V1"  # 预训练权重版本
    freeze_backbone: bool = True             # 是否冻结骨干
    unfreeze_layers: int = 0                 # 解冻最后 N 个残差块

@dataclass(frozen=True)
class TrainConfig:
    """训练配置（不可变）"""
    batch_size: int = 256
    epochs: int = 50
    learning_rate: float = 0.0001            # 1e-4
    optimizer: str = "adamw"                 # adamw / adam / sgd
    weight_decay: float = 0.0005             # 5e-4
    momentum: float = 0.9                    # SGD 时生效
    loss: str = "cross_entropy"
    lr_scheduler: str = "plateau"            # plateau / cosine / none
    lr_patience: int = 5
    lr_factor: float = 0.5
    early_stop_patience: int = 10
    early_stop_min_delta: float = 0.001
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    save_best_only: bool = False             # True=仅保存最佳，False=也保存定期检查点
    save_interval: int = 10                  # 每 N epoch 保存定期检查点
    resume: str = ""                         # 恢复训练的检查点路径，空字符串=从头训练

@dataclass(frozen=True)
class InferenceConfig:
    """推理配置（不可变）"""
    batch_size: int = 256
    top_k: int = 2
    conf_warning_threshold: float = 0.3
    output_dir: str = "./outputs/predictions"
    save_format: str = "csv"                 # csv / json

@dataclass(frozen=True)
class VisualizationConfig:
    """可视化配置（不可变）"""
    dpi: int = 200
    figsize: Tuple[int, int] = (10, 8)
    cmap: str = "Blues"
    show_values: bool = True
    plot_format: str = "png"

@dataclass(frozen=True)
class Config:
    """顶层配置聚合（不可变）"""
    experiment_name: str = "eurosat_classification"
    seed: int = 42
    system: SystemConfig = field(default_factory=SystemConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
```

### 2.3 核心函数详细设计

#### 2.3.1 load_config()

```
函数: load_config(path: str, cli_args: dict = None) -> Config

1. 入参校验:
   - path 不存在 → raise FileNotFoundError(f"配置文件不存在: {path}")
   
2. 加载 YAML:
   with open(path, 'r', encoding='utf-8') as f:
       raw_dict = yaml.safe_load(f)
   - yaml.YAMLError → raise ValueError(f"YAML 解析失败: {e}")

3. CLI 参数覆盖:
   if cli_args:
       for key, value in cli_args.items():
           if value is not None:
               _set_nested_key(raw_dict, key, value)
   # _set_nested_key 支持点号分隔的嵌套键: "train.batch_size" → raw_dict["train"]["batch_size"]

4. 递归构造 dataclass:
   config = _build_config(Config, raw_dict)
   # 对每个 dataclass 字段: 如果 raw_dict 有对应键 → 使用配置值；否则使用默认值
   # 多余字段（raw_dict 有但 dataclass 无）→ logging.warning(f"未知配置字段: {key}")

5. 后处理与校验:
   _validate(config)
   
6. 返回 Config 对象
```

#### 2.3.2 _validate()

```
内部函数: _validate(config: Config) -> None

校验规则:
1. system.gpu_memory_fraction:
   - 必须在 (0, 1] 范围内，否则 ValueError
   
2. data.root_dir:
   - 不能为空字符串，否则 ValueError("data.root_dir 未设置，请在配置文件中指定数据集路径")
   - 路径存在性检查推迟到 data.py 模块（解耦：config 不依赖文件系统状态）
   
3. data.train_ratio + data.val_ratio:
   - 两者之和 < 1.0，否则 ValueError("train_ratio + val_ratio 必须 < 1.0")
   
4. data.num_classes:
   - 必须 == len(data.class_names)，否则 ValueError
   
5. train.batch_size:
   - 必须 > 0，否则 ValueError
   
6. train.epochs:
   - 必须 > 0，否则 ValueError
   
7. train.learning_rate:
   - 必须 > 0，否则 ValueError
   
8. model.unfreeze_layers:
   - 必须在 [0, 4] 范围内，否则 ValueError
   - if unfreeze_layers > 0 and freeze_backbone == False:
       logging.warning("freeze_backbone=False 时 unfreeze_layers 被忽略")
       
9. 危险组合检测:
   - if model.unfreeze_layers >= 2 and train.batch_size >= 256:
       logging.warning("解冻多层 + 大批量可能导致显存超限，建议降低 batch_size")
```

#### 2.3.3 save_config_snapshot()

```
函数: save_config_snapshot(cfg: Config, output_dir: str) -> str

1. 生成文件名:
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   filename = f"config_{timestamp}.yaml"
   filepath = os.path.join(output_dir, filename)

2. 序列化 Config → dict:
   raw = dataclasses.asdict(cfg)
   # dataclasses.asdict 递归转换嵌套 dataclass 为嵌套 dict

3. 写入 YAML:
   os.makedirs(output_dir, exist_ok=True)
   with open(filepath, 'w', encoding='utf-8') as f:
       yaml.dump(raw, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

4. 返回 filepath
```

---

## 3. data.py — 数据加载与增强模块

### 3.1 模块概述

| 属性 | 说明 |
|------|------|
| 行数估算 | ~250 行 |
| 依赖 | `torch`, `torchvision`, `Pillow`, `numpy`, `sklearn.model_selection`, `config` |
| 核心职责 | 数据集加载、7:1:2 分层划分、增强管线构建、DataLoader 封装 |
| 对应 ADS | §4.1, 接口 I-01, I-02 |
| 对应 SRS | FR-1, FR-2 |

### 3.2 常量定义

```python
# ImageNet 标准化参数（与预训练权重一致）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# 支持的图像格式
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png')
```

### 3.3 _build_train_transform()

```
函数: _build_train_transform(config: Config) -> transforms.Compose

构建训练增强流水线:

transforms.Compose([
    transforms.Resize((config.data.input_size, config.data.input_size),
                      interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(p=config.augmentation.horizontal_flip),
    transforms.RandomRotation(degrees=config.augmentation.rotation_degrees,
                              fill=0),  # 旋转空白区用黑色填充
    transforms.ColorJitter(
        brightness=config.augmentation.brightness,
        contrast=config.augmentation.contrast
    ),
    transforms.ToTensor(),  # PIL Image [0,255] → Tensor [0,1]
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])
```

### 3.4 _build_eval_transform()

```
函数: _build_eval_transform(config: Config) -> transforms.Compose

构建评估/推理流水线（无随机增强）:

transforms.Compose([
    transforms.Resize((config.data.input_size, config.data.input_size),
                      interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])
```

### 3.5 create_datasets()

```
函数: create_datasets(config: Config) -> tuple[Dataset, Dataset, Dataset]

算法流程:

1. 路径校验:
   root = Path(config.data.root_dir)
   if not root.exists():
       raise FileNotFoundError(f"数据集根目录不存在: {root}")
   
2. 类别文件夹校验:
   期望: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
          'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
   actual = sorted([d.name for d in root.iterdir() if d.is_dir()])
   if actual != 期望:
       raise ValueError(f"类别文件夹不匹配\n期望: {期望}\n实际: {actual}")

3. 检查每个类别文件夹非空:
   for class_dir in actual:
       images = list((root / class_dir).glob('*.jpg')) + list((root / class_dir).glob('*.png'))
       if len(images) == 0:
           raise ValueError(f"类别文件夹为空: {class_dir}")

4. 使用 ImageFolder 加载全部数据:
   full_dataset = torchvision.datasets.ImageFolder(
       root=str(root),
       transform=None  # 先不加 transform，划分后再分别设置
   )
   # ImageFolder 按字母序映射标签，确保 class_to_idx 与 class_names 顺序一致
   assert full_dataset.classes == list(config.data.class_names), \
          f"类别映射不一致: {full_dataset.classes} vs {config.data.class_names}"

5. 计算划分索引:
   total = len(full_dataset)  # 27,000
   labels = full_dataset.targets  # List[int], 长度 27,000
   
   # 第一次划分: train (70%) vs temp (30%)
   train_indices, temp_indices = train_test_split(
       range(total),
       train_size=config.data.train_ratio,
       stratify=labels,
       random_state=config.seed
   )
   
   # 第二次划分: val (10%) vs test (20%) — temp 中各取 1/3 和 2/3
   temp_labels = [labels[i] for i in temp_indices]
   val_ratio_in_temp = config.data.val_ratio / (1 - config.data.train_ratio)  # 0.1 / 0.3 ≈ 0.333
   val_indices, test_indices = train_test_split(
       temp_indices,
       train_size=val_ratio_in_temp,
       stratify=temp_labels,
       random_state=config.seed
   )

6. 创建子集 Dataset:
   train_dataset = SubsetWithTransform(full_dataset, train_indices,
                                        _build_train_transform(config))
   val_dataset   = SubsetWithTransform(full_dataset, val_indices,
                                        _build_eval_transform(config))
   test_dataset  = SubsetWithTransform(full_dataset, test_indices,
                                        _build_eval_transform(config))
   
   # SubsetWithTransform 是一个内部类，继承 torch.utils.data.Subset，
   # 重写 __getitem__ 以应用指定的 transform

7. 打印分布统计:
   logging.info(f"数据集划分完成:")
   logging.info(f"  训练集: {len(train_indices)} 张 ({config.data.train_ratio:.0%})")
   logging.info(f"  验证集: {len(val_indices)} 张 ({config.data.val_ratio:.0%})")
   logging.info(f"  测试集: {len(test_indices)} 张 ({1-config.data.train_ratio-config.data.val_ratio:.0%})")
   
   # 各类别分布
   for class_id, class_name in enumerate(config.data.class_names):
       train_count = sum(1 for i in train_indices if labels[i] == class_id)
       val_count   = sum(1 for i in val_indices   if labels[i] == class_id)
       test_count  = sum(1 for i in test_indices  if labels[i] == class_id)
       logging.info(f"  {class_name}: train={train_count}, val={val_count}, test={test_count}")

8. 返回:
   return (train_dataset, val_dataset, test_dataset)


内部类: SubsetWithTransform

class SubsetWithTransform(torch.utils.data.Subset):
    """带 transform 的数据子集"""
    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices)
        self.transform = transform
    
    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        # image: PIL Image (原始尺寸, 64×64)
        # label: int (0~9)
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.indices)
```

### 3.6 get_dataloaders()

```
函数: get_dataloaders(config: Config, datasets: tuple) -> tuple[DataLoader, DataLoader, DataLoader]

输入: (train_dataset, val_dataset, test_dataset)
输出: (train_loader, val_loader, test_loader)

参数:
- train_loader: batch_size=config.train.batch_size, shuffle=True,
                num_workers=min(config.system.num_workers, cpu_count//2),
                pin_memory=True, drop_last=True
                # drop_last=True: 丢弃最后不足一个 batch 的数据，避免 batch 大小不一致
- val_loader:   batch_size=config.train.batch_size, shuffle=False,
                num_workers=min(config.system.num_workers, cpu_count//2),
                pin_memory=True, drop_last=False
- test_loader:  同 val_loader

num_workers 保护:
  cpu_count = os.cpu_count() or 1
  actual_workers = min(config.system.num_workers, max(1, cpu_count // 2))
  if actual_workers != config.system.num_workers:
      logging.warning(f"num_workers 从 {config.system.num_workers} 降为 {actual_workers} (CPU核心限制)")
```

### 3.7 损坏文件处理

`ImageFolder` 在 `__getitem__` 时自动加载图像。若图像损坏，Pillow 会抛出 `PIL.UnidentifiedImageError`。处理策略：

- **训练阶段**：DataLoader 的 `collate_fn` 无法处理异常（异常在 worker 进程中发生）
- **解决方案**：自定义 `ImageFolder` 子类，重写 `__getitem__`

```python
class RobustImageFolder(torchvision.datasets.ImageFolder):
    """带损坏文件检测的 ImageFolder"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skipped_files = []
    
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except (PIL.UnidentifiedImageError, OSError) as e:
            path = self.imgs[index][0]  # (path, label) 元组
            logging.warning(f"跳过损坏文件: {path} ({e})")
            self.skipped_files.append(path)
            # 返回下一个有效样本（递归查找）
            next_index = (index + 1) % len(self)
            return self.__getitem__(next_index)
```

---

## 4. model.py — 模型构建与迁移学习模块

### 4.1 模块概述

| 属性 | 说明 |
|------|------|
| 行数估算 | ~150 行 |
| 依赖 | `torch`, `torchvision`, `config` |
| 核心职责 | ResNet18 构建、ImageNet 预训练权重加载、分类头替换、骨干冻结/解冻、检查点存取 |
| 对应 ADS | §4.2, 接口 I-03, I-04, I-05 |
| 对应 SRS | FR-3 |

### 4.2 build_model()

```
函数: build_model(config: Config) -> nn.Module

算法流程:

1. 根据模型名称选择构建函数:
   model_name = config.model.name.lower()
   if model_name == "resnet18":
       model_fn = torchvision.models.resnet18
   elif model_name == "mobilenet_v3_large":
       model_fn = torchvision.models.mobilenet_v3_large
   else:
       raise ValueError(f"不支持的模型: {model_name}，合法值: ['resnet18', 'mobilenet_v3_large']")

2. 加载预训练权重:
   if config.model.pretrained:
       try:
           model = model_fn(weights=config.model.pretrained_weights)
           logging.info(f"已加载预训练权重: {config.model.pretrained_weights}")
       except Exception as e:
           logging.warning(f"预训练权重加载失败 ({e})，降级为随机初始化")
           model = model_fn(weights=None)
   else:
       model = model_fn(weights=None)
       logging.info("使用随机初始化权重")

3. 替换分类头:
   if model_name == "resnet18":
       in_features = model.fc.in_features  # 512
       model.fc = nn.Linear(in_features, config.data.num_classes)  # nn.Linear(512, 10)
       # Kaiming 初始化新 FC 层
       nn.init.kaiming_normal_(model.fc.weight, mode='fan_out', nonlinearity='relu')
       nn.init.zeros_(model.fc.bias)
   elif model_name == "mobilenet_v3_large":
       in_features = model.classifier[-1].in_features  # 960
       model.classifier[-1] = nn.Linear(in_features, config.data.num_classes)
       nn.init.kaiming_normal_(model.classifier[-1].weight, mode='fan_out', nonlinearity='relu')
       nn.init.zeros_(model.classifier[-1].bias)

4. 打印参数统计:
   total_params = sum(p.numel() for p in model.parameters())
   logging.info(f"模型总参数量: {total_params:,}")

5. 冻结/解冻控制:
   if config.model.freeze_backbone:
       freeze_backbone(model, model_name)
       unfreeze_layers_n = config.model.unfreeze_layers
       if unfreeze_layers_n > 0:
           _unfreeze_last_n_blocks(model, model_name, unfreeze_layers_n)

6. 设备迁移:
   device = torch.device(f"cuda:{config.system.gpu_id}" 
                         if torch.cuda.is_available() and config.system.gpu_id >= 0
                         else "cpu")
   model = model.to(device)
   logging.info(f"模型已移至: {device}")

7. 返回 model
```

### 4.3 freeze_backbone()

```
函数: freeze_backbone(model: nn.Module, model_name: str) -> None

算法:

1. 设置全部参数 requires_grad = True（重置状态）

2. 根据模型名称确定分类头参数名:
   - resnet18: "fc" 
   - mobilenet_v3_large: "classifier"

3. 冻结除分类头外的所有参数:
   head_param_name = "fc" if model_name == "resnet18" else "classifier"
   for name, param in model.named_parameters():
       if not name.startswith(head_param_name):
           param.requires_grad = False

4. 统计可训练参数:
   trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
   total = sum(p.numel() for p in model.parameters())
   logging.info(f"可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
   # ResNet18: 5,130 / 11,689,610 (0.04%)
```

### 4.4 _unfreeze_last_n_blocks()

```
函数: _unfreeze_last_n_blocks(model: nn.Module, model_name: str, n: int) -> None

算法 (以 ResNet18 为例):

ResNet18 残差块命名规则:
  layer1 → layer2 → layer3 → layer4

n=1: 解冻 layer4
n=2: 解冻 layer3 + layer4
n=3: 解冻 layer2 + layer3 + layer4
n=4: 解冻 layer1 + layer2 + layer3 + layer4 (≈解冻全部骨干)

实现:
  blocks = ["layer1", "layer2", "layer3", "layer4"]
  blocks_to_unfreeze = blocks[-n:]  # 取最后 N 个
  
  for name, param in model.named_parameters():
      for block_name in blocks_to_unfreeze:
          if name.startswith(block_name):
              param.requires_grad = True
              break
  
  trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
  logging.info(f"解冻最后 {n} 个残差块后，可训练参数: {trainable_after:,}")
```

### 4.5 get_loss_fn()

```
函数: get_loss_fn() -> nn.Module

返回: nn.CrossEntropyLoss()
# 简单封装，保持接口统一。若未来需要切换损失函数（如 LabelSmoothing），仅修改此函数即可。
```

### 4.6 save_checkpoint()

```
函数: save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,     # 含 "val_acc": float
    path: str,
    config: Config
) -> None

算法:

1. 构建检查点字典:
   checkpoint = {
       'epoch': epoch,
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'best_val_acc': metrics.get('best_val_acc', 0.0),
       'config': dataclasses.asdict(config),  # 配置快照
   }

2. 原子写入:
   tmp_path = path + '.tmp'
   torch.save(checkpoint, tmp_path)
   os.replace(tmp_path, path)  # Windows 同磁盘原子操作
   # os.replace 在 Windows 上是原子的（同磁盘），防止写入中断导致原文件损坏

3. 日志:
   logging.info(f"检查点已保存: {path} (epoch={epoch}, val_acc={metrics.get('val_acc', 'N/A')})")
```

### 4.7 load_checkpoint()

```
函数: load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> tuple[int, dict]

算法:

1. 校验:
   if not os.path.exists(path):
       raise FileNotFoundError(f"检查点不存在: {path}")

2. 加载:
   checkpoint = torch.load(path, map_location='cpu')  # 先加载到 CPU 避免设备不匹配
   
3. 恢复模型状态:
   model.load_state_dict(checkpoint['model_state_dict'])
   
4. 恢复优化器状态（如有）:
   if optimizer is not None:
       if 'optimizer_state_dict' in checkpoint:
           optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
       else:
           logging.warning("检查点中无 optimizer_state_dict，优化器使用初始状态")

5. 提取元信息:
   epoch = checkpoint.get('epoch', 0)
   best_val_acc = checkpoint.get('best_val_acc', 0.0)
   metrics = {'best_val_acc': best_val_acc}

6. 返回:
   return (epoch, metrics)
```

---

## 5. train.py — 模型训练模块

### 5.1 模块概述

| 属性 | 说明 |
|------|------|
| 行数估算 | ~350 行 |
| 依赖 | `torch`, `torch.nn`, `torch.optim`, `tensorboard`, `tqdm`, `config`, `data`, `model`, `utils` |
| 核心职责 | 训练循环编排、验证评估、早停判断、学习率调度、OOM 恢复、检查点管理 |
| 对应 ADS | §4.3, 接口 I-06 |
| 对应 SRS | FR-4 |

### 5.2 Trainer 类设计

```python
class Trainer:
    """训练器：封装训练状态与逻辑"""
    
    def __init__(
        self,
        config: Config,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,          # ReduceLROnPlateau 或 CosineAnnealingLR 实例
        criterion: nn.Module,  # CrossEntropyLoss
        writer,             # TensorBoard SummaryWriter
        device: torch.device
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.writer = writer
        self.device = device
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.train_history = {
            'train_loss': [], 'val_loss': [],
            'val_acc': [], 'lr': [], 'peak_mem_mb': []
        }
```

### 5.3 _train_one_epoch()

```
方法: Trainer._train_one_epoch() -> float

返回: 该 epoch 的平均训练 loss

算法:

self.model.train()
running_loss = 0.0
num_batches = len(self.train_loader)
progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

for batch_idx, (images, labels) in enumerate(progress_bar):
    images = images.to(self.device, non_blocking=True)    # (B, 3, 224, 224)
    labels = labels.to(self.device, non_blocking=True)    # (B,)
    
    # ----- 前向传播 -----
    outputs = self.model(images)   # (B, 10) logits
    loss = self.criterion(outputs, labels)
    
    # ----- 反向传播 -----
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    
    # ----- 记录 -----
    running_loss += loss.item()
    
    # 更新进度条
    avg_loss = running_loss / (batch_idx + 1)
    progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})

return running_loss / num_batches
```

### 5.4 _validate()

```
方法: Trainer._validate(data_loader: DataLoader) -> tuple[float, float]

返回: (val_loss, val_accuracy)

算法:

self.model.eval()
running_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for images, labels in data_loader:
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        
        outputs = self.model(images)       # (B, 10) logits
        loss = self.criterion(outputs, labels)
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)  # (B,) 预测类别索引
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_loss = running_loss / len(data_loader)
val_acc = correct / total if total > 0 else 0.0

return val_loss, val_acc
```

### 5.5 run()

```
方法: Trainer.run() -> None

主训练循环算法:

1. 恢复训练（如从检查点）:
   start_epoch = 1
   如果从检查点恢复:
       self.current_epoch = checkpoint_epoch
       self.best_val_acc = checkpoint_best_val_acc
       start_epoch = checkpoint_epoch + 1

2. 训练前显存限制:
   if torch.cuda.is_available():
       torch.cuda.set_per_process_memory_fraction(
           self.config.system.gpu_memory_fraction
       )
       torch.cuda.empty_cache()

3. 主循环:
   for epoch in range(start_epoch, self.config.train.epochs + 1):
       self.current_epoch = epoch
       epoch_start_time = time.time()
       
       # --- 训练 ---
       try:
           train_loss = self._train_one_epoch()
       except RuntimeError as e:
           if "out of memory" in str(e).lower():
               train_loss = self._handle_oom(e)
               if train_loss is None:
                   break  # 无法恢复，终止训练
           else:
               raise
       
       # --- 验证 ---
       val_loss, val_acc = self._validate(self.val_loader)
       
       # --- 学习率调度 ---
       if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
           self.scheduler.step(val_loss)       # 监控 val_loss
       else:
           self.scheduler.step()               # CosineAnnealing 按 epoch 步进
       
       current_lr = self.optimizer.param_groups[0]['lr']
       
       # --- 显存监控 ---
       peak_mem = 0.0
       if torch.cuda.is_available():
           peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
           torch.cuda.reset_peak_memory_stats()
           torch.cuda.empty_cache()
       
       # --- 记录历史 ---
       epoch_time = time.time() - epoch_start_time
       self._log_epoch(train_loss, val_loss, val_acc, current_lr, peak_mem, epoch_time)
       
       # --- TensorBoard ---
       self.writer.add_scalar('Loss/train', train_loss, epoch)
       self.writer.add_scalar('Loss/val', val_loss, epoch)
       self.writer.add_scalar('Accuracy/val', val_acc, epoch)
       self.writer.add_scalar('LR', current_lr, epoch)
       self.writer.add_scalar('Memory/peak_gb', peak_mem, epoch)
       
       # --- 检查点保存 ---
       is_best = val_acc > self.best_val_acc
       if is_best:
           self.best_val_acc = val_acc
           self.best_epoch = epoch
           self.epochs_no_improve = 0
           _save_best(self)
       else:
           self.epochs_no_improve += 1
       
       # 定期保存
       if not self.config.train.save_best_only:
           if epoch % self.config.train.save_interval == 0:
               _save_periodic(self, epoch)
       
       # --- 早停检查 ---
       if self.epochs_no_improve >= self.config.train.early_stop_patience:
           logging.info(f"早停触发: val_acc 已 {self.epochs_no_improve} 轮未改善")
           logging.info(f"最佳模型: epoch {self.best_epoch}, val_acc={self.best_val_acc:.4f}")
           break

4. 训练结束:
   self.writer.close()
   logging.info(f"训练完成。最佳 val_acc={self.best_val_acc:.4f} (epoch {self.best_epoch})")
```

### 5.6 _handle_oom()

```
方法: Trainer._handle_oom(error: RuntimeError) -> Optional[float]

CUDA OOM 恢复算法:

1. 清理显存:
   torch.cuda.empty_cache()
   import gc; gc.collect()

2. 计算新的 batch_size:
   current_bs = self.train_loader.batch_size  # 当前是类属性，非实例属性
   new_bs = max(32, current_bs // 2)
   
   if new_bs == current_bs:
       logging.error(f"batch_size 已降至下限 (32)，仍 OOM。终止训练。")
       return None  # 无法恢复
   
   logging.warning(f"CUDA OOM 恢复: batch_size {current_bs} → {new_bs}")

3. 重建 DataLoader:
   # 需要临时修改 config 中的 batch_size
   # 注意: Config 是 frozen dataclass，不可修改
   # 方案: 直接构造新的 DataLoader，传入新的 batch_size
   self.train_loader = _rebuild_dataloader_with_new_bs(
       self.train_loader.dataset, new_bs, self.config
   )
   
4. 重试当前 epoch:
   logging.info("重试当前 epoch...")
   return self._train_one_epoch()
```

### 5.7 _save_best / _save_periodic

```
内部函数: _save_best(trainer: Trainer) -> None

save_checkpoint(
    model=trainer.model,
    optimizer=trainer.optimizer,
    epoch=trainer.current_epoch,
    metrics={'val_acc': trainer.best_val_acc, 'best_val_acc': trainer.best_val_acc},
    path=os.path.join(trainer.config.train.checkpoint_dir, 'best_model.pth'),
    config=trainer.config
)

内部函数: _save_periodic(trainer: Trainer, epoch: int) -> None

filename = f"checkpoint_epoch_{epoch:03d}.pth"
save_checkpoint(
    ...,
    path=os.path.join(trainer.config.train.checkpoint_dir, filename),
    ...
)
```

### 5.8 run_training() 入口

```
函数: run_training(config: Config) -> None

编排流程:

1. 设置随机种子:
   set_seed(config.seed)

2. 创建输出目录:
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   output_dir = os.path.join(config.system.output_dir, f"train_{timestamp}")
   os.makedirs(output_dir, exist_ok=True)
   os.makedirs(config.train.checkpoint_dir, exist_ok=True)
   os.makedirs(config.train.log_dir, exist_ok=True)

3. 保存配置快照:
   save_config_snapshot(config, output_dir)

4. 准备数据:
   datasets = create_datasets(config)
   train_loader, val_loader, test_loader = get_dataloaders(config, datasets)
   # test_loader 暂时不用，留给 evaluate.py

5. 构建模型:
   model = build_model(config)

6. 定义优化器、调度器、损失:
   device = next(model.parameters()).device
   
   # 仅优化可训练参数
   trainable_params = filter(lambda p: p.requires_grad, model.parameters())
   
   optimizer = torch.optim.AdamW(
       trainable_params,
       lr=config.train.learning_rate,
       weight_decay=config.train.weight_decay
   )
   
   criterion = get_loss_fn()
   
   if config.train.lr_scheduler == "plateau":
       scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
           optimizer, mode='min', patience=config.train.lr_patience,
           factor=config.train.lr_factor
       )
   elif config.train.lr_scheduler == "cosine":
       scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
           optimizer, T_max=config.train.epochs
       )
   else:
       scheduler = None

7. 创建 Trainer 并执行:
   writer = SummaryWriter(log_dir=config.train.log_dir)
   
   trainer = Trainer(
       config, model, train_loader, val_loader,
       optimizer, scheduler, criterion, writer, device
   )
   
   # 从检查点恢复（如指定）
   if config.train.resume:
       epoch, metrics = load_checkpoint(
           config.train.resume, model, optimizer
       )
       trainer.current_epoch = epoch
       trainer.best_val_acc = metrics['best_val_acc']
   
   trainer.run()

8. 返回（不返回对象，仅副作用：checkpoint + 日志）
```

---

## 6. evaluate.py — 模型评估模块

### 6.1 模块概述

| 属性 | 说明 |
|------|------|
| 行数估算 | ~250 行 |
| 依赖 | `torch`, `numpy`, `sklearn.metrics`, `pandas`, `config`, `data`, `model`, `utils` |
| 核心职责 | 测试集评估、指标计算、混淆矩阵、分类报告生成、误分类导出 |
| 对应 ADS | §4.4, 接口 I-07 |
| 对应 SRS | FR-5 |

### 6.2 _compute_metrics()

```
函数: _compute_metrics(
    all_preds: np.ndarray,    # shape (N,), int64, 预测类别
    all_labels: np.ndarray,   # shape (N,), int64, 真实标签
    all_probs: np.ndarray,    # shape (N, 10), float32, softmax 概率
    class_names: List[str]
) -> dict

算法:

N = len(all_labels)

1. 整体准确率:
   top1_acc = (all_preds == all_labels).sum() / N
   top2_preds = np.argsort(all_probs, axis=1)[:, -2:]  # 每行 Top-2 类别
   top2_acc = np.mean([all_labels[i] in top2_preds[i] for i in range(N)])

2. 各类别 Precision/Recall/F1:
   per_class = {}
   for class_id in range(len(class_names)):
       tp = np.sum((all_preds == class_id) & (all_labels == class_id))
       fp = np.sum((all_preds == class_id) & (all_labels != class_id))
       fn = np.sum((all_preds != class_id) & (all_labels == class_id))
       
       precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
       recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
       f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
       
       per_class[class_names[class_id]] = {
           'precision': round(precision, 4),
           'recall':    round(recall, 4),
           'f1':        round(f1, 4),
           'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
           'support':   int(np.sum(all_labels == class_id))
       }

3. 宏平均和加权平均:
   macro_p = np.mean([v['precision'] for v in per_class.values()])
   macro_r = np.mean([v['recall'] for v in per_class.values()])
   macro_f1 = np.mean([v['f1'] for v in per_class.values()])
   
   supports = [v['support'] for v in per_class.values()]
   weighted_p = np.average([v['precision'] for v in per_class.values()], weights=supports)
   weighted_r = np.average([v['recall'] for v in per_class.values()], weights=supports)
   weighted_f1 = np.average([v['f1'] for v in per_class.values()], weights=supports)

4. 混淆矩阵:
   cm = sklearn.metrics.confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
   cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # 行归一化
   cm_normalized = np.nan_to_num(cm_normalized)  # 处理 0/0 → 0

5. 易混淆类别对 (Top-3):
   confused_pairs = []
   n = len(class_names)
   for i in range(n):
       for j in range(n):
           if i != j:
               confused_pairs.append({
                   'pair': (class_names[i], class_names[j]),
                   'rate': cm_normalized[i][j]
               })
   confused_pairs.sort(key=lambda x: x['rate'], reverse=True)
   top3_confused = confused_pairs[:3]

6. 返回:
   return {
       'top1_acc': round(top1_acc, 4),
       'top2_acc': round(top2_acc, 4),
       'per_class': per_class,
       'macro_avg': {'precision': macro_p, 'recall': macro_r, 'f1': macro_f1},
       'weighted_avg': {'precision': weighted_p, 'recall': weighted_r, 'f1': weighted_f1},
       'confusion_matrix': cm.tolist(),
       'confusion_matrix_normalized': cm_normalized.tolist(),
       'confused_pairs': top3_confused,
   }
```

### 6.3 _generate_report()

```
函数: _generate_report(metrics: dict, output_dir: str) -> str

生成 Markdown 评估报告，包含:

1. 整体指标表:
   | 指标 | 数值 |
   | Top-1 Accuracy | 0.9341 |
   | Top-2 Accuracy | 0.9812 |
   | Macro Avg F1   | 0.9338 |
   | Weighted Avg F1| 0.9340 |

2. 各类别指标表:
   | 类别 | Precision | Recall | F1 | Support |
   | AnnualCrop | 0.95 | 0.93 | 0.94 | 540 |
   | ... | ... | ... | ... | ... |

3. 易混淆类别分析:
   "最易混淆的 3 对类别:
    1. Industrial → Residential (混淆率 12.3%)
    2. AnnualCrop → PermanentCrop (混淆率 8.7%)
    3. ..."

4. 混淆矩阵 (Markdown 表格形式 + 图片引用)

返回: 报告文件路径
```

### 6.4 run_evaluation()

```
函数: run_evaluation(config: Config, model_path: str) -> dict

算法流程:

1. 加载模型:
   model = build_model(config)
   epoch, _ = load_checkpoint(model_path, model)  # optimizer=None
   logging.info(f"已加载模型: {model_path} (epoch {epoch})")
   model.eval()

2. 准备测试数据:
   datasets = create_datasets(config)
   _, _, test_loader = get_dataloaders(config, datasets)

3. 收集全部预测:
   all_preds = []
   all_labels = []
   all_probs = []
   device = next(model.parameters()).device
   
   with torch.no_grad():
       progress_bar = tqdm(test_loader, desc="评估中")
       for images, labels in progress_bar:
           images = images.to(device, non_blocking=True)
           
           outputs = model(images)            # (B, 10) logits
           probs = torch.softmax(outputs, dim=1)  # (B, 10)
           
           _, predicted = torch.max(outputs, 1)
           
           all_preds.append(predicted.cpu().numpy())
           all_labels.append(labels.numpy())
           all_probs.append(probs.cpu().numpy())

4. 合并为 numpy 数组:
   all_preds = np.concatenate(all_preds)
   all_labels = np.concatenate(all_labels)
   all_probs = np.concatenate(all_probs)

5. 计算指标:
   metrics = _compute_metrics(all_preds, all_labels, all_probs, config.data.class_names)

6. 生成报告:
   output_dir = os.path.join(config.system.output_dir, "evaluation")
   os.makedirs(output_dir, exist_ok=True)
   
   # JSON 输出
   json_path = os.path.join(output_dir, "eval_results.json")
   with open(json_path, 'w', encoding='utf-8') as f:
       json.dump(metrics, f, indent=2, ensure_ascii=False)
   
   # Markdown 报告
   report_path = _generate_report(metrics, output_dir)
   
   # 混淆矩阵图
   cm = np.array(metrics['confusion_matrix_normalized'])
   plot_confusion_matrix(cm, config.data.class_names,
                         os.path.join(output_dir, "confusion_matrix.png"))
   
   # 各类别准确率柱状图
   class_accs = {name: metrics['per_class'][name]['recall']  # recall = per-class accuracy
                 for name in config.data.class_names}
   _plot_class_accuracy_bar(class_accs,
                            os.path.join(output_dir, "class_accuracy.png"))

7. 导出误分类样本:
   misclassified = _export_misclassified(all_preds, all_labels, all_probs,
                                          config.data.class_names, output_dir)

8. 返回 metrics
```

---

## 7. predict.py — 推理与预测模块

### 7.1 模块概述

| 属性 | 说明 |
|------|------|
| 行数估算 | ~200 行 |
| 依赖 | `torch`, `Pillow`, `numpy`, `pandas`, `config`, `model` |
| 核心职责 | 单张/批量推理、通道自适应、Top-K 预测、结果汇总 |
| 对应 ADS | §4.5, 接口 I-08, I-09 |
| 对应 SRS | FR-6 |

### 7.2 _load_and_preprocess()

```
函数: _load_and_preprocess(image_path: str, config: Config, device: torch.device) -> tuple[Tensor, PIL.Image]

算法:

1. 加载图像:
   try:
       image = Image.open(image_path)
   except (PIL.UnidentifiedImageError, OSError) as e:
       raise ValueError(f"图像加载失败: {image_path} ({e})")

2. 通道自适应:
   mode = image.mode
   if mode == 'L':        # 灰度图
       image = image.convert('RGB')
       logging.info(f"灰度图已转换为 RGB: {image_path}")
   elif mode == 'RGBA':   # RGBA
       image = image.convert('RGB')
       logging.info(f"RGBA 已丢弃 Alpha 通道: {image_path}")
   elif mode == 'CMYK':
       image = image.convert('RGB')
       logging.info(f"CMYK 已转换为 RGB: {image_path}")
   elif mode != 'RGB':
       raise ValueError(f"不支持的图像模式: {mode} ({image_path})")

3. 预处理流水线:
   transform = transforms.Compose([
       transforms.Resize((config.data.input_size, config.data.input_size),
                         interpolation=transforms.InterpolationMode.BILINEAR),
       transforms.ToTensor(),
       transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
   ])
   tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)
   tensor = tensor.to(device)

4. 返回:
   return tensor, image  # image 用于后续可视化
```

### 7.3 predict_single_image()

```
函数: predict_single_image(
    image_path: str,
    model_path: str,
    config: Config
) -> tuple[str, float, list]

算法:

1. 校验图像:
   if not os.path.exists(image_path):
       raise FileNotFoundError(f"图像不存在: {image_path}")

2. 构建并加载模型:
   model = build_model(config)
   load_checkpoint(model_path, model)  # optimizer=None
   model.eval()
   device = next(model.parameters()).device

3. 预处理:
   image_tensor, _ = _load_and_preprocess(image_path, config, device)

4. 推理:
   with torch.no_grad():
       outputs = model(image_tensor)           # (1, 10)
       probs = torch.softmax(outputs, dim=1)   # (1, 10)
       top_k_probs, top_k_indices = torch.topk(probs, config.inference.top_k, dim=1)
   
   top_k_probs = top_k_probs.squeeze(0).cpu().numpy()
   top_k_indices = top_k_indices.squeeze(0).cpu().numpy()

5. 组装结果:
   predicted_class = config.data.class_names[top_k_indices[0]]
   confidence = float(top_k_probs[0])
   
   top_k_list = [
       (config.data.class_names[int(idx)], float(prob))
       for idx, prob in zip(top_k_indices, top_k_probs)
   ]

6. 低置信度告警:
   if confidence < config.inference.conf_warning_threshold:
       logging.warning(
           f"低置信度预测: {predicted_class} ({confidence:.4f}) — "
           f"可能为域外图像"
       )

7. 返回:
   return (predicted_class, round(confidence, 4), top_k_list)
```

### 7.4 predict_batch()

```
函数: predict_batch(
    input_dir: str,
    model_path: str,
    config: Config
) -> str  # 返回 CSV 文件路径

算法:

1. 校验输入目录:
   input_path = Path(input_dir)
   if not input_path.exists():
       raise FileNotFoundError(f"输入目录不存在: {input_dir}")
   
   image_files = []
   for ext in SUPPORTED_EXTENSIONS:
       image_files.extend(input_path.glob(f'*{ext}'))
       image_files.extend(input_path.glob(f'*{ext.upper()}'))
   image_files = sorted(set(image_files))  # 去重+排序
   
   if len(image_files) == 0:
       logging.info(f"输入目录无有效图像文件: {input_dir}")
       return ""

2. 构建模型（仅一次）:
   model = build_model(config)
   load_checkpoint(model_path, model)
   model.eval()
   device = next(model.parameters()).device

3. 批量推理:
   results = []
   skipped = []
   success_count = 0
   
   progress_bar = tqdm(image_files, desc="批量推理")
   for img_path in progress_bar:
       try:
           tensor, _ = _load_and_preprocess(str(img_path), config, device)
           with torch.no_grad():
               outputs = model(tensor)
               probs = torch.softmax(outputs, dim=1)
               top_k_probs, top_k_indices = torch.topk(probs, config.inference.top_k, dim=1)
           
           top_k_probs = top_k_probs.squeeze(0).cpu().numpy()
           top_k_indices = top_k_indices.squeeze(0).cpu().numpy()
           
           result = {
               'image_path': str(img_path),
               'predicted_class': config.data.class_names[int(top_k_indices[0])],
               'confidence': round(float(top_k_probs[0]), 4)
           }
           # Top-K 列
           for k in range(config.inference.top_k):
               result[f'top{k+1}_class'] = config.data.class_names[int(top_k_indices[k])]
               result[f'top{k+1}_confidence'] = round(float(top_k_probs[k]), 4)
           
           results.append(result)
           success_count += 1
           
       except Exception as e:
           logging.warning(f"跳过: {img_path} ({e})")
           skipped.append({'file': str(img_path), 'error': str(e)})

4. 保存结果:
   output_dir = Path(config.inference.output_dir)
   output_dir.mkdir(parents=True, exist_ok=True)
   
   # CSV
   df = pd.DataFrame(results)
   csv_path = output_dir / "predictions.csv"
   df.to_csv(csv_path, index=False, encoding='utf-8-sig')
   
   # 跳过文件列表
   if skipped:
       skip_path = output_dir / "skipped_files.txt"
       with open(skip_path, 'w', encoding='utf-8') as f:
           for s in skipped:
               f.write(f"{s['file']}\t{s['error']}\n")

5. 汇总统计:
   total = len(image_files)
   class_distribution = df['predicted_class'].value_counts()
   
   logging.info(f"批量推理完成:")
   logging.info(f"  成功: {success_count}/{total}")
   logging.info(f"  失败: {len(skipped)}/{total}")
   logging.info(f"  类别分布:")
   for class_name in config.data.class_names:
       count = class_distribution.get(class_name, 0)
       pct = count / success_count * 100 if success_count > 0 else 0
       logging.info(f"    {class_name}: {count} ({pct:.1f}%)")

6. 返回:
   return str(csv_path)
```

---

## 8. utils.py — 工具函数模块

### 8.1 模块概述

| 属性 | 说明 |
|------|------|
| 行数估算 | ~200 行 |
| 依赖 | `torch`, `tensorboard`, `matplotlib`, `seaborn`, `tqdm`, `numpy`, `random` |
| 核心职责 | TensorBoard 日志、训练曲线图、混淆矩阵热力图、GPU 显存监控、随机种子设置 |
| 对应 ADS | §4.7 |
| 对应 SRS | FR-7 |

### 8.2 set_seed()

```
函数: set_seed(seed: int = 42) -> None

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 确保 GPU 上卷积算子的确定性（轻微性能代价）
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logging.info(f"全局随机种子已设置: {seed}")
```

### 8.3 log_gpu_memory()

```
函数: log_gpu_memory(device: torch.device) -> float

if device.type == 'cpu':
    return 0.0

# 读取峰值显存
peak_bytes = torch.cuda.max_memory_allocated(device)
peak_gb = peak_bytes / (1024 ** 3)

# 重置峰值计数器（为下一 epoch 做准备）
torch.cuda.reset_peak_memory_stats(device)

# 记录日志
if peak_gb > 2.5:
    logging.warning(f"峰值显存偏高: {peak_gb:.2f} GB")
else:
    logging.info(f"峰值显存: {peak_gb:.2f} GB")

return round(peak_gb, 4)
```

### 8.4 plot_training_curves()

```
函数: plot_training_curves(history: dict, output_dir: str) -> None

输入 history:
  {
      'train_loss': [float, ...],
      'val_loss': [float, ...],
      'val_acc': [float, ...],
      'lr': [float, ...],
  }

生成图表:

1. Loss 曲线 (train_loss + val_loss, 双线):
   fig, ax = plt.subplots(figsize=(10, 6))
   epochs = range(1, len(history['train_loss']) + 1)
   
   ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=1.5)
   ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=1.5)
   ax.set_xlabel('Epoch')
   ax.set_ylabel('Loss')
   ax.set_title('Training and Validation Loss')
   ax.legend()
   ax.grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=200)
   plt.close()

2. Accuracy 曲线:
   fig, ax = plt.subplots(figsize=(10, 6))
   ax.plot(epochs, history['val_acc'], 'g-', label='Val Accuracy', linewidth=1.5)
   ax.set_xlabel('Epoch')
   ax.set_ylabel('Accuracy')
   ax.set_title('Validation Accuracy')
   ax.legend()
   ax.grid(True, alpha=0.3)
   
   # 标注最佳点
   best_epoch = np.argmax(history['val_acc']) + 1
   best_acc = max(history['val_acc'])
   ax.annotate(f'Best: {best_acc:.4f}',
               xy=(best_epoch, best_acc),
               xytext=(best_epoch + 2, best_acc - 0.02),
               arrowprops=dict(arrowstyle='->', color='red'),
               fontsize=10, color='red')
   
   plt.tight_layout()
   plt.savefig(os.path.join(output_dir, 'accuracy_curve.png'), dpi=200)
   plt.close()
```

### 8.5 plot_confusion_matrix()

```
函数: plot_confusion_matrix(
    cm: np.ndarray,        # shape (10, 10), 归一化后 [0, 1]
    class_names: List[str],
    output_path: str,
    dpi: int = 200
) -> None

算法:

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,                  # 格子显示数值
    fmt='.2f',                   # 保留 2 位小数
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names,
    vmin=0, vmax=1,
    cbar_kws={'label': '比例'}  # colorbar 标签
)
plt.xlabel('预测类别', fontsize=12)
plt.ylabel('真实类别', fontsize=12)
plt.title('混淆矩阵 (行归一化)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
plt.close()
```

### 8.6 _plot_class_accuracy_bar()（在 evaluate.py 中本可内联，但封装到 utils 中可复用）

```
函数: plot_class_accuracy_bar(
    class_accs: Dict[str, float],
    output_path: str,
    dpi: int = 200
) -> None

算法:

# 按准确率降序排列
sorted_items = sorted(class_accs.items(), key=lambda x: x[1], reverse=True)
names = [item[0] for item in sorted_items]
accs = [item[1] for item in sorted_items]

# 计算平均准确率（用于高亮低于均值的类别）
mean_acc = np.mean(accs)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#2ecc71' if acc >= mean_acc else '#e74c3c' for acc in accs]
bars = ax.barh(names, accs, color=colors)

ax.axvline(x=mean_acc, color='gray', linestyle='--', linewidth=1, label=f'平均: {mean_acc:.3f}')
ax.set_xlabel('Accuracy (Recall)')
ax.set_title('各类别准确率')
ax.legend()
ax.set_xlim(0, 1.05)

# 柱上标注数值
for bar, acc in zip(bars, accs):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f'{acc:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
plt.close()
```

---

## 9. main.py — CLI 统一入口模块

### 9.1 模块概述

| 属性 | 说明 |
|------|------|
| 行数估算 | ~150 行 |
| 依赖 | `argparse`, `sys`, `config`, `data`, `model`, `train`, `evaluate`, `predict`, `utils` |
| 核心职责 | CLI 参数解析、模式路由、环境检测 |
| 对应 ADS | §4.8 |
| 对应 SRS | FR-8 |

### 9.2 build_parser()

```
函数: build_parser() -> argparse.ArgumentParser

定义全部 CLI 参数:

parser = argparse.ArgumentParser(
    description="EuroSAT_Classification — 基于 ResNet18 的卫星影像土地覆盖分类系统",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
用法示例:
  python main.py --mode train --config configs/config.yaml
  python main.py --mode evaluate --model checkpoints/best_model.pth
  python main.py --mode predict --input test_image.jpg
  python main.py --mode check
    """
)

parser.add_argument('--mode', type=str, required=True,
                    choices=['train', 'evaluate', 'predict', 'check'],
                    help='运行模式')
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help='配置文件路径')
parser.add_argument('--data-dir', type=str, default=None,
                    help='EuroSAT 数据集根目录（覆盖配置文件）')
parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                    help='模型权重路径（evaluate/predict 模式）')
parser.add_argument('--input', type=str, default=None,
                    help='输入图像路径或文件夹（predict 模式）')
parser.add_argument('--output', type=str, default=None,
                    help='结果输出目录（predict 模式，覆盖配置文件）')
parser.add_argument('--top-k', type=int, default=None,
                    help='返回 Top-K 预测（predict 模式）')
parser.add_argument('--batch-size', type=int, default=None,
                    help='批大小（覆盖配置文件）')
parser.add_argument('--epochs', type=int, default=None,
                    help='训练轮数（覆盖配置文件）')
parser.add_argument('--lr', type=float, default=None,
                    help='学习率（覆盖配置文件）')
parser.add_argument('--resume', type=str, default=None,
                    help='从检查点恢复训练')

return parser
```

### 9.3 main()

```
函数: main() -> None

算法:

1. 解析参数:
   parser = build_parser()
   args = parser.parse_args()

2. 准备 CLI 覆盖字典:
   cli_overrides = {}
   if args.data_dir:
       cli_overrides['data.root_dir'] = args.data_dir
   if args.batch_size:
       cli_overrides['train.batch_size'] = args.batch_size
   if args.epochs:
       cli_overrides['train.epochs'] = args.epochs
   if args.lr:
       cli_overrides['train.learning_rate'] = args.lr
   if args.output:
       cli_overrides['inference.output_dir'] = args.output
   if args.top_k:
       cli_overrides['inference.top_k'] = args.top_k
   if args.resume:
       cli_overrides['train.resume'] = args.resume

3. 模式路由:
   if args.mode == 'check':
       mode_check(args)
   else:
       config = load_config(args.config, cli_overrides)
       set_seed(config.seed)
       
       if args.mode == 'train':
           mode_train(config)
       elif args.mode == 'evaluate':
           mode_evaluate(config, args.model)
       elif args.mode == 'predict':
           mode_predict(config, args, args.model)
```

### 9.4 mode_check()

```
函数: mode_check(args) -> None

环境检测项:

print("=" * 60)
print("EuroSAT_Classification 环境检测")
print("=" * 60)

1. Python 版本:
   py_ver = sys.version_info
   print(f"[{'通过' if py_ver.major==3 and py_ver.minor==8 else '警告'}] "
         f"Python 版本: {py_ver.major}.{py_ver.minor}.{py_ver.micro}")
   if (py_ver.major, py_ver.minor) != (3, 8):
       print("  ⚠ 建议使用 Python 3.8.x (当前 PyTorch 1.12.1 最佳兼容版本)")

2. PyTorch:
   try:
       import torch
       print(f"[通过] PyTorch 版本: {torch.__version__}")
       print(f"[通过] CUDA 可用: {torch.cuda.is_available()}")
       if torch.cuda.is_available():
           print(f"[通过] CUDA 版本: {torch.version.cuda}")
           print(f"[通过] GPU: {torch.cuda.get_device_name(0)}")
           mem_total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
           status = "通过" if mem_total >= 3.5 else "警告"
           print(f"[{status}] GPU 显存总量: {mem_total:.1f} GB")
   except ImportError:
       print("[失败] PyTorch 未安装")

3. TorchVision:
   try:
       import torchvision
       print(f"[通过] TorchVision 版本: {torchvision.__version__}")
   except ImportError:
       print("[失败] TorchVision 未安装")

4. 预训练权重可访问性:
   try:
       model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
       print("[通过] ImageNet-1K 预训练权重可访问")
   except Exception as e:
       print(f"[警告] 预训练权重访问失败: {e}")
       print("  首次训练时将从网络下载（约 45 MB）")

5. 数据路径:
   data_dir = args.data_dir or "D:/DataDownload/EuroSat_Dataset/EuroSAT"
   data_path = Path(data_dir)
   if data_path.exists():
       class_dirs = [d.name for d in data_path.iterdir() if d.is_dir()]
       expected = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
                   'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
       missing = set(expected) - set(class_dirs)
       extra = set(class_dirs) - set(expected)
       if not missing and not extra:
           print(f"[通过] 数据路径: {data_dir}")
           for cls in expected:
               count = len(list((data_path / cls).glob('*.jpg'))) + \
                       len(list((data_path / cls).glob('*.png')))
               print(f"  {cls}: {count} 张")
       else:
           if missing:
               print(f"[失败] 缺失类别文件夹: {missing}")
           if extra:
               print(f"[警告] 多余文件夹: {extra}")
   else:
       print(f"[警告] 数据路径不存在: {data_dir}")
       print("  请在 config.yaml 中设置正确的 data.root_dir")

6. 磁盘空间:
   import shutil
   for label, check_path in [("项目目录", "."), ("数据目录", data_dir)]:
       try:
           usage = shutil.disk_usage(check_path)
           free_gb = usage.free / (1024**3)
           if free_gb >= 10:
               print(f"[通过] {label} 剩余空间: {free_gb:.1f} GB ({check_path})")
           elif free_gb >= 1:
               print(f"[警告] {label} 剩余空间不足: {free_gb:.1f} GB ({check_path})")
           else:
               print(f"[失败] {label} 剩余空间严重不足: {free_gb:.1f} GB ({check_path})")
       except Exception:
           print(f"[跳过] 无法检查 {label} 磁盘空间")

print("=" * 60)
```

---

## 附录A 异常码定义

| 异常码 | 异常类型 | 触发条件 | 处理方式 |
|--------|---------|---------|---------|
| E-CFG-001 | `FileNotFoundError` | 配置文件不存在 | 终止，提示检查路径 |
| E-CFG-002 | `ValueError` | YAML 格式错误 | 终止，提示错误位置 |
| E-CFG-003 | `ValueError` | 配置字段非法值 | 终止，列出合法值范围 |
| E-DAT-001 | `FileNotFoundError` | 数据根目录不存在 | 终止，提示检查 data.root_dir |
| E-DAT-002 | `ValueError` | 类别文件夹不完整 | 终止，列出缺失/多余文件夹 |
| E-DAT-003 | `IOError` | 图像文件损坏 | 跳过该文件，记录 WARNING |
| E-MOD-001 | `ValueError` | 不支持的模型名称 | 终止，列出合法值 |
| E-MOD-002 | `RuntimeError` | 检查点格式不兼容 | 终止，提示检查 PyTorch 版本 |
| E-TRN-001 | `RuntimeError` | CUDA OOM | 自动恢复（降 batch_size） |
| E-TRN-002 | `KeyboardInterrupt` | 用户中断训练 | 保存紧急检查点后退出 |
| E-TRN-003 | `ValueError` | loss 为 NaN | WARNING，输出超参诊断 |
| E-PRD-001 | `ValueError` | 图像文件损坏/不支持的格式 | 单张=终止；批量=跳过继续 |
| E-PRD-002 | `PermissionError` | 输出目录不可写 | 终止，提示检查权限 |

---

## 附录B 需求追溯矩阵

| 需求编号 | 需求简述 | 本文章节 | 对应函数/类 | 状态 |
|:---:|------|------|------|:---:|
| FR-1 | 数据加载与预处理 | §3 | `create_datasets()`, `get_dataloaders()`, `RobustImageFolder` | ⚪ |
| FR-2 | 数据增强 | §3 | `_build_train_transform()`, `_build_eval_transform()` | ⚪ |
| FR-3 | 模型构建与迁移学习 | §4 | `build_model()`, `freeze_backbone()`, `_unfreeze_last_n_blocks()` | ⚪ |
| FR-4 | 模型训练 | §5 | `Trainer`, `run_training()`, `_handle_oom()` | ⚪ |
| FR-5 | 模型评估 | §6 | `run_evaluation()`, `_compute_metrics()`, `_generate_report()` | ⚪ |
| FR-6 | 推理与预测 | §7 | `predict_single_image()`, `predict_batch()`, `_load_and_preprocess()` | ⚪ |
| FR-7 | 训练监控与可视化 | §8 | `set_seed()`, `log_gpu_memory()`, `plot_training_curves()`, `plot_confusion_matrix()` | ⚪ |
| FR-8 | CLI 统一入口 | §9 | `main()`, `build_parser()`, `mode_check()` | ⚪ |
| NFR-P1 | 训练显存 ≤ 3.0 GB | §5.5 | `Trainer.run()` 显存限制 + `log_gpu_memory()` | ⚪ |
| NFR-P5 | 训练耗时 ≤ 50 min | §5 | 50 epochs, batch=256 | ⚪ |
| NFR-R1 | 断点续训 | §4.7, §5.5 | `load_checkpoint()`, `Trainer.run()` resume 分支 | ⚪ |
| NFR-R4 | 随机种子固定 | §8.2 | `set_seed(42)` | ⚪ |
| NFR-M1 | 模块化 | §2~§9 | 8 模块独立设计 | ⚪ |

---

> **文档结束** | 本文档从代码级详细描述了 EuroSAT_Classification 系统全部 8 个模块的实现逻辑。开发者可按本文档逐函数翻译为 Python 代码，测试者可对照本文档编写白盒测试用例。
