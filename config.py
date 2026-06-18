"""
配置管理模块
依据: DDS §2, ADS §4.6, 接口 I-10
职责: YAML 配置加载 → 校验 → 后处理 → 不可变 Config 对象
"""
import os
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Tuple, Any

import yaml

logger = logging.getLogger(__name__)


# ============================================================
# 7 个冻结数据类（不可变配置对象）— 依据: DDS §2.2
# ============================================================

@dataclass(frozen=True)
class SystemConfig:
    """系统配置：GPU显存限制、日志级别、输出目录"""
    gpu_id: int = 0                    # GPU 设备编号（-1=CPU）
    gpu_memory_fraction: float = 0.75  # 进程显存使用比例上限 (0,1]
    num_workers: int = 4               # DataLoader 子进程数
    log_level: str = "INFO"            # 日志级别: DEBUG/INFO/WARNING/ERROR
    output_dir: str = "./runs"         # 输出根目录


@dataclass(frozen=True)
class DataConfig:
    """数据配置：数据集路径、类别信息、划分比例、图像尺寸"""
    root_dir: str = "D:/DataDownload/EuroSat_Dataset/EuroSAT"  # EuroSAT 根目录（必填）
    num_classes: int = 10                                       # 类别数（固定 10）
    class_names: Tuple[str, ...] = (                            # 类别名列表（按字母序→标签0~9）
        "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
        "Industrial", "Pasture", "PermanentCrop", "Residential",
        "River", "SeaLake"
    )
    train_ratio: float = 0.7    # 训练集比例
    val_ratio: float = 0.1      # 验证集比例（测试集=1-train_ratio-val_ratio）
    input_size: int = 224       # ResNet18 输入尺寸（固定）
    original_size: int = 64     # EuroSAT 原始图像尺寸（用于日志记录）


@dataclass(frozen=True)
class AugmentationConfig:
    """数据增强配置：翻转、旋转、色彩抖动"""
    horizontal_flip: float = 0.5     # 随机水平翻转概率（0=关闭）
    rotation_degrees: float = 15.0   # 随机旋转最大角度（0=关闭）
    brightness: float = 0.2          # 亮度调整幅度（0=关闭）
    contrast: float = 0.2            # 对比度调整幅度（0=关闭）


@dataclass(frozen=True)
class ModelConfig:
    """模型配置：架构选择、预训练权重、冻结策略"""
    name: str = "resnet18"                       # 模型名称: resnet18 / mobilenet_v3_large
    pretrained: bool = True                      # 是否加载预训练权重
    pretrained_weights: str = "IMAGENET1K_V1"   # 在线加载: IMAGENET1K_V1; 本地路径: D:/weights/resnet18.pth
    pretrained_path: str = ""                    # 本地预训练权重文件路径（优先级高于在线下载，空=在线）
    freeze_backbone: bool = True                 # 是否冻结骨干网络（迁移学习）
    unfreeze_layers: int = 0                     # 解冻最后 N 个残差块（0=全部冻结, 1=layer4, ..., 4=全部解冻）


@dataclass(frozen=True)
class TrainConfig:
    """训练配置：超参数、优化器、学习率调度、早停、检查点"""
    batch_size: int = 256                # 训练批次大小
    epochs: int = 50                     # 总训练轮数（推荐 50，快速验证可设 30）
    learning_rate: float = 0.0001        # 初始学习率（1e-4）
    optimizer: str = "adamw"             # 优化器: adamw / adam / sgd
    weight_decay: float = 0.0005         # 权重衰减（5e-4，仅 AdamW/Adam 生效）
    momentum: float = 0.9                # SGD 动量（仅 optimizer=sgd 时生效）
    loss: str = "cross_entropy"          # 损失函数类型
    lr_scheduler: str = "plateau"        # 学习率调度策略: plateau / cosine / none
    lr_patience: int = 5                 # ReduceLROnPlateau 的 patience（等待轮数）
    lr_factor: float = 0.5               # ReduceLROnPlateau 的衰减因子
    early_stop_patience: int = 10        # 早停耐心值（连续 N 轮无改善则停止，0=关闭）
    early_stop_min_delta: float = 0.001  # 早停最小改善阈值
    checkpoint_dir: str = "checkpoints"  # 检查点保存目录
    log_dir: str = "logs"                # TensorBoard 日志目录
    save_best_only: bool = False         # True=仅保存最佳模型, False=额外保存定期检查点
    save_interval: int = 10              # 每隔 N 个 epoch 保存定期检查点
    resume: str = ""                     # 恢复训练的检查点路径（空字符串=从头训练）


@dataclass(frozen=True)
class InferenceConfig:
    """推理配置：批大小、Top-K、置信度告警阈值、输出格式"""
    batch_size: int = 256                    # 推理批次大小
    top_k: int = 2                           # 返回 Top-K 预测结果
    conf_warning_threshold: float = 0.3      # 低置信度告警阈值（max confidence < 此值时告警）
    output_dir: str = "./outputs/predictions"  # 预测结果输出目录
    save_format: str = "csv"                 # 结果保存格式: csv / json


@dataclass(frozen=True)
class VisualizationConfig:
    """可视化配置：DPI、图像尺寸、配色方案"""
    dpi: int = 200               # 输出图像分辨率（DPI）
    figsize: Tuple[int, int] = (10, 8)  # 图像尺寸（英寸，宽×高）
    cmap: str = "Blues"          # 混淆矩阵配色方案（Matplotlib colormap 名称）
    show_values: bool = True     # 混淆矩阵格子中是否显示数值
    plot_format: str = "png"     # 输出图像格式: png / pdf / svg


@dataclass(frozen=True)
class Config:
    """顶层配置聚合（不可变）— 依据: DDS §2.2
    包含 7 个子配置对象 + 实验名称 + 全局随机种子。
    通过 load_config() 创建，创建后不可修改。
    """
    experiment_name: str = "eurosat_classification"  # 实验名称
    seed: int = 42                                     # 全局随机种子（torch/numpy/random）
    system: SystemConfig = field(default_factory=SystemConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


# ============================================================
# 子配置注册表 — YAML 键名 → 对应的 dataclass
# ============================================================
_SUB_CONFIGS = {
    "system":        SystemConfig,
    "data":          DataConfig,
    "augmentation":  AugmentationConfig,
    "model":         ModelConfig,
    "train":         TrainConfig,
    "inference":     InferenceConfig,
    "visualization": VisualizationConfig,
}


# ============================================================
# 核心函数 — 依据: DDS §2.3
# ============================================================

def _set_nested_key(raw_dict: dict, key_path: str, value: Any) -> None:
    """通过点号分隔的路径设置嵌套字典值。
    例: _set_nested_key(d, 'train.batch_size', 128) → d['train']['batch_size'] = 128
    Args:
        raw_dict: 原始字典（就地修改）
        key_path: 点号分隔的键路径，如 'train.batch_size'
        value: 要设置的值
    """
    keys = key_path.split(".")
    d = raw_dict
    for k in keys[:-1]:
        if k not in d:
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def _build_config(raw_dict: dict) -> Config:
    """从原始字典递归构造 Config 对象。
    - YAML 中有但 dataclass 无的字段 → 忽略 + WARNING
    - dataclass 中有但 YAML 无的字段 → 使用默认值
    Args:
        raw_dict: yaml.safe_load 返回的原始字典
    Returns:
        不可变 Config 对象
    """
    top_kwargs = {}
    for key, sub_cls in _SUB_CONFIGS.items():
        sub_dict = raw_dict.get(key, {})
        if not isinstance(sub_dict, dict):
            sub_dict = {}
        fields = {f.name for f in sub_cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in sub_dict.items() if k in fields}
        extra = [k for k in sub_dict if k not in fields]
        for k in extra:
            logger.warning(f"未知配置字段将被忽略: {key}.{k}")
        sub_config = sub_cls(**filtered)
        top_kwargs[key] = sub_config

    # 顶层字段: experiment_name, seed
    top_fields = {"experiment_name", "seed"}
    for k, v in raw_dict.items():
        if k in top_fields:
            top_kwargs[k] = v
        elif k not in _SUB_CONFIGS:
            logger.warning(f"未知顶层配置字段将被忽略: {k}")

    return Config(**top_kwargs)


def _validate(config: Config) -> None:
    """校验 Config 对象的合法性（9 条规则）。
    依据: DDS §2.3.2
    Args:
        config: Config 对象
    Raises:
        ValueError: 配置值不合法时抛出
    """
    # 1. gpu_memory_fraction 必须在 (0, 1] 范围内
    if not (0 < config.system.gpu_memory_fraction <= 1):
        raise ValueError(
            f"system.gpu_memory_fraction 必须在 (0, 1] 范围内，当前值: {config.system.gpu_memory_fraction}"
        )

    # 2. data.root_dir 不能为空
    if not config.data.root_dir:
        raise ValueError("data.root_dir 未设置，请在 configs/config.yaml 中指定数据集路径")

    # 3. train_ratio + val_ratio 必须 < 1.0
    if config.data.train_ratio + config.data.val_ratio >= 1.0:
        raise ValueError(
            f"train_ratio ({config.data.train_ratio}) + val_ratio ({config.data.val_ratio}) 必须 < 1.0"
        )

    # 4. num_classes 必须等于 class_names 的长度
    if config.data.num_classes != len(config.data.class_names):
        raise ValueError(
            f"num_classes ({config.data.num_classes}) 与 class_names 数量 ({len(config.data.class_names)}) 不一致"
        )

    # 5. batch_size 必须 > 0
    if config.train.batch_size <= 0:
        raise ValueError(f"train.batch_size 必须 > 0，当前值: {config.train.batch_size}")

    # 6. epochs 必须 > 0
    if config.train.epochs <= 0:
        raise ValueError(f"train.epochs 必须 > 0，当前值: {config.train.epochs}")

    # 7. learning_rate 必须 > 0
    if config.train.learning_rate <= 0:
        raise ValueError(f"train.learning_rate 必须 > 0，当前值: {config.train.learning_rate}")

    # 8. unfreeze_layers 必须在 [0, 4] 范围内
    if not (0 <= config.model.unfreeze_layers <= 4):
        raise ValueError(
            f"model.unfreeze_layers 必须在 [0, 4] 范围内，当前值: {config.model.unfreeze_layers}"
        )
    if config.model.unfreeze_layers > 0 and not config.model.freeze_backbone:
        logger.warning("freeze_backbone=False 时 unfreeze_layers 参数被忽略")

    # 9. 危险参数组合检测
    if config.model.unfreeze_layers >= 2 and config.train.batch_size >= 256:
        logger.warning(
            "危险参数组合: unfreeze_layers>=2 + batch_size>=256 可能导致显存超限，"
            "建议降低 batch_size 或减少解冻层数"
        )


def load_config(path: str, cli_args: dict = None) -> Config:
    """加载 YAML 配置文件 → CLI 参数覆盖 → 校验 → 返回不可变 Config 对象。
    依据: DDS §2.3.1
    Args:
        path: YAML 配置文件路径
        cli_args: 命令行参数字典，键为点号分隔路径（如 'train.batch_size'），值为覆盖值
    Returns:
        校验后的不可变 Config 对象
    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: 配置值不合法
    """
    # 1. 检查文件存在
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    # 2. 加载 YAML
    with open(config_path, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f)
    if raw_dict is None:
        raw_dict = {}

    # 3. CLI 参数覆盖（优先级高于配置文件）
    if cli_args:
        for key, value in cli_args.items():
            if value is not None:
                _set_nested_key(raw_dict, key, value)

    # 4. 递归构造 Config dataclass
    config = _build_config(raw_dict)

    # 5. 校验
    _validate(config)

    # 6. 根据配置设置日志级别
    logging.getLogger().setLevel(getattr(logging, config.system.log_level.upper(), logging.INFO))

    return config


def save_config_snapshot(cfg: Config, output_dir: str) -> str:
    """将 Config 对象保存为带时间戳的 YAML 快照文件，用于实验复现。
    依据: DDS §2.3.3
    Args:
        cfg: Config 对象
        output_dir: 输出目录
    Returns:
        快照文件的完整路径
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"config_{timestamp}.yaml"
    filepath = os.path.join(output_dir, filename)

    raw = asdict(cfg)
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(raw, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    logger.info(f"配置快照已保存: {filepath}")
    return filepath
