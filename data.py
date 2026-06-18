"""
数据加载与增强模块
依据: DDS §3, ADS §4.1, 接口 I-01 / I-02, SRS FR-1 / FR-2
职责: EuroSAT 数据集加载 → 7:1:2 分层划分 → 64→224 上采样 → ImageNet 标准化
      → 数据增强（仅训练集）→ PyTorch DataLoader 封装
"""
import os
import logging
from typing import Tuple, List

from PIL import UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
from torchvision import transforms
from sklearn.model_selection import train_test_split

from config import Config

logger = logging.getLogger(__name__)


# ============================================================
# 常量 — 依据: DDS §3.2
# ============================================================

# ImageNet 标准化参数（与 ImageNet 预训练权重一致）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# 支持的图像格式
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")


# ============================================================
# 内部辅助类
# ============================================================

class RobustImageFolder(torchvision.datasets.ImageFolder):
    """带损坏文件容错的 ImageFolder。
    重写 __getitem__ 以捕获 PIL 图像加载异常，跳过损坏文件并记录日志。
    依据: DDS §3.7
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skipped_files: List[str] = []

    def __getitem__(self, index: int):
        """获取第 index 个样本。若图像损坏则递归查找下一个有效样本。
        Args:
            index: 样本索引
        Returns:
            (image, label) 元组
        """
        try:
            return super().__getitem__(index)
        except (UnidentifiedImageError, OSError) as e:
            path = self.imgs[index][0]  # ImageFolder 内部: imgs = [(path, label), ...]
            logger.warning(f"跳过损坏文件: {path} ({e})")
            self.skipped_files.append(path)
            # 递归到下一个样本
            next_index = (index + 1) % len(self)
            return self.__getitem__(next_index)


class SubsetWithTransform(Subset):
    """带 transform 的数据子集。
    依据: DDS §3.5 内部类设计
    """

    def __init__(self, dataset: Dataset, indices: List[int], transform):
        """Args:
            dataset: 原始完整数据集
            indices: 子集索引列表
            transform: torchvision transforms.Compose 对象（或 None）
        """
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx: int):
        image, label = self.dataset[self.indices[idx]]
        # image: PIL Image（原始 64×64）
        # label: int (0~9)
        if self.transform:
            image = self.transform(image)
        return image, label


# ============================================================
# 增强管线构建 — 依据: DDS §3.3 / §3.4
# ============================================================

def _build_train_transform(config: Config) -> transforms.Compose:
    """构建训练集增强流水线。
    顺序: Resize(224, BILINEAR) → RandomHorizontalFlip(0.5) → RandomRotation(±15°)
          → ColorJitter(±0.2) → ToTensor → Normalize(ImageNet)
    依据: DDS §3.3
    Args:
        config: Config 对象
    Returns:
        torchvision.transforms.Compose 对象
    """
    return transforms.Compose([
        transforms.Resize(
            (config.data.input_size, config.data.input_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.RandomHorizontalFlip(p=config.augmentation.horizontal_flip),
        transforms.RandomRotation(
            degrees=config.augmentation.rotation_degrees,
            fill=0,  # 旋转空白区用黑色填充
        ),
        transforms.ColorJitter(
            brightness=config.augmentation.brightness,
            contrast=config.augmentation.contrast,
        ),
        transforms.ToTensor(),                                          # [0,255] → [0,1]
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),     # ImageNet 分布
    ])


def _build_eval_transform(config: Config) -> transforms.Compose:
    """构建验证/测试/推理流水线（无随机增强，确保确定性）。
    顺序: Resize(224, BILINEAR) → ToTensor → Normalize(ImageNet)
    依据: DDS §3.4
    Args:
        config: Config 对象
    Returns:
        torchvision.transforms.Compose 对象
    """
    return transforms.Compose([
        transforms.Resize(
            (config.data.input_size, config.data.input_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ============================================================
# 数据集创建 — 依据: DDS §3.5 (create_datasets)
# ============================================================

def create_datasets(config: Config) -> Tuple[Dataset, Dataset, Dataset]:
    """加载 EuroSAT 数据集，按 7:1:2 分层划分训练/验证/测试集。
    依据: DDS §3.5, ADS I-01
    Args:
        config: Config 对象
    Returns:
        (train_dataset, val_dataset, test_dataset) 三元组
    Raises:
        FileNotFoundError: 数据根目录不存在
        ValueError: 类别文件夹不完整或为空
    """
    root = config.data.root_dir
    if not os.path.exists(root):
        raise FileNotFoundError(f"数据集根目录不存在: {root}")

    # 校验 10 个类别文件夹
    expected = list(config.data.class_names)
    actual = sorted(
        [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    )
    if actual != sorted(expected):
        missing = set(expected) - set(actual)
        extra = set(actual) - set(expected)
        msg_parts = []
        if missing:
            msg_parts.append(f"缺失: {sorted(missing)}")
        if extra:
            msg_parts.append(f"多余: {sorted(extra)}")
        raise ValueError(f"类别文件夹不匹配 | {' | '.join(msg_parts)}")

    # 检查每个类别文件夹至少含 1 张图像
    for cls in expected:
        cls_dir = os.path.join(root, cls)
        imgs = [f for f in os.listdir(cls_dir)
                if f.lower().endswith(SUPPORTED_EXTENSIONS)]
        if len(imgs) == 0:
            raise ValueError(f"类别文件夹为空: {cls}")

    # ImageFolder 加载全部数据（按字母序自动映射标签 0~9）
    full_dataset = RobustImageFolder(root=root, transform=None)

    # 校验标签映射一致性
    if full_dataset.classes != expected:
        raise ValueError(
            f"ImageFolder 类别映射不一致: {full_dataset.classes} vs {expected}"
        )

    total = len(full_dataset)
    labels = full_dataset.targets  # List[int], 长度 27,000

    # 第一次划分: train (70%) vs temp (30%)
    train_indices, temp_indices = train_test_split(
        range(total),
        train_size=config.data.train_ratio,
        stratify=labels,
        random_state=config.seed,
    )

    # 第二次划分: val (10%) vs test (20%)
    temp_labels = [labels[i] for i in temp_indices]
    val_ratio_in_temp = config.data.val_ratio / (1 - config.data.train_ratio)
    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_ratio_in_temp,
        stratify=temp_labels,
        random_state=config.seed,
    )

    # 创建带 transform 的子集
    train_dataset = SubsetWithTransform(
        full_dataset, train_indices, _build_train_transform(config)
    )
    val_dataset = SubsetWithTransform(
        full_dataset, val_indices, _build_eval_transform(config)
    )
    test_dataset = SubsetWithTransform(
        full_dataset, test_indices, _build_eval_transform(config)
    )

    # 打印分布统计
    logger.info("数据集划分完成:")
    logger.info(f"  训练集: {len(train_indices):,} 张 ({len(train_indices)/total:.0%})")
    logger.info(f"  验证集: {len(val_indices):,} 张 ({len(val_indices)/total:.0%})")
    logger.info(f"  测试集: {len(test_indices):,} 张 ({len(test_indices)/total:.0%})")
    for class_id, class_name in enumerate(config.data.class_names):
        train_n = sum(1 for i in train_indices if labels[i] == class_id)
        val_n   = sum(1 for i in val_indices if labels[i] == class_id)
        test_n  = sum(1 for i in test_indices if labels[i] == class_id)
        logger.info(f"  {class_name}: train={train_n}, val={val_n}, test={test_n}")

    # 若有损坏文件被跳过，输出统计
    if full_dataset.skipped_files:
        logger.warning(
            f"共跳过 {len(full_dataset.skipped_files)} 个损坏文件"
        )

    return train_dataset, val_dataset, test_dataset


# ============================================================
# DataLoader 创建 — 依据: DDS §3.6 (get_dataloaders)
# ============================================================

def get_dataloaders(
    config: Config, datasets: Tuple[Dataset, Dataset, Dataset]
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建训练/验证/测试 DataLoader。
    依据: DDS §3.6, ADS I-02
    Args:
        config: Config 对象
        datasets: (train_dataset, val_dataset, test_dataset) 三元组
    Returns:
        (train_loader, val_loader, test_loader) 三元组
    """
    train_ds, val_ds, test_ds = datasets

    # num_workers 保护: 不超过 CPU 核心数的一半
    cpu_count = os.cpu_count() or 1
    actual_workers = min(config.system.num_workers, max(1, cpu_count // 2))
    if actual_workers != config.system.num_workers:
        logger.warning(
            f"num_workers 从 {config.system.num_workers} 降为 {actual_workers}（CPU 核心限制）"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=actual_workers,
        pin_memory=True,
        drop_last=True,  # 丢弃最后不足一个 batch 的数据
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=actual_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=actual_workers,
        pin_memory=True,
        drop_last=False,
    )

    logger.info(
        f"DataLoader 创建完成: batch_size={config.train.batch_size}, "
        f"num_workers={actual_workers}"
    )
    return train_loader, val_loader, test_loader
