"""
模型构建与迁移学习模块
依据: DDS §4, ADS §4.2, 接口 I-03 / I-04 / I-05, SRS FR-3
职责: ResNet18 构建 → ImageNet 预训练权重加载 → 分类头替换 → 骨干冻结/解冻
      → CrossEntropyLoss 定义 → checkpoint 存取（os.replace 原子写入）
"""
import os
import logging
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torchvision

from config import Config

logger = logging.getLogger(__name__)


# ============================================================
# 模型构建 — 依据: DDS §4.2
# ============================================================

def build_model(config: Config) -> nn.Module:
    """构建 ResNet18（或 MobileNetV3-Large）分类模型。
    加载 ImageNet 预训练权重 → 替换分类头 → Kaiming 初始化新层
    → 冻结骨干（可选）→ 迁移到 GPU/CPU。
    依据: DDS §4.2
    Args:
        config: Config 对象
    Returns:
        已配置好的 nn.Module 模型对象
    Raises:
        ValueError: 模型名称不在支持列表中
    """
    model_name = config.model.name.lower()

    # 1. 选择模型构建函数
    if model_name == "resnet18":
        model_fn = torchvision.models.resnet18
    elif model_name == "mobilenet_v3_large":
        model_fn = torchvision.models.mobilenet_v3_large
    else:
        raise ValueError(
            f"不支持的模型: {model_name}，合法值: ['resnet18', 'mobilenet_v3_large']"
        )

    # 2. 加载预训练权重
    # 优先级: 本地路径 > 在线下载 > 随机初始化
    if config.model.pretrained:
        local_path = config.model.pretrained_path
        if local_path and os.path.exists(local_path):
            # 从本地文件加载
            model = model_fn(weights=None)
            state_dict = torch.load(local_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"已从本地加载预训练权重: {local_path}")
        else:
            if local_path:
                logger.warning(f"本地权重路径不存在: {local_path}，回退到在线下载")
            # 从 TorchVision 在线加载
            try:
                model = model_fn(weights=config.model.pretrained_weights)
                logger.info(f"已在线加载预训练权重: {config.model.pretrained_weights}")
            except Exception as e:
                logger.warning(f"预训练权重加载失败 ({e})，降级为 Kaiming 随机初始化")
                model = model_fn(weights=None)
    else:
        model = model_fn(weights=None)
        logger.info("使用随机初始化权重（未加载预训练）")

    # 3. 替换分类头（1000 → 10 类）
    in_features = _replace_classifier_head(model, model_name, config.data.num_classes)
    logger.info(f"分类头已替换: {in_features} → {config.data.num_classes}")

    # 4. 打印参数统计
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型总参数量: {total_params:,}")

    # 5. 冻结/解冻控制
    if config.model.freeze_backbone:
        freeze_backbone(model, model_name)
        if config.model.unfreeze_layers > 0:
            _unfreeze_last_n_blocks(model, model_name, config.model.unfreeze_layers)

    # 6. 设备迁移
    device = torch.device(
        f"cuda:{config.system.gpu_id}"
        if torch.cuda.is_available() and config.system.gpu_id >= 0
        else "cpu"
    )
    model = model.to(device)
    logger.info(f"模型已移至: {device}")

    return model


def _replace_classifier_head(model: nn.Module, model_name: str, num_classes: int) -> int:
    """替换模型的分类头，Kaiming 初始化新层权重。
    Args:
        model: 模型对象（就地修改）
        model_name: 模型名称
        num_classes: 目标类别数
    Returns:
        原分类头的输入特征维度
    """
    if model_name == "resnet18":
        in_features = model.fc.in_features  # 512
        model.fc = nn.Linear(in_features, num_classes)
        nn.init.kaiming_normal_(model.fc.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(model.fc.bias)
    elif model_name == "mobilenet_v3_large":
        # MobileNetV3 的 classifier 是一个 Sequential，最后一层是 Linear
        in_features = model.classifier[-1].in_features  # 960
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        nn.init.kaiming_normal_(model.classifier[-1].weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(model.classifier[-1].bias)
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    return in_features


# ============================================================
# 骨干冻结/解冻 — 依据: DDS §4.3 / §4.4
# ============================================================

def freeze_backbone(model: nn.Module, model_name: str) -> None:
    """冻结除分类头外的所有参数（迁移学习标准做法）。
    依据: DDS §4.3
    Args:
        model: 模型对象（就地修改）
        model_name: 模型名称，用于确定分类头参数名前缀
    """
    # 先全部设为可训练，再冻结骨干
    for p in model.parameters():
        p.requires_grad = True

    head_prefix = "fc" if model_name == "resnet18" else "classifier"

    for name, param in model.named_parameters():
        if not name.startswith(head_prefix):
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"骨干已冻结，可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)"
    )


def _unfreeze_last_n_blocks(model: nn.Module, model_name: str, n: int) -> None:
    """解冻最后 N 个残差块，用于分阶段微调。
    ResNet18 残差块命名: layer1 → layer2 → layer3 → layer4
    n=1: 解冻 layer4     n=2: 解冻 layer3+layer4
    n=3: 解冻 layer2+3+4  n=4: 解冻全部骨干
    依据: DDS §4.4
    Args:
        model: 模型对象（就地修改）
        model_name: 模型名称
        n: 解冻最后 N 个残差块数量，范围 [1, 4]
    """
    if model_name != "resnet18":
        logger.warning("解冻残差块仅支持 ResNet18 架构")
        return

    blocks = ["layer1", "layer2", "layer3", "layer4"]
    blocks_to_unfreeze = blocks[-n:]

    for name, param in model.named_parameters():
        for block_name in blocks_to_unfreeze:
            if name.startswith(block_name):
                param.requires_grad = True
                break

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"已解冻最后 {n} 个残差块 ({', '.join(blocks_to_unfreeze)})，"
                f"可训练参数: {trainable:,}")


# ============================================================
# 损失函数 — 依据: DDS §4.5
# ============================================================

def get_loss_fn() -> nn.Module:
    """返回 CrossEntropyLoss 实例。
    多分类标准损失函数，结合 LogSoftmax + NLLLoss。
    依据: DDS §4.5
    Returns:
        nn.CrossEntropyLoss 实例
    """
    return nn.CrossEntropyLoss()


# ============================================================
# checkpoint 存取 — 依据: DDS §4.6 / §4.7
# ============================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, Any],
    path: str,
    config: Config,
) -> None:
    """保存模型检查点（os.replace 原子写入，防止中断损坏）。
    依据: DDS §4.6
    Args:
        model: 模型对象
        optimizer: 优化器对象
        epoch: 当前 epoch 编号
        metrics: 指标字典（必须含 'val_acc' 和 'best_val_acc'）
        path: 保存路径
        config: Config 对象（配置快照）
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": metrics.get("best_val_acc", 0.0),
        "config": {
            "experiment_name": config.experiment_name,
            "seed": config.seed,
            "batch_size": config.train.batch_size,
            "learning_rate": config.train.learning_rate,
            "epochs": config.train.epochs,
        },
    }

    # os.replace 在 Windows 同磁盘上是原子操作
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)

    logger.info(
        f"检查点已保存: {path} "
        f"(epoch={epoch}, val_acc={metrics.get('val_acc', 'N/A')})"
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> tuple:
    """加载模型检查点，恢复模型/优化器状态和训练元信息。
    依据: DDS §4.7
    Args:
        path: 检查点文件路径
        model: 模型对象（就地恢复权重）
        optimizer: 优化器对象（可选，就地恢复状态）
    Returns:
        (epoch: int, metrics: dict) 元组
    Raises:
        FileNotFoundError: 检查点文件不存在
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"检查点不存在: {path}")

    # 先加载到 CPU，避免设备不匹配
    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            logger.warning("检查点中无 optimizer_state_dict，优化器使用初始状态")

    epoch = checkpoint.get("epoch", 0)
    best_val_acc = checkpoint.get("best_val_acc", 0.0)
    metrics = {"best_val_acc": best_val_acc}

    logger.info(f"检查点已加载: {path} (epoch={epoch}, best_val_acc={best_val_acc:.4f})")
    return epoch, metrics
