"""
工具函数模块
依据: DDS §8, ADS §4.7
职责: 随机种子设置、GPU 显存监控、训练曲线绘制、混淆矩阵热力图、
      TensorBoard 日志封装、tqdm 进度条（在 train.py 中直接调用）
"""
import os
import random
import logging
from typing import List, Dict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # 非交互式后端

# ── 中文字体支持 ──
import matplotlib.pyplot as plt
try:
    # Windows 系统优先使用 SimHei（黑体）
    matplotlib.font_manager.fontManager.addfont(
        "C:/Windows/Fonts/simhei.ttf"
    )
    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
except Exception:
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方块的问题

import seaborn as sns
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


# ============================================================
# 随机种子
# ============================================================

def set_seed(seed: int = 42) -> None:
    """设置全局随机种子，确保实验可复现。
    同时设置 random、numpy、torch、torch.cuda 四位一体种子，
    并启用 cudnn 确定性模式（轻微性能代价）。
    依据: DDS §8.2
    Args:
        seed: 随机种子值，默认 42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 确保 GPU 上卷积算子的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"全局随机种子已设置: {seed}")


# ============================================================
# GPU 显存监控
# ============================================================

def log_gpu_memory(device: torch.device) -> float:
    """读取当前进程的 GPU 峰值显存，重置计数器，返回 GB 值。
    超过 2.5 GB 时输出 WARNING 日志。
    依据: DDS §8.3
    Args:
        device: torch device 对象（cpu 或 cuda）
    Returns:
        峰值显存量（GB），保留 4 位小数。CPU 设备返回 0.0
    """
    if device.type == "cpu":
        return 0.0

    peak_bytes = torch.cuda.max_memory_allocated(device)
    peak_gb = peak_bytes / (1024 ** 3)
    torch.cuda.reset_peak_memory_stats(device)

    if peak_gb > 2.5:
        logger.warning(f"峰值显存偏高: {peak_gb:.2f} GB")
    else:
        logger.info(f"峰值显存: {peak_gb:.2f} GB")

    return round(peak_gb, 4)


# ============================================================
# 训练曲线图
# ============================================================

def plot_training_curves(history: Dict[str, List[float]], output_dir: str) -> None:
    """生成训练曲线图（Loss 双线 + Accuracy 单线），保存为 PNG。
    依据: DDS §8.4
    Args:
        history: 训练历史字典，必须包含:
            'train_loss': List[float]  每 epoch 训练损失
            'val_loss':   List[float]  每 epoch 验证损失
            'val_acc':    List[float]  每 epoch 验证准确率
        output_dir: 图表输出目录
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    dpi = 200

    # ── Loss 曲线（训练 + 验证双线）──
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_loss"], "b-", label="训练 Loss", linewidth=1.5)
    ax.plot(epochs, history["val_loss"], "r-", label="验证 Loss", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("训练和验证 Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=dpi)
    plt.close()
    logger.info(f"Loss 曲线已保存: {output_dir}/loss_curve.png")

    # ── Accuracy 曲线（标注最佳点）──
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["val_acc"], "g-", label="验证 Accuracy", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("验证集 Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 标注最佳 epoch
    if len(history["val_acc"]) > 0:
        best_idx = int(np.argmax(history["val_acc"]))
        best_acc = history["val_acc"][best_idx]
        ax.annotate(
            f"最佳: Epoch {best_idx + 1}\nAcc={best_acc:.4f}",
            xy=(best_idx + 1, best_acc),
            xytext=(best_idx + 3, best_acc - 0.03),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=10, color="red",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"), dpi=dpi)
    plt.close()
    logger.info(f"Accuracy 曲线已保存: {output_dir}/accuracy_curve.png")


# ============================================================
# 混淆矩阵热力图
# ============================================================

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str,
    dpi: int = 200,
) -> None:
    """使用 Seaborn 绘制归一化混淆矩阵热力图。
    依据: DDS §8.5
    Args:
        cm: 归一化混淆矩阵，shape (N, N)，值范围 [0, 1]
        class_names: 类别名称列表，长度 N
        output_path: 输出 PNG 文件路径
        dpi: 图像分辨率，默认 200
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,                  # 格子内显示数值
        fmt=".2f",                   # 保留 2 位小数
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "比例"},
    )
    plt.xlabel("预测类别", fontsize=12)
    plt.ylabel("真实类别", fontsize=12)
    plt.title("混淆矩阵（行归一化）", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    logger.info(f"混淆矩阵图已保存: {output_path}")


# ============================================================
# 各类别准确率柱状图
# ============================================================

def plot_class_accuracy_bar(
    class_accs: Dict[str, float],
    output_path: str,
    dpi: int = 200,
) -> None:
    """绘制各类别准确率水平柱状图。
    绿色 = 高于平均值，红色 = 低于平均值。
    依据: DDS §8.6
    Args:
        class_accs: 字典，{类别名: 准确率(float)}
        output_path: 输出 PNG 文件路径
        dpi: 图像分辨率，默认 200
    """
    # 按准确率降序排列
    sorted_items = sorted(class_accs.items(), key=lambda x: x[1], reverse=True)
    names = [item[0] for item in sorted_items]
    accs = [item[1] for item in sorted_items]

    mean_acc = np.mean(accs)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ecc71" if acc >= mean_acc else "#e74c3c" for acc in accs]
    bars = ax.barh(names, accs, color=colors)

    # 平均值参考线
    ax.axvline(x=mean_acc, color="gray", linestyle="--", linewidth=1,
               label=f"平均值: {mean_acc:.3f}")
    ax.set_xlabel("Accuracy (Recall)")
    ax.set_title("各类别准确率")
    ax.legend()
    ax.set_xlim(0, 1.05)

    # 柱上标注数值
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.3f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    logger.info(f"类别准确率图已保存: {output_path}")


# ============================================================
# TensorBoard 日志封装
# ============================================================

class TensorBoardWriter:
    """TensorBoard SummaryWriter 的轻量封装。
    提供 add_scalar / add_scalars / add_figure 便捷方法。
    依据: DDS §8
    """

    def __init__(self, log_dir: str):
        """初始化 TensorBoard writer。
        Args:
            log_dir: 事件文件输出目录
        """
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"TensorBoard 日志目录: {log_dir}")

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        """记录单个标量（如 loss、accuracy、lr）。
        Args:
            tag: 指标名称，如 'Loss/train'、'Accuracy/val'
            value: 标量值
            step: 步数（通常为 epoch 编号）
        """
        self.writer.add_scalar(tag, value, step)

    def add_scalars(self, main_tag: str, tag_value_dict: Dict[str, float], step: int) -> None:
        """在同一张图上记录多个标量（如 train loss + val loss）。
        Args:
            main_tag: 主标签，如 'Loss'
            tag_value_dict: 子标签到值的映射，如 {'train': 0.5, 'val': 0.6}
            step: 步数
        """
        self.writer.add_scalars(main_tag, tag_value_dict, step)

    def add_figure(self, tag: str, figure, step: int) -> None:
        """将 Matplotlib 图像写入 TensorBoard。
        Args:
            tag: 图像标签
            figure: Matplotlib Figure 对象
            step: 步数
        """
        self.writer.add_figure(tag, figure, step)

    def close(self) -> None:
        """关闭 writer，释放资源"""
        self.writer.close()
        logger.info("TensorBoard writer 已关闭")
