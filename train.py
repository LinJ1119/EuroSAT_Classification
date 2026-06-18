"""
模型训练模块
依据: DDS §5, ADS §4.3, 接口 I-06, SRS FR-4
职责: 训练循环编排 → 验证评估 → 早停判断 → 学习率调度 → OOM 恢复 → 检查点管理
"""
import os
import gc
import time
import logging
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import Config, save_config_snapshot
from data import create_datasets, get_dataloaders
from model import build_model, get_loss_fn, save_checkpoint, load_checkpoint
from utils import set_seed, log_gpu_memory, TensorBoardWriter

logger = logging.getLogger(__name__)


# ============================================================
# Trainer 类 — 依据: DDS §5.2
# ============================================================

class Trainer:
    """训练器：封装训练状态与逻辑。
    依据: DDS §5.2
    """

    def __init__(
        self,
        config: Config,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer: optim.Optimizer,
        scheduler,
        criterion: nn.Module,
        writer: TensorBoardWriter,
        device: torch.device,
    ):
        """初始化训练器。
        Args:
            config: Config 对象
            model: 模型对象
            train_loader: 训练 DataLoader
            val_loader: 验证 DataLoader
            optimizer: 优化器
            scheduler: 学习率调度器（ReduceLROnPlateau / CosineAnnealingLR / None）
            criterion: 损失函数
            writer: TensorBoard 封装
            device: torch device
        """
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
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
            "peak_mem_mb": [],
        }

    # ── 单 epoch 训练 ──

    def _train_one_epoch(self) -> float:
        """执行一个 epoch 的训练。
        Returns:
            该 epoch 的平均训练 loss
        """
        self.model.train()
        running_loss = 0.0
        num_batches = len(self.train_loader)

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch:03d}/{self.config.train.epochs}",
            unit="batch",
        )

        for images, labels in progress_bar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / (progress_bar.n + 1)
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

        return running_loss / num_batches

    # ── 验证 ──

    @torch.no_grad()
    def _validate(self, data_loader) -> tuple:
        """在给定 DataLoader 上评估模型。
        Args:
            data_loader: 验证/测试 DataLoader
        Returns:
            (val_loss, val_accuracy) 元组
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in data_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_loss = running_loss / len(data_loader)
        val_acc = correct / total if total > 0 else 0.0
        return val_loss, val_acc

    # ── OOM 恢复 ──

    def _handle_oom(self, error: RuntimeError) -> Optional[float]:
        """CUDA OOM 恢复：清理显存 → batch_size 减半 → 重建 DataLoader → 重试当前 epoch。
        依据: DDS §5.6
        Args:
            error: OOM 异常对象
        Returns:
            重试成功返回 train_loss，失败返回 None
        """
        logger.warning(f"CUDA OOM 触发恢复: {error}")

        # 1. 清理显存
        torch.cuda.empty_cache()
        gc.collect()

        # 2. 计算新的 batch_size（最小 32）
        current_bs = self.train_loader.batch_size
        new_bs = max(32, current_bs // 2)

        if new_bs == current_bs:
            logger.error("batch_size 已降至下限 (32)，仍 OOM，终止训练")
            return None

        logger.warning(f"batch_size: {current_bs} → {new_bs}")

        # 3. 重建 DataLoader
        ds = self.train_loader.dataset
        actual_workers = min(self.config.system.num_workers, max(1, os.cpu_count() or 1 // 2))
        self.train_loader = torch.utils.data.DataLoader(
            ds, batch_size=new_bs, shuffle=True,
            num_workers=actual_workers, pin_memory=True, drop_last=True,
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_loader.dataset, batch_size=new_bs, shuffle=False,
            num_workers=actual_workers, pin_memory=True, drop_last=False,
        )

        # 4. 重试当前 epoch
        logger.info("重试当前 epoch...")
        return self._train_one_epoch()

    # ── 主循环 ──

    def run(self, start_epoch: int = 1) -> None:
        """执行完整训练主循环。
        依据: DDS §5.5
        Args:
            start_epoch: 起始 epoch 编号（断点续训时 > 1）
        """
        logger.info(f"开始训练: epochs={self.config.train.epochs}, "
                     f"batch_size={self.config.train.batch_size}, "
                     f"lr={self.config.train.learning_rate}")

        for epoch in range(start_epoch, self.config.train.epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

            # ── 训练 ──
            try:
                train_loss = self._train_one_epoch()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    train_loss = self._handle_oom(e)
                    if train_loss is None:
                        break
                else:
                    raise

            # ── 验证 ──
            val_loss, val_acc = self._validate(self.val_loader)

            # ── 学习率调度 ──
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # ── 显存监控 ──
            peak_mem = log_gpu_memory(self.device) * 1024  # GB → MB

            # ── 记录历史 ──
            epoch_time = time.time() - epoch_start
            self.train_history["train_loss"].append(train_loss)
            self.train_history["val_loss"].append(val_loss)
            self.train_history["val_acc"].append(val_acc)
            self.train_history["lr"].append(current_lr)
            self.train_history["peak_mem_mb"].append(peak_mem)

            # ── 日志输出 ──
            logger.info(
                f"Epoch {epoch:03d}/{self.config.train.epochs} — "
                f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, "
                f"val_acc: {val_acc:.4f}, lr: {current_lr:.2e}, "
                f"peak_mem: {peak_mem:.1f} MB, time: {epoch_time:.1f}s"
            )

            # ── TensorBoard ──
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/val", val_acc, epoch)
            self.writer.add_scalar("LR", current_lr, epoch)
            self.writer.add_scalar("Memory/peak_mb", peak_mem, epoch)

            # ── 检查点保存 ──
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                save_checkpoint(
                    self.model, self.optimizer, epoch,
                    {"val_acc": val_acc, "best_val_acc": self.best_val_acc},
                    os.path.join(self.config.train.checkpoint_dir, "best_model.pth"),
                    self.config,
                )
                logger.info(f"  ✓ 新的最佳模型 (val_acc={val_acc:.4f})")
            else:
                self.epochs_no_improve += 1

            # 定期检查点
            if not self.config.train.save_best_only:
                if epoch % self.config.train.save_interval == 0:
                    save_checkpoint(
                        self.model, self.optimizer, epoch,
                        {"val_acc": val_acc, "best_val_acc": self.best_val_acc},
                        os.path.join(self.config.train.checkpoint_dir,
                                     f"checkpoint_epoch_{epoch:03d}.pth"),
                        self.config,
                    )

            # ── 早停检查 ──
            if self.epochs_no_improve >= self.config.train.early_stop_patience:
                logger.info(
                    f"早停触发: val_loss 已 {self.epochs_no_improve} 轮无改善 "
                    f"(min_delta={self.config.train.early_stop_min_delta})"
                )
                logger.info(f"最佳模型: epoch {self.best_epoch}, val_acc={self.best_val_acc:.4f}")
                break

        self.writer.close()
        logger.info(f"训练完成。最佳 val_acc={self.best_val_acc:.4f} (epoch {self.best_epoch})")


# ============================================================
# 训练入口 — 依据: DDS §5.8
# ============================================================

def run_training(config: Config) -> None:
    """训练入口函数：编排全部训练步骤。
    依据: DDS §5.8
    Args:
        config: Config 对象
    """
    # 1. 设置随机种子
    set_seed(config.seed)

    # 2. 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.system.output_dir, f"train_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(config.train.checkpoint_dir, exist_ok=True)
    os.makedirs(config.train.log_dir, exist_ok=True)

    # 3. 保存配置快照
    save_config_snapshot(config, output_dir)

    # 4. 准备数据
    logger.info("准备数据集...")
    datasets = create_datasets(config)
    train_loader, val_loader, _ = get_dataloaders(config, datasets)

    # 5. 构建模型
    logger.info("构建模型...")
    model = build_model(config)
    device = next(model.parameters()).device

    # 6. 优化器（仅优化可训练参数）
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(
        trainable_params,
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )

    # 7. 损失函数
    criterion = get_loss_fn()

    # 8. 学习率调度器
    if config.train.lr_scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            patience=config.train.lr_patience,
            factor=config.train.lr_factor,
        )
    elif config.train.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.train.epochs,
        )
    else:
        scheduler = None

    # 9. TensorBoard
    writer = TensorBoardWriter(log_dir=config.train.log_dir)

    # 10. 创建 Trainer
    trainer = Trainer(
        config, model, train_loader, val_loader,
        optimizer, scheduler, criterion, writer, device,
    )

    # 11. 断点续训
    start_epoch = 1
    if config.train.resume:
        # 如果解冻了更多层，可训练参数数量变了，optimizer state 无法复用
        skip_optim = config.model.unfreeze_layers > 0
        epoch, metrics = load_checkpoint(
            config.train.resume, model,
            optimizer=None if skip_optim else optimizer
        )
        trainer.current_epoch = epoch
        trainer.best_val_acc = metrics["best_val_acc"]
        start_epoch = epoch + 1
        if skip_optim:
            logger.info(f"从检查点恢复(仅权重): epoch {epoch}, "
                        f"best_val_acc={metrics['best_val_acc']:.4f}, "
                        f"optimizer 已重置（解冻层数变化）")
        else:
            logger.info(f"从检查点恢复: epoch {epoch}, "
                        f"best_val_acc={metrics['best_val_acc']:.4f}")

    # 12. 显存限制
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(config.system.gpu_memory_fraction)
        torch.cuda.empty_cache()
        logger.info(f"显存限制已设置: {config.system.gpu_memory_fraction*100:.0f}%")

    # 13. 启动训练
    trainer.run(start_epoch=start_epoch)
