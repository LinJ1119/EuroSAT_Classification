"""
端到端集成测试 (E2E)
依据: 研发指南 §25
验证: 迷你数据集 → 训练 3 epoch → 评估 → 推理 5 张 → 10分钟内完成
"""
import os
import time
import tempfile
import pytest
import torch

from config import load_config


@pytest.fixture(scope="module")
def config():
    return load_config("configs/config.yaml")


def test_e2e_mini_pipeline(config):
    """E2E 集成测试: 每类抽 5 张→训练3epoch→评估→推理5张"""
    start_time = time.time()

    # ── 1. 创建迷你数据集 (每类 5 张) ──
    from data import create_datasets
    train_ds, val_ds, test_ds = create_datasets(config)

    # 每类从测试集抽 2 张用于推理
    full_labels = test_ds.dataset.targets
    test_indices = test_ds.indices
    mini_indices = []
    class_counts = {}
    for idx in test_indices:
        cls = full_labels[idx]
        class_counts.setdefault(cls, 0)
        if class_counts[cls] < 5:
            mini_indices.append(idx)
            class_counts[cls] += 1
    assert len(mini_indices) >= 50, f"迷你测试集不足 50 张: {len(mini_indices)}"

    from torch.utils.data import DataLoader, Subset
    mini_test_ds = Subset(test_ds.dataset, mini_indices)
    mini_test_ds.transform = test_ds.transform

    # ── 2. 训练 3 epoch ──
    from model import build_model
    model = build_model(config)
    device = next(model.parameters()).device

    import torch.optim as optim
    from train import Trainer
    from utils import TensorBoardWriter
    from data import get_dataloaders

    train_loader, val_loader, _ = get_dataloaders(config, (train_ds, val_ds, test_ds))
    trainable = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable, lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    writer = TensorBoardWriter(log_dir=os.path.join(tempfile.gettempdir(), "e2e_logs"))

    trainer = Trainer(
        config, model, train_loader, val_loader,
        optimizer, None, criterion, writer, device,
    )
    trainer.config.train.epochs = 3  # 覆盖为 3 epoch
    trainer.run(start_epoch=1)
    writer.close()

    # 验证 best_model 存在
    ckpt_path = os.path.join(config.train.checkpoint_dir, "best_model.pth")
    assert os.path.exists(ckpt_path), "best_model.pth 未生成"

    # ── 3. 评估 ──
    from evaluate import _compute_metrics
    import numpy as np
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    mini_loader = DataLoader(mini_test_ds, batch_size=16, shuffle=False)
    with torch.no_grad():
        for img, lbl in mini_loader:
            out = model(img.to(device))
            probs = torch.softmax(out, dim=1)
            _, pred = torch.max(out, 1)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(lbl.numpy())
            all_probs.append(probs.cpu().numpy())

    metrics = _compute_metrics(
        np.concatenate(all_preds), np.concatenate(all_labels),
        np.concatenate(all_probs), list(config.data.class_names),
    )
    assert 0 <= metrics["top1_acc"] <= 1, f"Top-1 Acc 异常: {metrics['top1_acc']}"

    # ── 4. 推理 5 张 ──
    from predict import predict_single_image
    # 找 5 张图像文件
    import glob
    img_files = []
    for cls in config.data.class_names[:2]:
        cls_dir = os.path.join(config.data.root_dir, cls)
        img_files.extend(glob.glob(os.path.join(cls_dir, "*.jpg"))[:3])
    img_files = img_files[:5]

    for img_path in img_files:
        cls, conf, topk = predict_single_image(img_path, ckpt_path, config)
        assert isinstance(cls, str)
        assert 0 <= conf <= 1
        assert len(topk) >= 1

    elapsed = time.time() - start_time
    assert elapsed <= 600, f"E2E 测试超时: {elapsed:.0f}s (限制 600s)"
    print(f"\nE2E 测试通过，耗时 {elapsed:.0f}s")
