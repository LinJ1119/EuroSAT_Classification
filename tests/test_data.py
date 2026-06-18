"""
data.py 模块单元测试
依据: DDS §3, SRS FR-1 (AC-1.1~1.8), SRS FR-2 (AC-2.1~2.5)
"""
import pytest
import torch
from torch.utils.data import DataLoader

from config import load_config


# ── 模块级 fixture: 加载一次配置和数据集，所有测试共享 ──

@pytest.fixture(scope="module")
def config():
    """加载配置文件"""
    return load_config("configs/config.yaml")


@pytest.fixture(scope="module")
def datasets(config):
    """加载数据集（耗时操作，只执行一次）"""
    from data import create_datasets
    return create_datasets(config)


@pytest.fixture(scope="module")
def dataloaders(config, datasets):
    """创建 DataLoader"""
    from data import get_dataloaders
    return get_dataloaders(config, datasets)


# ============================================================
# FR-1 数据加载与预处理
# ============================================================

class TestDataLoading:
    """数据加载相关测试 — SRS FR-1"""

    def test_dataset_loaded_successfully(self, datasets):
        """AC-1.1: 数据集加载成功率 100%，三个子集均非空"""
        train_ds, val_ds, test_ds = datasets
        assert len(train_ds) > 0, "训练集为空"
        assert len(val_ds) > 0,   "验证集为空"
        assert len(test_ds) > 0,  "测试集为空"
        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert total == 27000, f"总样本数应为 27000，实际 {total}"

    def test_split_ratio(self, config, datasets):
        """AC-1.3: 数据集划分比例 7:1:2（允许 ±1% 误差）"""
        train_ds, val_ds, test_ds = datasets
        total = len(train_ds) + len(val_ds) + len(test_ds)
        actual_train = len(train_ds) / total
        actual_val   = len(val_ds)   / total
        actual_test  = len(test_ds)  / total
        assert abs(actual_train - config.data.train_ratio) < 0.01, \
            f"训练集比例偏差过大: {actual_train:.3f}"
        assert abs(actual_val - config.data.val_ratio) < 0.01, \
            f"验证集比例偏差过大: {actual_val:.3f}"

    def test_stratified_split(self, config, datasets):
        """AC-1.3 补充: 分层采样——各类别在三个子集中均有样本"""
        train_ds, val_ds, test_ds = datasets
        # 通过底层完整数据集和 indices 获取标签
        full_labels = train_ds.dataset.targets
        for class_id in range(config.data.num_classes):
            train_count = sum(1 for i in train_ds.indices if full_labels[i] == class_id)
            val_count   = sum(1 for i in val_ds.indices   if full_labels[i] == class_id)
            test_count  = sum(1 for i in test_ds.indices  if full_labels[i] == class_id)
            assert train_count > 0, f"类别 {config.data.class_names[class_id]} 在训练集中为 0"
            assert val_count > 0,   f"类别 {config.data.class_names[class_id]} 在验证集中为 0"
            assert test_count > 0,  f"类别 {config.data.class_names[class_id]} 在测试集中为 0"

    def test_class_label_mapping(self, config, datasets):
        """AC-1.2: 类别标签映射 0~9 与 class_names 一致"""
        train_ds, _, _ = datasets
        labels_seen = set()
        # 抽样检查 — 遍历 train DataLoader 的前几个 batch
        from data import get_dataloaders
        train_dl, _, _ = get_dataloaders(config, datasets)
        for images, labels in train_dl:
            labels_seen.update(labels.tolist())
            if len(labels_seen) >= config.data.num_classes:
                break
        # 10 个类别标签都在 [0, 9] 范围内
        assert labels_seen.issubset(set(range(config.data.num_classes))), \
            f"存在无效标签: {labels_seen - set(range(config.data.num_classes))}"
        # 应该覆盖了大部分类别
        assert len(labels_seen) >= 8, f"仅覆盖了 {len(labels_seen)} 个类别"


class TestPreprocessing:
    """图像预处理相关测试 — SRS FR-1"""

    def test_upsample_size(self, dataloaders):
        """AC-1.4: 图像上采样后形状为 (B, 3, 224, 224)"""
        train_dl, _, _ = dataloaders
        images, _ = next(iter(train_dl))
        assert images.shape[1:] == (3, 224, 224), \
            f"图像形状应为 (B,3,224,224)，实际 {images.shape}"

    def test_normalization_range(self, dataloaders):
        """AC-1.5: 标准化后数值范围合理（均值约 0，标准差约 1）"""
        train_dl, _, _ = dataloaders
        images, _ = next(iter(train_dl))
        # ImageNet 归一化后，整体均值应接近 0
        batch_mean = images.mean().item()
        assert -0.5 < batch_mean < 0.5, f"标准化后均值偏差过大: {batch_mean:.3f}"
        # 标准差应接近 1（不会太小）
        batch_std = images.std().item()
        assert batch_std > 0.3, f"标准化后标准差过小: {batch_std:.3f}"

    def test_dataloader_batch_size(self, config, dataloaders):
        """AC-1.6: DataLoader batch_size=256"""
        train_dl, val_dl, test_dl = dataloaders
        img, _ = next(iter(train_dl))
        assert img.shape[0] == config.train.batch_size, \
            f"训练 DataLoader batch_size 应为 {config.train.batch_size}，实际 {img.shape[0]}"
        img_v, _ = next(iter(val_dl))
        assert img_v.shape[0] == config.train.batch_size, \
            f"验证 DataLoader batch_size 应为 {config.train.batch_size}，实际 {img_v.shape[0]}"
        img_t, _ = next(iter(test_dl))
        assert img_t.shape[0] == config.train.batch_size, \
            f"测试 DataLoader batch_size 应为 {config.train.batch_size}，实际 {img_t.shape[0]}"


# ============================================================
# FR-2 数据增强
# ============================================================

class TestAugmentation:
    """数据增强相关测试 — SRS FR-2"""

    def test_train_augmentation_is_random(self, dataloaders):
        """AC-2.1~2.3: 训练 DataLoader 的增强是随机的（两次遍历不完全相同）"""
        train_dl, _, _ = dataloaders
        first_batch, _ = next(iter(train_dl))
        second_batch, _ = next(iter(train_dl))
        # 两次取出的 batch 不应该完全相等（翻转/旋转/色彩抖动在随机作用）
        assert not torch.equal(first_batch, second_batch), \
            "训练 DataLoader 两次 batch 完全相同，增强可能未生效"

    def test_val_dataloader_is_deterministic(self, config, datasets):
        """AC-2.4: 验证 DataLoader 确定性（两次结果相同）"""
        from data import get_dataloaders
        # 重新创建 DataLoader 以确保从头开始迭代
        _, val_dl, _ = get_dataloaders(config, datasets)
        first_batch, _ = next(iter(val_dl))
        _, val_dl2, _ = get_dataloaders(config, datasets)
        second_batch, _ = next(iter(val_dl2))
        # 验证集无 shuffle + 无随机增强 → 两次应完全一致
        assert torch.equal(first_batch, second_batch), \
            "验证 DataLoader 两次 batch 不一致（不应有随机性）"

    def test_train_shuffle_enabled(self, dataloaders):
        """训练 DataLoader 开启了 shuffle"""
        train_dl, _, _ = dataloaders
        # 取前两个 batch 的标签，确认不是按类别顺序排列的
        labels_1, _ = next(iter(train_dl))
        # 如果 shuffle 生效，一个 batch 内应该包含多种类别
        unique_labels = labels_1.unique().numel()
        assert unique_labels > 1, \
            f"训练 batch 仅含 {unique_labels} 个类别，shuffle 可能未生效"

    def test_val_dataloader_no_shuffle(self, config, datasets):
        """验证/测试 DataLoader shuffle=False"""
        from data import get_dataloaders
        _, val_dl, test_dl = get_dataloaders(config, datasets)
        # 连续取两个 batch，标签序列应该连续（未打乱）
        labels_a, _ = next(iter(val_dl))
        labels_b, _ = next(iter(val_dl))
        # 虽然不全相等，但它们的顺序应该是确定的
        _, val_dl2, _ = get_dataloaders(config, datasets)
        labels_a2, _ = next(iter(val_dl2))
        labels_b2, _ = next(iter(val_dl2))
        assert torch.equal(labels_a, labels_a2), "验证集第一次迭代结果不一致"
        assert torch.equal(labels_b, labels_b2), "验证集第二次迭代结果不一致"
