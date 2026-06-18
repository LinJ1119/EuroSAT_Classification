"""
model.py 模块单元测试
依据: DDS §4, SRS FR-3 (AC-3.1~3.6)
"""
import os
import tempfile
import pytest
import torch

from config import load_config


# ── 模块级 fixture ──

@pytest.fixture(scope="module")
def config():
    """加载配置文件"""
    return load_config("configs/config.yaml")


@pytest.fixture(scope="module")
def model(config):
    """构建模型（冻结模式，耗时操作只执行一次）"""
    from model import build_model
    return build_model(config)


# ============================================================
# FR-3 模型构建与迁移学习
# ============================================================

class TestModelConstruction:
    """模型构建相关测试 — SRS FR-3"""

    def test_model_built_successfully(self, model):
        """AC-3.1: 模型构建成功，参数量在合理范围"""
        total = sum(p.numel() for p in model.parameters())
        # ResNet18 约 11.7M，允许 ±0.5M 误差（不同 torchvision 版本可能有微小差异）
        assert 11_000_000 <= total <= 12_500_000, \
            f"模型参数量异常: {total:,}（期望约 11.7M）"

    def test_input_output_shape(self, model):
        """AC-3.3: 输入 (1,3,224,224) → 输出 (1,10) logits"""
        model.eval()
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (1, 10), \
            f"输出形状应为 (1,10)，实际 {output.shape}"

    def test_frozen_backbone_trainable_params(self, config, model):
        """AC-3.3: 冻结后仅分类头可训练（未解冻时约 5,130 参数）"""
        if not config.model.freeze_backbone:
            pytest.skip("当前配置未启用 freeze_backbone")
        if config.model.unfreeze_layers > 0:
            pytest.skip("当前配置已解冻部分层，与纯冻结模式断言不兼容")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        assert trainable <= 5_200, \
            f"冻结后可训练参数过多: {trainable:,}（期望 ≤ 5,200）"
        assert trainable > 0, "冻结后应有可训练参数（分类头）"
        ratio = trainable / total
        assert ratio < 0.001, \
            f"可训练参数占比过高: {ratio:.4f}（期望 < 0.1%）"

    def test_unfrozen_layer4_trainable_params(self, config, model):
        """解冻layer4时可训练参数在合理范围（~8.4M，占75%）"""
        if config.model.unfreeze_layers != 1:
            pytest.skip("当前配置未启用 unfreeze_layers=1，跳过解冻测试")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        assert 7_000_000 <= trainable <= 9_000_000, \
            f"解冻layer4后可训练参数异常: {trainable:,}（期望 7-9M）"
        ratio = trainable / total
        assert 0.6 < ratio < 0.85, \
            f"可训练参数占比异常: {ratio:.2%}（期望 60%-85%）"

    def test_output_is_logits_not_probs(self, model):
        """输出为 logits（未经 softmax），值可在正负范围"""
        model.eval()
        device = next(model.parameters()).device
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        # logits 可以有负值，而 softmax 概率全 ≥ 0
        assert output.min() < 0 or output.max() > 1, \
            "输出值范围异常，可能已经是 softmax 概率而非 logits"


class TestFreezeUnfreeze:
    """冻结/解冻控制测试"""

    def test_unfreeze_layers_increases_trainable(self):
        """解冻 layer4 后可训练参数增加"""
        import torchvision
        from model import freeze_backbone, _unfreeze_last_n_blocks
        # 独立构建模型（不依赖当前 config.yaml 的 unfreeze_layers）
        m = torchvision.models.resnet18(weights=None)
        m.fc = torch.nn.Linear(512, 10)
        freeze_backbone(m, "resnet18")
        frozen_trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        assert frozen_trainable <= 5_200
        # 解冻 layer4
        _unfreeze_last_n_blocks(m, "resnet18", n=1)
        unfrozen_trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        assert unfrozen_trainable > frozen_trainable, \
            f"解冻 layer4 后参数未增加: {frozen_trainable:,} → {unfrozen_trainable:,}"


class TestCheckpoint:
    """检查点存取测试 — SRS FR-3 AC-3.5"""

    def test_checkpoint_save_load_roundtrip(self, config, model):
        """AC-3.5: checkpoint 保存+加载往返一致"""
        from model import save_checkpoint, load_checkpoint
        import torch.optim as optim

        # 创建临时目录和优化器
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test.pth")
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-4,
            )

            # 做一次虚拟训练步，让 optimizer state 有内容
            device = next(model.parameters()).device
            dummy_input = torch.randn(4, 3, 224, 224).to(device)
            dummy_label = torch.randint(0, 10, (4,)).to(device)
            loss = torch.nn.CrossEntropyLoss()(model(dummy_input), dummy_label)
            loss.backward()
            optimizer.step()

            # 保存前记录权重
            fc_weight_before = model.fc.weight.data.clone()

            # 保存
            metrics = {"val_acc": 0.85, "best_val_acc": 0.90}
            save_checkpoint(model, optimizer, epoch=5, metrics=metrics,
                            path=ckpt_path, config=config)
            assert os.path.exists(ckpt_path), "检查点文件未生成"
            assert not os.path.exists(ckpt_path + ".tmp"), "临时文件未清理"

            # 修改权重模拟训练继续
            with torch.no_grad():
                model.fc.weight.data += 0.1

            # 加载恢复
            epoch, loaded_metrics = load_checkpoint(ckpt_path, model, optimizer)
            assert epoch == 5, f"恢复 epoch 不一致: {epoch}"
            assert loaded_metrics["best_val_acc"] == 0.90, \
                f"恢复 best_val_acc 不一致: {loaded_metrics['best_val_acc']}"

            # 验证权重恢复
            fc_weight_after = model.fc.weight.data
            assert torch.allclose(fc_weight_before, fc_weight_after, atol=1e-6), \
                "checkpoint 恢复后权重不一致"

    def test_load_nonexistent_checkpoint(self, model):
        """加载不存在的检查点应抛出 FileNotFoundError"""
        from model import load_checkpoint
        with pytest.raises(FileNotFoundError):
            load_checkpoint("nonexistent.pth", model)


class TestLossFunction:
    """损失函数测试"""

    def test_loss_is_cross_entropy(self):
        """损失函数为 CrossEntropyLoss，输出标量 ≥ 0"""
        from model import get_loss_fn
        criterion = get_loss_fn()
        logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        loss = criterion(logits, labels)
        assert loss.ndim == 0, f"Loss 应为标量，实际维度 {loss.ndim}"
        assert loss.item() >= 0, f"Loss 应 ≥ 0，实际 {loss.item():.4f}"
        assert not torch.isnan(loss), "Loss 为 NaN"
        assert not torch.isinf(loss), "Loss 为 Inf"
