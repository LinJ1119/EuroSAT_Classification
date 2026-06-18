"""
evaluate.py 模块单元测试
依据: DDS §6, SRS FR-5
"""
import pytest
import numpy as np

from config import load_config


@pytest.fixture(scope="module")
def config():
    return load_config("configs/config.yaml")


@pytest.fixture(scope="module")
def class_names(config):
    return list(config.data.class_names)


class TestMetrics:
    """指标计算测试"""

    def test_top1_perfect(self, class_names):
        """全部预测正确时 Top-1=1.0"""
        from evaluate import _compute_metrics
        labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        preds = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        probs = np.zeros((10, 10))
        for i in range(10):
            probs[i, i] = 1.0
        m = _compute_metrics(preds, labels, probs, class_names)
        assert m["top1_acc"] == 1.0
        assert m["top2_acc"] == 1.0

    def test_top1_zero(self, class_names):
        """全部预测错误时 Top-1=0.0"""
        from evaluate import _compute_metrics
        labels = np.array([0, 0, 0])
        preds = np.array([1, 1, 1])
        probs = np.ones((3, 10)) / 10
        m = _compute_metrics(preds, labels, probs, class_names)
        assert m["top1_acc"] == 0.0

    def test_confusion_matrix_shape(self, class_names):
        """混淆矩阵形状为 (10,10)"""
        from evaluate import _compute_metrics
        labels = np.random.randint(0, 10, 100)
        preds = np.random.randint(0, 10, 100)
        probs = np.random.rand(100, 10)
        probs = probs / probs.sum(axis=1, keepdims=True)
        m = _compute_metrics(preds, labels, probs, class_names)
        cm = np.array(m["confusion_matrix"])
        assert cm.shape == (10, 10)

    def test_per_class_keys(self, class_names):
        """各类别指标包含全部 10 个类别"""
        from evaluate import _compute_metrics
        labels = np.random.randint(0, 10, 50)
        preds = np.random.randint(0, 10, 50)
        probs = np.random.rand(50, 10)
        probs = probs / probs.sum(axis=1, keepdims=True)
        m = _compute_metrics(preds, labels, probs, class_names)
        for name in class_names:
            assert name in m["per_class"]
            assert set(m["per_class"][name].keys()) == {"precision", "recall", "f1", "tp", "fp", "fn", "support"}

    def test_confused_pairs_top3(self, class_names):
        """易混淆类别对返回 Top-3"""
        from evaluate import _compute_metrics
        labels = np.random.randint(0, 10, 200)
        preds = np.random.randint(0, 10, 200)
        probs = np.random.rand(200, 10)
        probs = probs / probs.sum(axis=1, keepdims=True)
        m = _compute_metrics(preds, labels, probs, class_names)
        assert len(m["confused_pairs"]) == 3
        for cp in m["confused_pairs"]:
            assert len(cp["pair"]) == 2
            assert 0 <= cp["rate"] <= 1
