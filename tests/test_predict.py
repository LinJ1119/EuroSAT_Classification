"""
predict.py 模块单元测试
依据: DDS §7, SRS FR-6
"""
import os
import pytest
import torch
import numpy as np
from PIL import Image

from config import load_config


@pytest.fixture(scope="module")
def config():
    return load_config("configs/config.yaml")


class TestPreprocessing:
    """图像预处理测试"""

    def test_rgb_image(self, config, tmp_path):
        """RGB 图像正常预处理"""
        from predict import _load_and_preprocess
        img = Image.new("RGB", (64, 64), color=(100, 150, 200))
        img_path = tmp_path / "test_rgb.jpg"
        img.save(img_path)

        device = torch.device("cpu")
        tensor = _load_and_preprocess(str(img_path), config, device)
        assert tensor.shape == (1, 3, 224, 224)

    def test_grayscale_to_rgb(self, config, tmp_path):
        """灰度图自动转换为 3 通道"""
        from predict import _load_and_preprocess
        img = Image.new("L", (64, 64), color=128)
        img_path = tmp_path / "test_gray.jpg"
        img.save(img_path)

        device = torch.device("cpu")
        tensor = _load_and_preprocess(str(img_path), config, device)
        assert tensor.shape == (1, 3, 224, 224)

    def test_rgba_to_rgb(self, config, tmp_path):
        """RGBA 自动丢弃 Alpha 通道"""
        from predict import _load_and_preprocess
        img = Image.new("RGBA", (64, 64), color=(100, 150, 200, 255))
        img_path = tmp_path / "test_rgba.png"
        img.save(img_path)

        device = torch.device("cpu")
        tensor = _load_and_preprocess(str(img_path), config, device)
        assert tensor.shape == (1, 3, 224, 224)

    def test_corrupted_image_raises(self, config, tmp_path):
        """损坏/非图像文件抛出 ValueError"""
        from predict import _load_and_preprocess
        fake_path = tmp_path / "fake.jpg"
        fake_path.write_text("not an image")

        device = torch.device("cpu")
        with pytest.raises(ValueError):
            _load_and_preprocess(str(fake_path), config, device)
