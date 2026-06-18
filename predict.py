"""
推理与预测模块
依据: DDS §7, ADS §4.5, 接口 I-08 / I-09, SRS FR-6
职责: 单张/批量分类预测 → Top-K 置信度输出 → 通道自适应 → CSV 汇总
"""
import os
import logging
from pathlib import Path
from typing import Tuple

import torch
import pandas as pd
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from tqdm import tqdm

from config import Config
from model import build_model, load_checkpoint
from data import IMAGENET_MEAN, IMAGENET_STD, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


# ============================================================
# 图像预处理 — 依据: DDS §7.2
# ============================================================

def _load_and_preprocess(
    image_path: str, config: Config, device: torch.device
) -> torch.Tensor:
    """加载图像 → 通道自适应 → 上采样 → 标准化 → Tensor。
    依据: DDS §7.2
    Args:
        image_path: 图像文件路径
        config: Config 对象
        device: torch device
    Returns:
        预处理后的 Tensor，形状 (1, 3, 224, 224)
    Raises:
        ValueError: 图像加载失败或不支持的格式
    """
    # 1. 加载图像
    try:
        image = Image.open(image_path)
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError(f"图像加载失败: {image_path} ({e})")

    # 2. 通道自适应
    mode = image.mode
    if mode == "L":           # 灰度图
        image = image.convert("RGB")
        logger.info(f"灰度图已转换为 RGB: {os.path.basename(image_path)}")
    elif mode == "RGBA":      # RGBA
        image = image.convert("RGB")
        logger.info(f"RGBA 已丢弃 Alpha 通道: {os.path.basename(image_path)}")
    elif mode == "CMYK":      # CMYK
        image = image.convert("RGB")
        logger.info(f"CMYK 已转换为 RGB: {os.path.basename(image_path)}")
    elif mode != "RGB":
        raise ValueError(f"不支持的图像通道模式: {mode} ({image_path})")

    # 3. 预处理管线（与验证集一致：无增强）
    preprocess = transforms.Compose([
        transforms.Resize(
            (config.data.input_size, config.data.input_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = preprocess(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)
    return tensor


# ============================================================
# 单张推理 — 依据: DDS §7.3, ADS I-08
# ============================================================

def predict_single_image(
    image_path: str,
    model_path: str,
    config: Config,
) -> Tuple[str, float, list]:
    """对单张卫星影像进行分类预测。
    依据: DDS §7.3, ADS I-08
    Args:
        image_path: 图像文件路径
        model_path: 模型权重路径
        config: Config 对象
    Returns:
        (predicted_class: str, confidence: float, top_k: [(类名, 置信度), ...])
    """
    # 校验
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像不存在: {image_path}")

    # 加载模型
    model = build_model(config)
    load_checkpoint(model_path, model)
    model.eval()
    device = next(model.parameters()).device

    # 预处理
    image_tensor = _load_and_preprocess(image_path, config, device)

    # 推理
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        top_k_probs, top_k_indices = torch.topk(probs, config.inference.top_k, dim=1)

    top_k_probs = top_k_probs.squeeze(0).cpu().numpy()
    top_k_indices = top_k_indices.squeeze(0).cpu().numpy()

    # 组装结果
    predicted_class = config.data.class_names[int(top_k_indices[0])]
    confidence = round(float(top_k_probs[0]), 4)
    top_k_list = [
        (config.data.class_names[int(idx)], round(float(prob), 4))
        for idx, prob in zip(top_k_indices, top_k_probs)
    ]

    # 低置信度告警
    if confidence < config.inference.conf_warning_threshold:
        logger.warning(
            f"低置信度预测: {predicted_class} ({confidence:.4f}) — 可能为域外图像"
        )

    return predicted_class, confidence, top_k_list


# ============================================================
# 批量推理 — 依据: DDS §7.4, ADS I-09
# ============================================================

def predict_batch(
    input_dir: str,
    model_path: str,
    config: Config,
) -> str:
    """对文件夹内所有影像进行批量分类预测。
    依据: DDS §7.4, ADS I-09
    Args:
        input_dir: 图像文件夹路径
        model_path: 模型权重路径
        config: Config 对象
    Returns:
        CSV 文件路径
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    # 收集图像文件
    image_files = []
    for ext in SUPPORTED_EXTENSIONS:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    image_files = sorted(set(image_files))

    if len(image_files) == 0:
        logger.info(f"输入目录无有效图像: {input_dir}")
        return ""

    # 加载模型（仅一次）
    model = build_model(config)
    load_checkpoint(model_path, model)
    model.eval()
    device = next(model.parameters()).device

    # 批量推理
    results = []
    skipped = []
    success_count = 0

    for img_path in tqdm(image_files, desc="批量推理", unit="张"):
        try:
            tensor = _load_and_preprocess(str(img_path), config, device)
            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)
                top_k_probs, top_k_indices = torch.topk(probs, config.inference.top_k, dim=1)

            top_k_probs = top_k_probs.squeeze(0).cpu().numpy()
            top_k_indices = top_k_indices.squeeze(0).cpu().numpy()

            result = {
                "image_path": str(img_path),
                "predicted_class": config.data.class_names[int(top_k_indices[0])],
                "confidence": round(float(top_k_probs[0]), 4),
            }
            for k in range(config.inference.top_k):
                result[f"top{k+1}_class"] = config.data.class_names[int(top_k_indices[k])]
                result[f"top{k+1}_confidence"] = round(float(top_k_probs[k]), 4)

            results.append(result)
            success_count += 1

        except Exception as e:
            logger.warning(f"跳过: {img_path} ({e})")
            skipped.append({"file": str(img_path), "error": str(e)})

    # 保存结果
    output_dir = Path(config.inference.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    csv_path = output_dir / "predictions.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    if skipped:
        skip_path = output_dir / "skipped_files.txt"
        with open(skip_path, "w", encoding="utf-8") as f:
            for s in skipped:
                f.write(f"{s['file']}\t{s['error']}\n")

    # 汇总统计
    total = len(image_files)
    class_dist = df["predicted_class"].value_counts()
    logger.info(f"批量推理完成: 成功 {success_count}/{total}, 失败 {len(skipped)}/{total}")
    for name in config.data.class_names:
        count = class_dist.get(name, 0)
        pct = count / success_count * 100 if success_count > 0 else 0
        logger.info(f"  {name}: {count} ({pct:.1f}%)")

    return str(csv_path)
