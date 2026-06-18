"""
模型评估模块
依据: DDS §6, ADS §4.4, 接口 I-07, SRS FR-5
职责: 测试集评估 → Top-1/2 Acc + P/R/F1 + 混淆矩阵 → Markdown 报告 + JSON + CSV + 可视化
"""
import os
import json
import logging
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from config import Config
from model import build_model, load_checkpoint
from data import create_datasets, get_dataloaders
from utils import plot_confusion_matrix, plot_class_accuracy_bar

logger = logging.getLogger(__name__)


# ============================================================
# 指标计算 — 依据: DDS §6.2
# ============================================================

def _compute_metrics(
    all_preds: np.ndarray,     # (N,) int, 预测类别
    all_labels: np.ndarray,    # (N,) int, 真实标签
    all_probs: np.ndarray,     # (N, C) float, softmax 概率
    class_names: List[str],
) -> dict:
    """计算全部评估指标。
    Args:
        all_preds: 预测类别索引数组
        all_labels: 真实标签数组
        all_probs: softmax 概率数组
        class_names: 类别名称列表
    Returns:
        指标字典
    """
    n = len(all_labels)
    n_classes = len(class_names)

    # ── 整体准确率 ──
    top1_correct = (all_preds == all_labels).sum()
    top1_acc = top1_correct / n

    top2_preds = np.argsort(all_probs, axis=1)[:, -2:]
    top2_correct = sum(all_labels[i] in top2_preds[i] for i in range(n))
    top2_acc = top2_correct / n

    # ── 各类别 Precision / Recall / F1 / TP / FP / FN ──
    per_class = {}
    for cid in range(n_classes):
        tp = int(np.sum((all_preds == cid) & (all_labels == cid)))
        fp = int(np.sum((all_preds == cid) & (all_labels != cid)))
        fn = int(np.sum((all_preds != cid) & (all_labels == cid)))
        support = int(np.sum(all_labels == cid))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[class_names[cid]] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "support": support,
        }

    # ── 宏平均 / 加权平均 ──
    macro_p = np.mean([v["precision"] for v in per_class.values()])
    macro_r = np.mean([v["recall"] for v in per_class.values()])
    macro_f1 = np.mean([v["f1"] for v in per_class.values()])

    supports = [v["support"] for v in per_class.values()]
    weighted_p = np.average([v["precision"] for v in per_class.values()], weights=supports)
    weighted_r = np.average([v["recall"] for v in per_class.values()], weights=supports)
    weighted_f1 = np.average([v["f1"] for v in per_class.values()], weights=supports)

    # ── 混淆矩阵（行归一化）──
    cm = sk_confusion_matrix(all_labels, all_preds, labels=range(n_classes))
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 某类无样本时避免除零
    cm_norm = cm.astype("float") / row_sums

    # ── Top-3 易混淆类别对 ──
    confused_pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j:
                confused_pairs.append({
                    "pair": [class_names[i], class_names[j]],
                    "rate": round(float(cm_norm[i][j]), 4),
                })
    confused_pairs.sort(key=lambda x: x["rate"], reverse=True)

    return {
        "top1_acc": round(top1_acc, 4),
        "top2_acc": round(top2_acc, 4),
        "per_class": per_class,
        "macro_avg": {"precision": round(macro_p, 4), "recall": round(macro_r, 4), "f1": round(macro_f1, 4)},
        "weighted_avg": {"precision": round(weighted_p, 4), "recall": round(weighted_r, 4), "f1": round(weighted_f1, 4)},
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "confused_pairs": confused_pairs[:3],
    }


# ============================================================
# 报告生成 — 依据: DDS §6.3
# ============================================================

def _generate_report(metrics: dict, class_names: List[str], output_dir: str) -> str:
    """生成 Markdown 格式评估报告。
    Args:
        metrics: _compute_metrics 返回的指标字典
        class_names: 类别名称列表
        output_dir: 输出目录
    Returns:
        报告文件路径
    """
    lines = []
    lines.append("# EuroSAT 卫星影像分类 — 模型评估报告\n")

    # 整体指标
    lines.append("## 一、整体指标\n")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")
    lines.append(f"| Top-1 Accuracy | {metrics['top1_acc']:.4f} |")
    lines.append(f"| Top-2 Accuracy | {metrics['top2_acc']:.4f} |")
    lines.append(f"| Macro Avg Precision | {metrics['macro_avg']['precision']:.4f} |")
    lines.append(f"| Macro Avg Recall    | {metrics['macro_avg']['recall']:.4f} |")
    lines.append(f"| Macro Avg F1        | {metrics['macro_avg']['f1']:.4f} |")
    lines.append("")

    # 各类别指标
    lines.append("## 二、各类别指标\n")
    lines.append("| 类别 | Precision | Recall | F1 | Support |")
    lines.append("|------|-----------|--------|----|---------|")
    for name in class_names:
        v = metrics["per_class"][name]
        lines.append(f"| {name} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} | {v['support']} |")
    lines.append("")

    # 易混淆类别
    lines.append("## 三、最易混淆类别对 (Top-3)\n")
    for i, cp in enumerate(metrics["confused_pairs"], 1):
        true_cls, pred_cls = cp["pair"]
        lines.append(f"{i}. **{true_cls}** → **{pred_cls}** （混淆率 {cp['rate']:.2%}）")
    lines.append("")

    # 混淆矩阵图
    lines.append("## 四、混淆矩阵\n")
    lines.append("![confusion_matrix](confusion_matrix.png)\n")

    # 类别准确率图
    lines.append("## 五、各类别准确率\n")
    lines.append("![class_accuracy](class_accuracy.png)\n")

    report_path = os.path.join(output_dir, "eval_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"评估报告已保存: {report_path}")
    return report_path


def _export_misclassified(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    class_names: List[str],
    output_dir: str,
) -> str:
    """导出误分类样本列表为 CSV。
    Args:
        all_preds: 预测类别
        all_labels: 真实标签
        all_probs: softmax 概率
        class_names: 类别名称列表
        output_dir: 输出目录
    Returns:
        CSV 文件路径
    """
    import pandas as pd

    wrong_mask = all_preds != all_labels
    wrong_indices = np.where(wrong_mask)[0]

    rows = []
    for idx in wrong_indices:
        true_id = all_labels[idx]
        pred_id = all_preds[idx]
        conf = float(all_probs[idx][pred_id])
        rows.append({
            "true_label": class_names[true_id],
            "predicted": class_names[pred_id],
            "confidence": round(conf, 4),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("confidence", ascending=False)
    csv_path = os.path.join(output_dir, "misclassified.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"误分类样本已保存: {csv_path} ({len(rows)} 张)")
    return csv_path


# ============================================================
# 评估入口 — 依据: DDS §6.4, ADS I-07
# ============================================================

def run_evaluation(config: Config, model_path: str) -> dict:
    """测试集评估入口。
    依据: DDS §6.4, ADS I-07
    Args:
        config: Config 对象
        model_path: 模型权重路径
    Returns:
        完整指标字典
    """
    # 1. 加载模型
    logger.info("加载模型...")
    model = build_model(config)
    epoch, _ = load_checkpoint(model_path, model)
    model.eval()
    device = next(model.parameters()).device
    logger.info(f"模型已加载: {model_path} (epoch={epoch})")

    # 2. 准备测试数据
    datasets = create_datasets(config)
    _, _, test_loader = get_dataloaders(config, datasets)

    # 3. 收集全部预测
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="评估中", unit="batch"):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            _, predicted = torch.max(outputs, 1)
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    # 4. 计算指标
    metrics = _compute_metrics(all_preds, all_labels, all_probs, list(config.data.class_names))

    # 5. 日志输出
    logger.info(f"评估完成: Top-1 Acc={metrics['top1_acc']:.4f}, "
                f"Top-2 Acc={metrics['top2_acc']:.4f}, "
                f"Macro F1={metrics['macro_avg']['f1']:.4f}")

    # 6. 输出目录
    output_dir = os.path.join("outputs", "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    # 7. JSON
    json_path = os.path.join(output_dir, "eval_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"JSON 已保存: {json_path}")

    # 8. Markdown 报告
    _generate_report(metrics, list(config.data.class_names), output_dir)

    # 9. 混淆矩阵图
    cm = np.array(metrics["confusion_matrix_normalized"])
    plot_confusion_matrix(cm, list(config.data.class_names),
                         os.path.join(output_dir, "confusion_matrix.png"))

    # 10. 各类别准确率柱状图
    class_accs = {
        name: metrics["per_class"][name]["recall"]
        for name in config.data.class_names
    }
    plot_class_accuracy_bar(class_accs,
                           os.path.join(output_dir, "class_accuracy.png"))

    # 11. 误分类 CSV
    _export_misclassified(all_preds, all_labels, all_probs,
                         list(config.data.class_names), output_dir)

    return metrics
