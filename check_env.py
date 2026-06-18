"""
EuroSAT 卫星影像分类系统 — 环境检测脚本
依据: DDS §9.4 (mode_check)
检测: Python版本 / PyTorch+CUDA+GPU / TorchVision / 数据路径 / 磁盘空间 / 依赖包
输出: 通过 / 警告 / 失败 三级报告
"""
import sys
import shutil
from pathlib import Path

# ── 终端颜色输出 ──
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

pass_count = 0
warn_count = 0
fail_count = 0


def result(status: str, label: str, detail: str = ""):
    """输出一条检测结果，更新全局计数器。
    Args:
        status: "PASS"|"WARN"|"FAIL"
        label: 检测项名称
        detail: 补充说明（可选）
    """
    global pass_count, warn_count, fail_count
    icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}[status]
    color = {"PASS": GREEN, "WARN": YELLOW, "FAIL": RED}[status]
    print(f"  {color}[{icon} {status}]{RESET} {label}")
    if detail:
        print(f"      {detail}")
    if status == "PASS":   pass_count += 1
    if status == "WARN":   warn_count += 1
    if status == "FAIL":   fail_count += 1


def check_python():
    """检测 Python 版本 == 3.8.x"""
    ver = sys.version_info
    is_38 = (ver.major == 3 and ver.minor == 8)
    actual = f"{ver.major}.{ver.minor}.{ver.micro}"
    if is_38:
        result("PASS", f"Python {actual}")
    else:
        result("WARN", f"Python {actual}（建议 3.8.x，与 PyTorch 1.12.1 兼容性最佳）")


def check_pytorch():
    """检测 PyTorch 版本、CUDA 可用性、GPU 名称和显存总量"""
    try:
        import torch
        ver = torch.__version__
        result("PASS", f"PyTorch {ver}")

        if torch.cuda.is_available():
            cuda_ver = torch.version.cuda
            gpu_name = torch.cuda.get_device_name(0)
            mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            result("PASS", f"CUDA {cuda_ver} | GPU: {gpu_name} ({mem_total:.1f} GB)")
            if mem_total < 3.5:
                result("WARN", f"GPU 显存 {mem_total:.1f} GB < 3.5 GB（训练可能受限）")
        else:
            result("WARN", "CUDA 不可用，将降级为 CPU 训练（速度较慢）")
    except ImportError:
        result("FAIL", "PyTorch 未安装")
    except Exception as e:
        result("FAIL", f"PyTorch 检测失败: {e}")


def check_torchvision():
    """检测 TorchVision 版本和 ResNet18 模型能否正常构建"""
    try:
        import torchvision
        result("PASS", f"TorchVision {torchvision.__version__}")
        try:
            _ = torchvision.models.resnet18(weights=None)
            result("PASS", "ResNet18 模型构建正常（预训练权重将在训练时加载）")
        except Exception as e:
            result("WARN", f"ResNet18 构建异常: {e}")
    except ImportError:
        result("FAIL", "TorchVision 未安装")


def check_data_path():
    """检测 EuroSAT 数据集路径和 10 个类别文件夹的完整性"""
    # 尝试从 config.py 读取路径，如果 config.py 还不存在则用默认路径
    try:
        from config import load_config
        cfg = load_config("configs/config.yaml")
        root_dir = cfg.data.root_dir
    except Exception:
        root_dir = "D:/DataDownload/EuroSat_Dataset/EuroSAT"

    data_path = Path(root_dir)
    if not data_path.exists():
        result("FAIL", f"数据路径不存在: {root_dir}")
        return

    expected = [
        "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
        "Industrial", "Pasture", "PermanentCrop", "Residential",
        "River", "SeaLake"
    ]
    class_dirs = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    missing = set(expected) - set(class_dirs)
    extra = set(class_dirs) - set(expected)

    if not missing:
        result("PASS", f"数据路径: {root_dir}")
        total_images = 0
        for cls in expected:
            cls_dir = data_path / cls
            count = len(list(cls_dir.glob("*.jpg"))) + len(list(cls_dir.glob("*.png")))
            total_images += count
            if count == 0:
                result("WARN", f"{cls}: 0 张图像")
        result("PASS", f"图像总数: {total_images:,} 张（10 个类别）")
        if total_images != 27000:
            result("WARN", f"期望 27,000 张，实际 {total_images:,} 张")
    else:
        result("FAIL", f"缺失类别文件夹: {missing}")
        if extra:
            result("WARN", f"多余文件夹: {extra}")


def check_disk_space():
    """检测项目目录和数据盘剩余磁盘空间"""
    checks = {
        "项目目录 (.)": ".",
        "D盘 (D:/)": "D:/",
    }
    for label, path in checks.items():
        try:
            usage = shutil.disk_usage(path)
            free_gb = usage.free / (1024**3)
            if free_gb >= 10:
                result("PASS", f"{label}: {free_gb:.1f} GB 可用")
            elif free_gb >= 1:
                result("WARN", f"{label}: {free_gb:.1f} GB 可用（建议 ≥ 10 GB）")
            else:
                result("FAIL", f"{label}: {free_gb:.1f} GB 可用（严重不足，请释放空间）")
        except Exception:
            pass  # 无法检测时不报错


def check_dependencies():
    """检测 Python 依赖包版本是否与 requirements.txt 一致"""
    # 格式: (导入名, 显示名, 期望版本)
    required = [
        ("numpy",       "numpy",        "1.21.6"),
        ("PIL",         "Pillow",       "9.5.0"),
        ("matplotlib",  "matplotlib",   "3.5.3"),
        ("seaborn",     "seaborn",      "0.11.2"),
        ("tqdm",        "tqdm",         "4.64.1"),
        ("yaml",        "PyYAML",       "6.0.1"),
        ("pandas",      "pandas",       "1.3.5"),
        ("sklearn",     "scikit-learn", "1.0.2"),
    ]
    for import_name, display_name, expected_ver in required:
        try:
            mod = __import__(import_name)
            actual_ver = getattr(mod, "__version__", "?")
            if actual_ver.startswith(expected_ver.rsplit(".", 1)[0]):
                result("PASS", f"{display_name} {actual_ver}")
            else:
                result("WARN", f"{display_name} {actual_ver}（期望 ~{expected_ver}）")
        except ImportError:
            result("FAIL", f"{display_name} 未安装")


def main():
    """主入口：依次执行全部 6 项环境检测，输出汇总报告"""
    print()
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  EuroSAT 卫星影像分类系统 — 环境检测{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print()

    checks = [
        ("Python 版本",            check_python),
        ("PyTorch + CUDA + GPU",   check_pytorch),
        ("TorchVision",            check_torchvision),
        ("EuroSAT 数据路径",       check_data_path),
        ("磁盘空间",               check_disk_space),
        ("Python 依赖包",          check_dependencies),
    ]

    for name, fn in checks:
        print(f"  {BOLD}── {name} ──{RESET}")
        fn()
        print()

    print(f"{BOLD}{'='*60}{RESET}")
    total = pass_count + warn_count + fail_count
    print(f"  检测结果: {GREEN}{pass_count} 通过{RESET}  "
          f"{YELLOW}{warn_count} 警告{RESET}  "
          f"{RED}{fail_count} 失败{RESET}  （共 {total} 项）")
    if fail_count > 0:
        print(f"\n  {RED}✗ {fail_count} 项检测失败，请在继续前修复。{RESET}")
        sys.exit(1)
    elif warn_count > 0:
        print(f"\n  {YELLOW}⚠ {warn_count} 项警告，建议检查但可继续。{RESET}")
    else:
        print(f"\n  {GREEN}✓ 全部检测通过，环境就绪。{RESET}")
    print()


if __name__ == "__main__":
    main()
