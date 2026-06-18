"""
CLI 统一入口模块
依据: DDS §9, ADS §4.8, SRS FR-8
职责: 命令行参数解析 → 模式路由 → 调用应用层
"""
import argparse
import logging

from config import load_config
from utils import set_seed

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="EuroSAT 卫星影像分类系统 — 基于 ResNet18 迁移学习",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
用法示例:
  python main.py --mode train
  python main.py --mode train --epochs 2            # 快速试训练
  python main.py --mode evaluate --model checkpoints/best_model.pth
  python main.py --mode predict --input test.jpg
  python main.py --mode check
        """,
    )
    parser.add_argument("--mode", type=str, required=True,
                        choices=["train", "evaluate", "predict", "check"],
                        help="运行模式")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="配置文件路径")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="数据集根目录（覆盖配置文件）")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pth",
                        help="模型权重路径")
    parser.add_argument("--input", type=str, default=None,
                        help="输入图像路径或文件夹（predict 模式）")
    parser.add_argument("--output", type=str, default=None,
                        help="结果输出目录（predict 模式）")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="批大小（覆盖配置文件）")
    parser.add_argument("--epochs", type=int, default=None,
                        help="训练轮数（覆盖配置文件）")
    parser.add_argument("--lr", type=float, default=None,
                        help="学习率（覆盖配置文件）")
    parser.add_argument("--top-k", type=int, default=None,
                        help="返回 Top-K 预测（predict 模式）")
    parser.add_argument("--resume", type=str, default=None,
                        help="从检查点恢复训练")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # CLI 参数覆盖
    cli_overrides = {}
    if args.data_dir:   cli_overrides["data.root_dir"] = args.data_dir
    if args.batch_size: cli_overrides["train.batch_size"] = args.batch_size
    if args.epochs:     cli_overrides["train.epochs"] = args.epochs
    if args.lr:         cli_overrides["train.learning_rate"] = args.lr
    if args.resume:     cli_overrides["train.resume"] = args.resume
    if args.output:     cli_overrides["inference.output_dir"] = args.output

    # 统一日志配置（所有模式都需要）
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    # 模式路由
    if args.mode == "check":
        from check_env import main as check_main
        check_main()
        return

    config = load_config(args.config, cli_overrides)
    set_seed(config.seed)

    if args.mode == "train":
        from train import run_training
        run_training(config)
    elif args.mode == "evaluate":
        from evaluate import run_evaluation
        run_evaluation(config, args.model)
    elif args.mode == "predict":
        from predict import predict_single_image, predict_batch
        if args.input:
            if args.input.lower().endswith((".jpg", ".jpeg", ".png")):
                cls, conf, topk = predict_single_image(args.input, args.model, config)
                print(f"类别: {cls}, 置信度: {conf:.4f}")
            else:
                predict_batch(args.input, args.model, config)
        else:
            print("请指定 --input（图像路径或文件夹）")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
