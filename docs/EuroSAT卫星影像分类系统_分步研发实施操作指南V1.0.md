# EuroSAT 卫星影像分类系统 — 分步研发实施操作指南

> **文档编号**：IMPL-GUIDE-EuroSAT-001  
> **版本**：V1.0  
> **编制日期**：2026-06-17  
> **编制依据**：`docs/` 文件夹下全部设计文档（SRS/ADS/DDS/技术选型/研发工作流程）  
> **项目代号**：EuroSAT_Classification  
> **用途**：本指南将项目研发分解为可逐步骤执行的具体操作，AI 可参照本指南分步执行，人工在检核点（⭐）介入确认。

---

## 使用说明

1. **执行顺序**：按步骤编号从 1 到 37 顺序执行，不可跳步
2. **执行人标注**：👤 = 你来做 | 🤖 = AI 来做 | 👤+🤖 = 你发需求，AI 执行，你验收
3. **⭐ 标记** = 人工检核点（必须人工确认通过才能进入下一步）
4. **🟢 标记** = 关键里程碑
5. **AI 执行约束**：AI 生成代码必须依据 `docs/` 中对应的设计文档，不得自行发挥。引用格式：`// 依据: DDS §X.X`
6. **需求文档映射**：每个步骤标注其对应的 SRS FR 编号和 DDS 章节

---

## 准备工作（一次性）

| 检查项 | 状态 |
|------|:---:|
| EuroSAT 数据集已下载到 `D:\DataDownload\EuroSat_Dataset\EuroSAT`，含 10 个类别子文件夹 | ☐ |
| 项目根目录 `d:\myproject\EuroSat_images_classification` 已创建 | ☐ |
| `docs/` 文件夹含 6 份设计文档且已通读 | ☐ |
| AI 编程工具可用（Claude Code） | ☐ |
| Git 已安装并配置（含 GitHub 远程仓库） | ☐ |
| Miniconda/Anaconda 已安装 | ☐ |

---

## 阶段一：环境搭建与核心工具（步骤 1-6，预计 0.5 天）

### 步骤 1：创建项目目录结构

**执行人**：👤

**做什么**：
在 `d:\myproject\EuroSat_images_classification` 下创建扁平化目录结构（本项目的 8 个 `.py` 文件放在根目录）：

```
d:\myproject\EuroSat_images_classification\
├── docs/               # [已有] 设计文档
├── configs/            # 配置文件目录
├── checkpoints/        # [运行时生成] 模型检查点
├── logs/               # [运行时生成] 训练日志 + TensorBoard
├── runs/               # [运行时生成] 训练输出
├── outputs/            # [运行时生成] 评估/推理结果
│   ├── evaluation/
│   └── predictions/
└── tests/              # 测试文件目录
```

**操作**：

```bash
# === Git Bash（推荐）=== 
cd d:/myproject/EuroSat_images_classification
mkdir -p configs checkpoints logs runs outputs/evaluation outputs/predictions tests
```

```powershell
# === PowerShell（备选）===
cd d:\myproject\EuroSat_images_classification
'configs','checkpoints','logs','runs','outputs\evaluation','outputs\predictions','tests' | ForEach-Object { New-Item -ItemType Directory -Path $_ -Force }
```

**✅ 验收**：6 个目录（docs/configs/tests 已存在 + checkpoints/logs/runs/outputs 新创建）均存在。

---

### 步骤 2：创建虚拟环境并安装 PyTorch

**执行人**：👤

**做什么**：

```bash
# 1. 创建 Python 3.8 虚拟环境
conda create -n eurosat python=3.8 -y
conda activate eurosat

# 2. 安装 PyTorch（在线安装 — 从官方 wheel 源下载）
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
  -f https://download.pytorch.org/whl/torch_stable.html

# 或者：本地安装（如果已提前下载 .whl 文件到 D:CodingStudywheels_download）
	cd D:/CodingStudy/wheels_download
	pip install torch-1.12.1+cu113-cp38-cp38-win_amd64.whl
	pip install torchvision-0.13.1+cu113-cp38-cp38-win_amd64.whl
	cd d:/myproject/EuroSat_images_classification   # 回到项目目录

# 3. 验证 GPU 可用
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

**预期输出**：最后一行输出 `GeForce GTX 1050 Ti`。

**常见问题**：
- `conda create` 网络慢 → 换清华镜像源：`conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/`
- `python=3.8.20` 精确版本不存在 → 用 `python=3.8`，conda 自动选 3.8 系最新版（micro 版本号不关键）

**✅ 验收**：`torch.cuda.is_available()` 返回 `True`，GPU 名称正确。

---

### 步骤 3：安装其余 Python 依赖

**执行人**：👤+🤖

**做什么**：让 AI 依据 `docs/卫星影像分类技术选型方案V1.0.md` 第 10 章选型汇总和第 5.2 节依赖包清单，生成 `requirements.txt` 并保存到项目根目录。

**操作**：
```bash
# 确认当前在项目根目录
cd d:/myproject/EuroSat_images_classification

# 安装依赖
pip install -r requirements.txt
```

**`requirements.txt` 内容参考**（AI 协助生成精确版本）：
```
torch==1.12.1+cu113
torchvision==0.13.1+cu113
numpy==1.21.6
Pillow==9.5.0
matplotlib==3.5.3
seaborn==0.11.2
tqdm==4.64.1
tensorboard==2.9.1
pyyaml==6.0.1
pandas==1.3.5
scikit-learn==1.0.2
```

**✅ 验收**：`pip list` 可见上述包，版本号匹配。

---

### 步骤 4：安装 Git 并初始化仓库

**执行人**：👤

**做什么**：

```bash
cd d:/myproject/EuroSat_images_classification

# 初始化 Git 仓库
git init

# 创建 .gitignore（让 AI 生成，内容参考下方）
```

**`.gitignore` 内容**（让 AI 生成）：
```gitignore
# 运行时生成
checkpoints/
logs/
runs/
outputs/
*.pth
*.tmp

# Python
__pycache__/
*.pyc
*.pyo
.ipynb_checkpoints/

# 虚拟环境
venv/
.venv/
*.egg-info/

# IDE
.vscode/
.idea/

# 系统
.DS_Store
Thumbs.db
```

**✅ 验收**：`git status` 正常显示未跟踪文件，`checkpoints/` 等目录被忽略。

---

### 步骤 5：编写 check_env.py 环境检测脚本

**执行人**：👤+🤖

**依据文档**：DDS §9.4 `mode_check()` 函数设计、技术选型 §2.2 硬件约束

**做什么**：将以下需求发给 AI，让它生成 `check_env.py`（保存到项目根目录）：

> 依据 DDS §9.4，编写 `check_env.py`，一键检测以下 6 项，输出"通过/警告/失败"三级报告：
> 1. Python 版本 == 3.8.x
> 2. PyTorch 版本 + CUDA 可用性 + GPU 名称 + 显存总量
> 3. TorchVision 版本 + 预训练权重可访问性（试加载 resnet18）
> 4. 数据路径 `D:\DataDownload\EuroSat_Dataset\EuroSAT` 存在 + 10 个类别文件夹完整 + 各类别图像数量统计
> 5. 项目目录和磁盘剩余空间（≥ 10 GB 通过，1-10 GB 警告，< 1 GB 失败）
> 6. 依赖包版本是否匹配 requirements.txt
>
> 参考：DDS §9.4 有完整的检查项伪代码。

AI 生成代码后，运行验证：
```bash
python check_env.py
```

**✅ 验收**：全部检测项输出"通过"，或仅磁盘空间为"警告"（不阻塞）。

---

### ⭐ 步骤 6：环境检测通过确认（人工检核点 CP-1）

**执行人**：👤（必须人工确认）

**检核内容**：
1. `check_env.py` 全部输出"通过"
2. 数据路径下 10 个类别文件夹齐全，图像总数 ≈ 27,000
3. GPU 显存 ≥ 3.5 GB（GTX 1050 Ti 为 4 GB）

**通过后记录**：日期、检核人、验证方法、结果

**✅ 验收**：CP-1 通过，阶段一完成。进入阶段二。

---

## 阶段二：配置与数据模块（步骤 7-10，预计 0.5-1 天）

### 步骤 7：编写 config.py 配置管理模块

**执行人**：👤+🤖

**依据文档**：DDS §2（config.py 完整设计，含 7 个 frozen dataclass 字段定义 + 3 个核心函数伪代码）、ADS §4.6 及 I-10 接口契约

**做什么**：将 DDS §2 的完整设计发给 AI，生成 `config.py`。关键要点：

1. **7 个 frozen dataclass**：SystemConfig / DataConfig / AugmentationConfig / ModelConfig / TrainConfig / InferenceConfig / VisualizationConfig（字段和默认值严格按 DDS §2.2 定义）
2. **顶层 Config dataclass**：聚合上述 7 个，含 `experiment_name` 和 `seed`
3. **`load_config(path, cli_args=None) -> Config`**：YAML → dict → 递归构造 dataclass → CLI 参数覆盖 → `_validate()` → 返回
4. **`_validate(config)`**：9 条校验规则（按 DDS §2.3.2）
5. **`save_config_snapshot(cfg, output_dir) -> str`**：Config → dict → YAML 文件（含时间戳）

**`configs/config.yaml` 文件**：同时生成配置文件模板，关键字段：
- `data.root_dir: "D:/DataDownload/EuroSat_Dataset/EuroSAT"` ← **必须与实际路径一致**
- `data.num_classes: 10`
- `model.name: "resnet18"`, `model.pretrained: true`, `model.freeze_backbone: true`
- `train.batch_size: 256`, `train.epochs: 50`, `train.learning_rate: 0.0001`
- `system.gpu_memory_fraction: 0.75`

**✅ 验收**：
```bash
python -c "from config import load_config; c = load_config('configs/config.yaml'); print(f'数据路径: {c.data.root_dir}'); print(f'类别数: {c.data.num_classes}')"
```
正常输出，无报错。

---

### 步骤 8：编写 utils.py 工具函数模块

**执行人**：👤+🤖

**依据文档**：DDS §8（utils.py 完整设计，含 6 个核心函数伪代码）、ADS §4.7

**做什么**：将 DDS §8 的完整设计发给 AI，生成 `utils.py`。关键函数：

1. **`set_seed(seed=42)`**：设置 random/numpy/torch/torch.cuda 四位一体随机种子 + `cudnn.deterministic=True`
2. **`log_gpu_memory(device) -> float`**：读取峰值显存 → 重置计数器 → 返回 GB 值 → 超 2.5 GB 时 WARNING
3. **`plot_training_curves(history, output_dir)`**：Loss 曲线（train+val 双线）+ Accuracy 曲线（含最佳点标注），≥ 200 DPI
4. **`plot_confusion_matrix(cm, class_names, output_path, dpi=200)`**：Seaborn heatmap 10×10 归一化矩阵
5. **`plot_class_accuracy_bar(class_accs, output_path, dpi=200)`**：各类别准确率水平柱状图，红绿分色
6. **`TensorBoardWriter` 类**：封装 `SummaryWriter`，提供 `add_scalar` / `add_scalars` / `add_figure`

**✅ 验收**：
```bash
python -c "from utils import set_seed, log_gpu_memory; set_seed(42); print('OK')"
```
正常输出，无报错。

---

### 步骤 9：编写 data.py 数据加载与增强模块

**执行人**：👤+🤖

**依据文档**：DDS §3（data.py 完整设计，含 RobustImageFolder + create_datasets + get_dataloaders + 增强管线）、ADS §4.1 及 I-01/I-02 接口契约、SRS FR-1（AC-1.1~1.8）和 FR-2（AC-2.1~2.5）

**做什么**：将 DDS §3 的完整设计发给 AI，生成 `data.py`。关键组件：

1. **常量**：`IMAGENET_MEAN`、`IMAGENET_STD`、`SUPPORTED_EXTENSIONS`
2. **`RobustImageFolder` 类**：继承 `ImageFolder`，重写 `__getitem__` 捕获损坏文件（`PIL.UnidentifiedImageError`）
3. **`_build_train_transform(config)`**：Resize(224,BILINEAR) → RandomHorizontalFlip(0.5) → RandomRotation(±15°) → ColorJitter(0.2,0.2) → ToTensor → Normalize
4. **`_build_eval_transform(config)`**：Resize(224,BILINEAR) → ToTensor → Normalize（无增强）
5. **`create_datasets(config)`**：ImageFolder 加载 → 7:1:2 分层划分（seed=42）→ SubsetWithTransform 封装
6. **`get_dataloaders(config, datasets)`**：train(batch=256,shuffle=True) / val(batch=256,shuffle=False) / test(batch=256,shuffle=False)，num_workers 保护

**✅ 验收**：
```bash
python -c "
from config import load_config
from data import create_datasets, get_dataloaders
c = load_config('configs/config.yaml')
train_ds, val_ds, test_ds = create_datasets(c)
raw_img, raw_label = train_ds.dataset[train_ds.indices[0]]  # transform 之前
print(f'原始图像尺寸: {raw_img.size}')  # 应为 (64, 64)
print(f'训练集: {len(train_ds)}, 验证集: {len(val_ds)}, 测试集: {len(test_ds)}')
train_dl, val_dl, test_dl = get_dataloaders(c, (train_ds, val_ds, test_ds))
img, label = next(iter(train_dl))
print(f'图像形状: {img.shape}, 标签形状: {label.shape}')
"
```
预期输出：训练集约 18900 张、图像形状 `(256,3,224,224)`、标签形状 `(256,)`。

---

### 步骤 10：编写 test_data.py 数据模块单元测试

**执行人**：👤+🤖

**依据文档**：SRS FR-1 验收标准（AC-1.1~1.8）、DDS §3

**做什么**：让 AI 生成 `tests/test_data.py`，至少覆盖：
1. 数据集加载成功，配对成功率 100%（AC-1.1）
2. 类别标签映射 0~9 与 class_names 一致（AC-1.2）
3. 数据集划分比例 7:1:2 ± 1%（AC-1.3）
4. 图像上采样后形状为 `(3, 224, 224)`（AC-1.4）
5. 标准化后数值范围合理（均值约 0，标准差约 1）（AC-1.5）
6. DataLoader batch_size=256 正确（AC-1.6）
7. 训练 DataLoader 的增强生效（两次取同一 batch 不完全相同）
8. 验证 DataLoader 确定性（两次取同一 batch 完全相同）

运行：
```bash
pytest tests/test_data.py -v
```

**✅ 验收**：全部测试通过。

---

## 阶段三：模型与训练模块（步骤 11-16，预计 1-2 天 + 训练时间）

### 步骤 11：编写 model.py 模型构建模块

**执行人**：👤+🤖

**依据文档**：DDS §4（model.py 完整设计，含 7 个核心函数伪代码）、ADS §4.2 及 I-03/I-04/I-05 接口契约、SRS FR-3（AC-3.1~3.6）

**做什么**：将 DDS §4 的完整设计发给 AI，生成 `model.py`。关键函数：

1. **`build_model(config) -> nn.Module`**：模型选择→预训练加载→分类头替换（512→10）→Kaiming 初始化→冻结骨干→设备迁移
2. **`freeze_backbone(model, model_name)`**：设置除分类头外所有参数 `requires_grad=False`
3. **`_unfreeze_last_n_blocks(model, model_name, n)`**：解冻最后 N 个残差块（1=layer4，2=layer3+4，…）
4. **`get_loss_fn() -> nn.Module`**：返回 `CrossEntropyLoss()`
5. **`save_checkpoint(model, optimizer, epoch, metrics, path, config)`**：构建检查点字典 → `os.replace(tmp, target)` 原子写入
6. **`load_checkpoint(path, model, optimizer=None) -> tuple[int, dict]`**：加载到 CPU → 恢复模型+优化器 → 返回 epoch+metrics

**✅ 验收**：
```bash
python -c "
from config import load_config
from model import build_model, freeze_backbone, get_loss_fn
c = load_config('configs/config.yaml')
model = build_model(c)
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'总参数: {total:,}, 可训练: {trainable:,}')
"
```
预期：总参数约 11,689,610，可训练约 5,130（冻结模式）。

---

### 步骤 12：编写 test_model.py 模型模块单元测试

**执行人**：👤+🤖

**依据文档**：SRS FR-3 验收标准、DDS §4

**做什么**：让 AI 生成 `tests/test_model.py`，至少覆盖：
1. 模型构建成功，参数量约 11.7M（AC-3.1）
2. 输入 `(1,3,224,224)` → 输出 `(1,10)`（AC-3.3）
3. 冻结后仅分类头可训练，可训练参数 ≈ 5,130（AC-3.3 部分）
4. 预训练权重可加载（AC-3.4）
5. checkpoint 保存+加载往返一致（AC-3.5）
6. 损失函数输出为标量且 ≥ 0

运行：
```bash
pytest tests/test_model.py -v
```

**✅ 验收**：全部测试通过。

---

### 步骤 13：编写 train.py 模型训练模块

**执行人**：👤+🤖

**依据文档**：DDS §5（train.py 完整设计，含 Trainer 类设计 + 7 个方法伪代码）、ADS §4.3 及 I-06 接口契约、SRS FR-4（AC-4.1~4.12）

**做什么**：将 DDS §5 的完整设计发给 AI，生成 `train.py`。关键组件：

1. **`Trainer` 类**：
   - `__init__`：初始化全部状态（best_val_acc=0, epochs_no_improve=0, train_history）
   - `_train_one_epoch()`：训练循环 → 前向 → loss → 反向 → optimizer.step() → tqdm 更新
   - `_validate(data_loader)`：no_grad → 累加 loss+correct → 返回 (val_loss, val_acc)
   - `_handle_oom(error)`：清理显存 → batch_size//=2（最小32）→ 重建 DataLoader → 重试
   - `run()`：主循环（训练→验证→LR调度→显存→TensorBoard→检查点→早停）
2. **`run_training(config)` 入口**：编排所有步骤（set_seed→数据→模型→优化器→调度器→Trainer→run）

**✅ 验收**：代码审查——OOM 恢复逻辑完整、早停逻辑正确、LR 调度挂钩正确。

---

### ⭐ 步骤 14：首次试训练（2 epoch，关键里程碑）

**执行人**：👤（人工执行，🟢 里程碑）

**目的**：快速验证训练流程从头到尾能跑通。

**操作**：
1. 临时修改 `configs/config.yaml` 中 `train.epochs: 2`
2. 运行：
```bash
python main.py --mode train --config configs/config.yaml
```
3. 观察：
   - 数据加载正常（训练集 18900 张、验证集 2700 张）
   - 模型构建成功（总参数约 11.7M，可训练约 5,130）
   - loss 正常下降（不出现 NaN）
   - 2 epoch 完成，`checkpoints/best_model.pth` 生成

**✅ 验收**（CP-2 人工检核点）：
- 2 epoch 完整跑通，无异常退出
- `best_model.pth` 存在（文件 > 40 MB）
- `checkpoints/config_*.yaml` 配置快照存在
- `logs/` 下有 TensorBoard 事件文件

**通过后**：恢复 `epochs: 50`

---

### 步骤 15：正式训练（50 epochs）

**执行人**：👤

**操作**：
```bash
python main.py --mode train --config configs/config.yaml
```

训练期间可查看 TensorBoard（另开终端）：
```bash
tensorboard --logdir logs/
```

**✅ 验收**（🟢 里程碑）：
- 50 epochs 完成，中途无 CUDA OOM
- 峰值显存 ≤ 3.0 GB（查看每 epoch 日志或 TensorBoard）
- 验证集 Top-1 Accuracy ≥ 90%（期望值）
- `checkpoints/best_model.pth` 为最佳 epoch 模型

---

### 步骤 16：训练调优分析（如果不达标）

**执行人**：👤+🤖

**什么时候**：步骤 15 结束后 val_acc < 90%

**做什么**：
1. 把训练日志（loss 曲线、val_acc 曲线、最终指标）发给 AI
2. 让 AI 分析：过拟合？欠拟合？学习率不合适？
3. 按 DDS §5 升级路径尝试：
   - ① 解冻 layer4（`model.unfreeze_layers: 1`），lr=1e-5，再训练 20 epochs
   - ② 切换为 `model.name: "mobilenet_v3_large"`
4. **每次只改一个变量**，记录对比

**⚠️ 注意**：AI 只能给建议，调参决策由人拍板。

**✅ 验收**：val_acc ≥ 90% 或确认已达当前配置上限。

---

## 阶段四：评估、推理与入口模块（步骤 17-21，预计 1 天）

### 步骤 17：编写 evaluate.py 模型评估模块

**执行人**：👤+🤖

**依据文档**：DDS §6（evaluate.py 完整设计，含 4 个核心函数伪代码）、ADS §4.4 及 I-07 接口契约、SRS FR-5（AC-5.1~5.8）

**做什么**：将 DDS §6 的完整设计发给 AI，生成 `evaluate.py`。关键函数：

1. **`_compute_metrics(all_preds, all_labels, all_probs, class_names) -> dict`**：Top-1/2 Acc + 各类别 P/R/F1/TP/FP/FN + 混淆矩阵（行归一化）+ Top-3 易混淆对
2. **`_generate_report(metrics, output_dir) -> str`**：Markdown 报告（整体指标表 + 各类别表 + 易混淆分析 + 混淆矩阵图引用）
3. **`_export_misclassified(all_preds, all_labels, all_probs, class_names, output_dir)`**：预测错误样本 → CSV
4. **`run_evaluation(config, model_path) -> dict`**：加载模型 → 测试集评估 → 计算指标 → JSON+MD+PNG+CSV

**✅ 验收**：
```bash
python main.py --mode evaluate --config configs/config.yaml --model checkpoints/best_model.pth
```
评估正常完成，`outputs/evaluation/` 下生成 `eval_results.json`、`eval_report.md`、`confusion_matrix.png`、`misclassified_samples.csv`。

---

### 步骤 18：编写 predict.py 推理预测模块

**执行人**：👤+🤖

**依据文档**：DDS §7（predict.py 完整设计，含 3 个核心函数伪代码）、ADS §4.5 及 I-08/I-09 接口契约、SRS FR-6（AC-6.1~6.8）

**做什么**：将 DDS §7 的完整设计发给 AI，生成 `predict.py`。关键函数：

1. **`_load_and_preprocess(image_path, config, device)`**：PIL 加载 → 通道自适应（灰度→RGB, RGBA→RGB, CMYK→RGB）→ Resize → Normalize → Tensor
2. **`predict_single_image(image_path, model_path, config)`**：单张推理 → Top-K 类别+置信度 → 低置信度告警
3. **`predict_batch(input_dir, model_path, config)`**：批量推理 → 错误隔离 → CSV 汇总 → 类别分布统计

**✅ 验收**：
```bash
# 找一张测试集图像验证
python main.py --mode predict --config configs/config.yaml \
  --model checkpoints/best_model.pth \
  --input "D:/DataDownload/EuroSat_Dataset/EuroSAT/Forest/Forest_1.jpg"
```
输出：`类别: Forest, 置信度: 0.XXXX`，耗时 ≤ 50 ms。

---

### 步骤 19：编写 main.py CLI 统一入口

**执行人**：👤+🤖

**依据文档**：DDS §9（main.py 完整设计，含 build_parser + main + mode_check 伪代码）、ADS §4.8、SRS FR-8（AC-8.1~8.8）

**做什么**：将 DDS §9 的完整设计发给 AI，生成 `main.py`。关键功能：

1. **`build_parser()`**：argparse，含 `--mode`（train/evaluate/predict/check）+ `--config` + `--data-dir` + `--model` + `--input` + `--output` + `--top-k` + `--batch-size` + `--epochs` + `--lr` + `--resume` + `--help`（含 4 个用法示例）
2. **`main()`**：解析参数 → CLI 覆盖 → 模式路由
3. **`mode_train(config)` / `mode_evaluate(config, model_path)` / `mode_predict(config, args, model_path)` / `mode_check(args)`**

**✅ 验收**：
```bash
python main.py --help
```
输出完整参数说明 + 4 个用法示例。

---

### 步骤 20：编写 test_evaluate.py 和 test_predict.py

**执行人**：👤+🤖

**做什么**：让 AI 生成：
1. `tests/test_evaluate.py`：指标计算正确性测试（用已知标签验证 P/R/F1 公式）、混淆矩阵形状测试
2. `tests/test_predict.py`：通道自适应测试（灰度/RGBA）、预处理尺寸测试、Top-K 输出格式测试

运行：
```bash
pytest tests/test_evaluate.py tests/test_predict.py -v
```

**✅ 验收**：全部测试通过。

---

### ⭐ 步骤 21：模块集成联调（人工检核点 CP-3）

**执行人**：👤（必须人工确认）

**检核内容**：
1. 四种模式（train/evaluate/predict/check）均能正常启动
2. train → evaluate → predict 完整流程跑通（用 2 epoch 迷你训练）
3. 命令行参数覆盖配置文件生效（`--epochs 5` 实际训练 5 epoch）
4. `--help` 输出完整

**✅ 验收**：CP-3 通过，阶段四完成。

---

## 阶段五：代码审查（步骤 22-24，静态审查 + 人工审查，预计 0.5 天）

### ⭐ 步骤 22：代码静态审查（自动工具扫描）

**执行人**：👤+🤖

**依据文档**：《深度学习系统开发项目_研发工作流程与方法规则V1.0》§5.1（SC-01~SC-11）

**操作**：

```bash
cd d:/myproject/EuroSat_images_classification

# 1. 代码风格检查
pip install flake8
flake8 *.py --max-line-length=120 --ignore=E501,W503

# 2. 搜索危险模式
grep -rn "except:" *.py              # 检查裸 except（应无结果或极少）
grep -rn "os\.rename" *.py          # 应无结果（全部用 os.replace）

# 3. 搜索硬编码路径（不应有除 config.yaml 外的绝对路径）
grep -rn "D:\\\\" *.py
grep -rn "C:\\\\" *.py

# 4. 导入未使用检查
flake8 *.py --select=F401

# 5. 函数长度检查（单函数 ≤ 150 行，可在 IDE 中查看）

# 6. 测试覆盖率
pip install pytest-cov
pytest tests/ --cov=. --cov-report=term
```

**审查标准**：

| 编号 | 审查项 | 通过标准 |
|:---:|------|------|
| SC-01 | flake8 | 无 E 类 Error |
| SC-02 | 裸 except | 0 个（或 ≤ 2 且有注释说明理由） |
| SC-03 | `os.rename` | 0 个 |
| SC-04 | 硬编码路径 | 0 个 |
| SC-05 | 未使用导入 | 0 个 |
| SC-06 | 测试覆盖率 | 核心模块 ≥ 60% |

**✅ 验收**：SC-01~SC-06 全部通过或不阻塞。

---

### ⭐ 步骤 23：代码人工审查（逐模块走查）

**执行人**：👤（必须人工执行）

**依据文档**：《深度学习系统开发项目_研发工作流程与方法规则V1.0》§5.2（MR-01~MR-10）

**审查清单**（逐模块检查，每项打 ✅ 或 ❌）：

| 编号 | 审查项 | config | data | model | train | eval | predict | utils | main |
|:---:|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MR-01 | 需求覆盖：每个 FR 有对应实现 | | | | | | | | |
| MR-02 | 接口一致性：函数签名与 ADS/DDS 一致 | | | | | | | | |
| MR-03 | 异常处理：每种异常有 try/except | | | | | | | | |
| MR-04 | 边界条件：空/None/极值有处理 | | | | | | | | |
| MR-05 | 显存管理：empty_cache 调用正确 | | | | | | | | |
| MR-06 | 随机种子：set_seed 在训练入口 | | | | | | | | |
| MR-07 | 日志：INFO/WARNING/ERROR 使用合理 | | | | | | | | |
| MR-08 | 数据流：tensor 形状变化正确 | | | | | | | | |
| MR-09 | **过度开发检测（7 项）** | | | | | | | | |
| MR-10 | 注释：公开接口有 docstring | | | | | | | | |

**MR-09 过度开发检测（重点关注）**：
- [ ] 是否存在 SRS/ADS/DDS 中未定义的功能？
- [ ] 是否存在"预留"参数（标注"暂未使用"或"未来扩展"）？
- [ ] 是否引入了需求文档未要求的第三方依赖？
- [ ] 是否使用了工厂模式/抽象类等过度设计？
- [ ] 是否导入了但未实际使用的模块？
- [ ] 函数复杂度是否远超需求？（如分类任务不需要自定义 Dataset 继承体系）
- [ ] 配置文件是否有超过 20% 的字段当前版本不使用？

**✅ 验收**（CP-4 人工检核点）：MR-01~MR-10 全部通过，特别是 MR-09 无过度开发。

---

### 步骤 24：审查问题修复与复审

**执行人**：👤+🤖

**做什么**：
1. 根据步骤 22 和 23 的审查结果，逐条修复
2. AI 修复代码后，重新运行步骤 22 静态审查
3. 人工复审高风险修复点（异常处理逻辑、显存管理）
4. 记录修复内容到审查报告

**✅ 验收**：所有审查问题关闭，静态审查 6 项通过，人工审查 10 项通过。

---

## 阶段六：集成测试与验收（步骤 25-27，预计 0.5 天）

### 步骤 25：端到端集成测试 E2E

**执行人**：👤+🤖

**做什么**：让 AI 生成 `tests/test_e2e.py`：

> 编写 E2E 集成测试：用 50 张图像的迷你数据集（从训练集中每类抽取 5 张），执行完整流程：
> 1. 数据加载与划分（迷你训练集 35 / 验证集 5 / 测试集 10）
> 2. 训练 3 epoch
> 3. 评估（输出指标）
> 4. 推理 5 张图（输出结果）
> 预期 10 分钟内完成，所有步骤无异常。

运行：
```bash
pytest tests/test_e2e.py -v
```

**✅ 验收**：E2E 通过，耗时 ≤ 10 分钟。

---

### 步骤 26：需求验收矩阵检查

**执行人**：👤（人工逐条对照）

**做什么**：以 SRS 第 7 章"验收标准汇总"为检查清单，逐条验证：

| 需求编号 | 验收项 | 通过标准 | 实测结果 | 判定 |
|---------|--------|---------|---------|:---:|
| FR-3.2 | 模型参数量 | ≤ 12M | | |
| FR-4.10 | 训练峰值显存 | ≤ 3.0 GB | | |
| FR-5.1 | Top-1 Accuracy | ≥ 90% | | |
| ... | ... | ... | ... | |

**✅ 验收**：全部 Must 需求通过。

---

### ⭐ 步骤 27：边界场景逐一验证（人工检核点 CP-5）

**执行人**：👤（必须人工执行）

**做什么**：对照 SRS 第 6 章"边界情况与异常处理"（BC-1~BC-14），逐条触发测试：

| BC | 场景 | 验证方式 | 结果 |
|:---:|------|---------|:---:|
| BC-1 | 数据路径不存在 | 临时改 config 指向错误路径 | |
| BC-5 | 预训练权重下载失败 | 断网后运行（或临时改 weights 名） | |
| BC-6 | Ctrl+C 中断训练 | 训练中按 Ctrl+C，检查 emergency_*.pth | |
| BC-7 | 图像尺寸不是 64×64 | 用其他尺寸图像测试推理 | |
| BC-8 | 灰度图输入 | 用灰度图测试推理 | |
| BC-10 | 低置信度预测 | 用非卫星图像测试推理 | |
| ... | ... | ... | |

**✅ 验收**（CP-5）：SRS 中全部 BC 场景已验证，处理方案生效。

---

## 阶段七：程序发布与技术文档（步骤 28-33，预计 0.5-1 天）

### 步骤 28：代码整理与发布准备

**执行人**：👤

**操作**：
```bash
cd d:/myproject/EuroSat_images_classification

# 清理临时文件
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
find . -name "*.tmp" -delete 2>/dev/null

# 确认 .gitignore 完整
git status  # 应只显示源代码和文档，无生成文件
```

**✅ 验收**：`git status` 无 checkpoints/logs/runs/outputs 下的文件。

---

### 步骤 29：编写用户文档（5 份）

**执行人**：👤+🤖

**依据文档**：《深度学习系统开发项目_研发工作流程与方法规则V1.0》§7.2

| 文档 | 文件 | 内容要点 |
|------|------|------|
| 项目概览 | `README.md` | 项目简介、环境要求、快速开始（3步）、目录结构、许可证 |
| 安装指南 | `INSTALL.md` | conda 创建环境、PyTorch 安装（强调 cu113 wheel 源）、依赖安装、环境检测 |
| 使用指南 | `USER_GUIDE.md` | 四种模式详解（train/evaluate/predict/check）、CLI 参数表、典型场景示例 |
| 配置指南 | `CONFIG_GUIDE.md` | 7 个配置域详解、关键参数取值范围、3 种典型场景（快速验证/精度优先/显存受限） |
| 常见问题 | `TROUBLESHOOTING.md` | CUDA OOM、安装失败、loss NaN、推理速度慢、预训练下载失败、路径问题 |

**操作**：逐份将内容需求发给 AI，AI 生成初稿，人工审核修改后保存到 `docs/`。

**✅ 验收**：5 份文档齐全，按文档操作可成功运行项目。

---

### 步骤 30：编写技术文档（3 份）

**执行人**：👤+🤖

**依据文档**：《深度学习系统开发项目_研发工作流程与方法规则V1.0》§7.3

| 文档 | 文件 | 内容要点 |
|------|------|------|
| 更新日志 | `CHANGELOG.md` | V1.0 功能列表（按 Added 分类） |
| 架构说明 | `ARCHITECTURE.md` | 四层架构图、8 模块依赖关系、数据流、技术栈版本锁定理由 |
| 模型说明 | `MODEL.md` | ResNet18 选型理由、参数量/FLOPs、训练策略、调优建议、升级路径 |

**✅ 验收**：3 份文档齐全。

---

### ⭐ 步骤 31：发布包完整性校验（人工检核点 CP-6）

**执行人**：👤（必须人工执行）

**操作**：
1. 在另一个临时目录 clone 或复制项目（模拟全新用户环境）
2. 创建新虚拟环境
3. 按 `INSTALL.md` 全新安装依赖
4. 运行 `python check_env.py` → 全部通过
5. 用预训练的 `best_model.pth` 跑一次推理 → 输出正确
6. 生成文件 checksum：
```bash
find . -name "*.py" -o -name "*.yaml" -o -name "*.md" | sort | xargs sha256sum > release_checksums.txt
```

**✅ 验收**（CP-6）：全新环境从头安装运行成功，输出与开发环境一致。

---

### 步骤 32：Git 提交与备份

**执行人**：👤

**操作**：
```bash
cd d:/myproject/EuroSat_images_classification

# 1. 确认状态
git status

# 2. 添加所有源代码和文档（排除 .gitignore 中的生成文件）
git add *.py configs/ docs/ tests/ requirements.txt README.md INSTALL.md \
       USER_GUIDE.md CONFIG_GUIDE.md TROUBLESHOOTING.md CHANGELOG.md \
       ARCHITECTURE.md MODEL.md .gitignore

# 3. 提交
git commit -m "V1.0: EuroSAT卫星影像分类系统首次发布

- 8模块架构: config/data/model/train/evaluate/predict/utils/main
- ResNet18迁移学习, EuroSAT 10类土地覆盖分类
- 50 epochs训练, batch=256, FP32, GTX 1050 Ti
- 完整文档: SRS/ADS/DDS/技术选型/用户文档/技术文档
- 测试: 单元测试+E2E集成测试"

# 4. 创建 V1.0 标签
git tag -a v1.0.0 -m "V1.0 正式发布"
```

**✅ 验收**：`git log --oneline` 可见提交，`git tag` 可见 v1.0.0。

---

### 步骤 33：推送到 GitHub

**执行人**：👤

**前提**：GitHub 远程仓库已创建（如 `https://github.com/<用户名>/EuroSAT_Classification`）

**操作**：
```bash
cd d:/myproject/EuroSat_images_classification

# 1. 添加远程仓库（首次）
git remote add origin https://github.com/<用户名>/EuroSAT_Classification.git

# 2. 推送代码和标签
git push -u origin main
git push origin v1.0.0

# 3. 验证
# 在浏览器打开 https://github.com/<用户名>/EuroSAT_Classification
# 确认所有文件已上传、tag v1.0.0 可见
```

**注意事项**：
- `.gitignore` 确保 `checkpoints/`、`logs/`、`runs/`、`outputs/`、`*.pth` 不会上传
- 配置文件中的个人路径（`D:\DataDownload\...`）已在 `config.yaml` 中可修改，README 中说明
- 模型权重文件（`.pth`）不上传 GitHub（文件大），在 README 中说明训练方式

**✅ 验收**：GitHub 仓库可见完整源代码和文档，tag v1.0.0 存在。

---

## 阶段八：研发总结与维护（步骤 34-37，总结 0.5 天 + 维护持续）

### 步骤 34：编写发布说明（Release Notes）

**执行人**：👤

**操作**：在 GitHub 仓库的 Releases 页面（或 `RELEASE_NOTES.md` 文件）编写 V1.0 发布说明：

```markdown
# V1.0.0 — 首次正式发布

## 功能
- ResNet18 迁移学习，EuroSAT 10 类土地覆盖分类
- 完整训练流程（50 epochs, batch=256, FP32）
- 评估（Top-1/2 Acc, P/R/F1, 混淆矩阵, 易混淆类别分析）
- 推理（单张/批量, Top-K 预测, 通道自适应）
- 训练监控（TensorBoard, 训练曲线, GPU 显存）
- 环境检测（check_env.py）

## 性能（GTX 1050 Ti 4GB）
- 训练显存: ~1.5 GB
- 推理耗时: ≤ 50 ms/张
- 模型参数量: 11.7M

## 已知限制
- 仅支持 EuroSAT RGB 版本（64×64），不支持全波段 13 通道
- FP16 训练未启用（Pascal 架构无加速）
- 纯 CPU 推理速度较慢

## 安装
见 INSTALL.md
```

**✅ 验收**：Release Notes 可读、功能列表完整。

---

### 步骤 35：建立维护基线

**执行人**：👤

**操作**：
1. 归档全部文档到 `docs/`（SRS/ADS/DDS/技术选型/用户文档/技术文档/测试报告）
2. 归档训练产物：`best_model.pth` + `config_snapshot.yaml` + 训练日志
3. 建立文件完整性记录：
```bash
cd d:/myproject/EuroSat_images_classification
find . -type f \( -name "*.py" -o -name "*.yaml" -o -name "*.md" -o -name "*.txt" \) \
  -not -path "./checkpoints/*" -not -path "./logs/*" -not -path "./runs/*" \
  | sort | xargs sha256sum > V1.0_checksums.txt
```

**✅ 验收**：`V1.0_checksums.txt` 可验证任意文件完整性。

---

### 步骤 36：定义维护规则

**执行人**：👤

**维护规则**（记录在 `MAINTENANCE.md` 或项目 Wiki）：

| 规则 | 说明 |
|------|------|
| **非必要不修改** | V1.0 交付后，除非 Bug 修复或明确新需求，不进行代码变更 |
| **每改必测** | 任何代码修改必须通过 `pytest tests/` 所有现有测试 |
| **文档同步** | 代码变更后同步更新 CHANGELOG 和受影响的文档 |
| **依赖冻结** | 不主动升级 PyTorch/CUDA，仅修复安全漏洞 |
| **问题追踪** | 使用 GitHub Issues（Bug/Feature/Question 标签） |
| **分支策略** | `main` 保持稳定；新功能在 feature 分支开发，合并前通过审查 |

---

### 步骤 37：编写研发技术总结报告

**执行人**：👤+🤖

**依据文档**：《深度学习系统开发项目_研发工作流程与方法规则V1.0》§8.3

**做什么**：编写 `docs/研发总结报告V1.0.md`，包含：

| 章节 | 内容 |
|------|------|
| **研发流程回顾** | 实际执行的阶段与步骤、计划 vs 实际时间、偏离分析 |
| **程序架构分析** | 最终架构图、模块依赖关系、与 ADS 设计的一致性评估 |
| **模块结构与代码规模** | 每个模块实际代码行数、与 DDS 估算的偏差、未按设计实现的部分及理由 |
| **关键流程验证** | 训练/推理流程实测性能 vs SRS 目标值（显存/耗时/精度） |
| **算法模型总结** | 最终模型指标、调优过程、最终配置 |
| **疑难问题与解决方案** | 研发过程中遇到的 Top-N 问题、根因分析、解决方案 |
| **最终交付指标** | 需求达成率（Must/Should/Could/Won't）、质量指标、交付物清单 |
| **经验教训与改进建议** | AI 编程的有效模式/无效模式；设计文档完整度对编码效率的影响；哪些设计决策正确/错误 |

**✅ 验收**：总结报告完整，疑难问题有根因分析，经验教训可指导后续项目。

---

## 快速参考卡片

### 37 步速查表

| 步骤 | 阶段 | 内容 | 执行人 | 预计耗时 |
|:---:|:---:|------|:---:|:---:|
| 1 | 一 | 创建目录结构 | 👤 | 5 min |
| 2 | 一 | 安装 PyTorch | 👤 | 20 min |
| 3 | 一 | 安装依赖 | 👤+🤖 | 15 min |
| 4 | 一 | Git 初始化 | 👤 | 10 min |
| 5 | 一 | check_env.py | 👤+🤖 | 30 min |
| ⭐6 | 一 | **CP-1 环境验收** | 👤 | 10 min |
| 7 | 二 | config.py | 👤+🤖 | 1 h |
| 8 | 二 | utils.py | 👤+🤖 | 45 min |
| 9 | 二 | data.py | 👤+🤖 | 1.5 h |
| 10 | 二 | test_data.py | 👤+🤖 | 30 min |
| 11 | 三 | model.py | 👤+🤖 | 1 h |
| 12 | 三 | test_model.py | 👤+🤖 | 30 min |
| 13 | 三 | train.py | 👤+🤖 | 2 h |
| 🟢14 | 三 | **试训练 2 epoch** | 👤 | 15 min |
| 15 | 三 | 正式训练 50 epoch | 👤 | ~50 min |
| 16 | 三 | 训练调优（如需） | 👤+🤖 | 1-3 h |
| 17 | 四 | evaluate.py | 👤+🤖 | 1 h |
| 18 | 四 | predict.py | 👤+🤖 | 1 h |
| 19 | 四 | main.py | 👤+🤖 | 45 min |
| 20 | 四 | test_evaluate + test_predict | 👤+🤖 | 30 min |
| ⭐21 | 四 | **CP-3 集成联调** | 👤 | 20 min |
| ⭐22 | 五 | **静态审查** | 👤+🤖 | 30 min |
| ⭐23 | 五 | **人工审查** | 👤 | 1 h |
| 24 | 五 | 审查修复 | 👤+🤖 | 30 min |
| 25 | 六 | E2E 测试 | 👤+🤖 | 20 min |
| 26 | 六 | 需求验收矩阵 | 👤 | 30 min |
| ⭐27 | 六 | **CP-5 边界验证** | 👤 | 1 h |
| 28 | 七 | 代码整理 | 👤 | 15 min |
| 29 | 七 | 用户文档（5份） | 👤+🤖 | 1.5 h |
| 30 | 七 | 技术文档（3份） | 👤+🤖 | 45 min |
| ⭐31 | 七 | **CP-6 发布校验** | 👤 | 30 min |
| 32 | 七 | Git 提交 | 👤 | 10 min |
| 33 | 七 | 推送 GitHub | 👤 | 10 min |
| 34 | 八 | Release Notes | 👤 | 15 min |
| 35 | 八 | 维护基线 | 👤 | 15 min |
| 36 | 八 | 维护规则 | 👤 | 10 min |
| 37 | 八 | 研发总结报告 | 👤+🤖 | 1.5 h |

### AI 执行时的文档引用规范

AI 生成代码时，必须标注依据：

| 代码文件 | 依据的设计文档 | 关键引用 |
|---------|-------------|---------|
| `config.py` | DDS §2 | ADS I-10 |
| `data.py` | DDS §3 | ADS I-01, I-02; SRS FR-1, FR-2 |
| `model.py` | DDS §4 | ADS I-03, I-04, I-05; SRS FR-3 |
| `train.py` | DDS §5 | ADS I-06; SRS FR-4 |
| `evaluate.py` | DDS §6 | ADS I-07; SRS FR-5 |
| `predict.py` | DDS §7 | ADS I-08, I-09; SRS FR-6 |
| `utils.py` | DDS §8 | SRS FR-7 |
| `main.py` | DDS §9 | SRS FR-8 |

---

> **文档结束** | 本指南将 EuroSAT 卫星影像分类系统研发分解为 37 个可逐步骤执行的具体操作，每步标注执行人、依据文档、操作内容和验收标准。AI 可参照本指南分步执行，人工在 8 个检核点（⭐CP-1~CP-6 + 🟢试训练 + 🟢正式训练）介入确认。全流程预计 6-8 天（含训练时间）。
