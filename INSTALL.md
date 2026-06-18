# 安装指南

## 1. 环境要求

| 项目 | 最低要求 | 推荐 |
|------|---------|------|
| GPU | NVIDIA GTX 1050 Ti 4 GB | 同左（支持纯 CPU） |
| 操作系统 | Windows 10 64-bit | Windows 10/11, Linux |
| Python | 3.8.x | 3.8 |
| CUDA | 11.3 | 11.3 |
| 磁盘空间 | ≥ 10 GB | ≥ 20 GB (含数据集) |

## 2. 安装步骤

### 2.1 创建虚拟环境

```bash
conda create -n eurosat python=3.8 -y
conda activate eurosat
```

> `python=3.8.20` 精确版本可能不存在，用 `python=3.8` 即可，conda 自动选最新 3.8.x。

### 2.2 安装 PyTorch

**在线安装**（从官方源下载）：
```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

**本地安装**（如已下载 .whl 文件）：
```bash
cd D:/CodingStudy/wheels_download
pip install torch-1.12.1+cu113-cp38-cp38-win_amd64.whl
pip install torchvision-0.13.1+cu113-cp38-cp38-win_amd64.whl
cd d:/myproject/EuroSat_images_classification
```

**验证**：
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# 输出: True \n GeForce GTX 1050 Ti
```

### 2.3 安装其余依赖

```bash
pip install -r requirements.txt
```

如果网络慢，用清华镜像源：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2.4 环境检测

```bash
python main.py --mode check
```

全部输出"通过"即可。若数据路径未配置会有警告，下一步解决。

## 3. 配置数据路径

编辑 `configs/config.yaml`，修改 `data.root_dir` 为你的 EuroSAT 数据集路径：

```yaml
data:
  root_dir: "D:/DataDownload/EuroSat_Dataset/EuroSAT"
```

数据集目录结构应为：
```
EuroSAT/
├── AnnualCrop/
├── Forest/
├── HerbaceousVegetation/
├── Highway/
├── Industrial/
├── Pasture/
├── PermanentCrop/
├── Residential/
├── River/
└── SeaLake/
```

## 4. 常见安装问题

### `torch.cuda.is_available()` 返回 False
- 检查 NVIDIA 驱动版本 ≥ 465.x
- 确认安装的是 `+cu113` 版本而非 CPU 版本
- 运行 `nvidia-smi` 确认 GPU 被系统识别

### `conda create` 网络超时
```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
```

### `pip install` 网络超时
```bash
pip install <包名> -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 预训练权重下载失败
从浏览器下载 `resnet18-f37072fd.pth` (约45MB)：
```
https://download.pytorch.org/models/resnet18-f37072fd.pth
```
放到本地任意路径，然后在 `configs/config.yaml` 中设置：
```yaml
model:
  pretrained_path: "D:/DataDownload/Models/resnet18-f37072fd.pth"
```
