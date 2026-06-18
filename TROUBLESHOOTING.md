# 常见问题排查

## 环境问题

### `torch.cuda.is_available()` 返回 False
- NVIDIA 驱动版本 ≥ 465.x
- 确认 PyTorch 是 `+cu113` 版本：`pip show torch | grep Version`
- 不要装 CPU 版本

### `conda create` 网络超时
```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda create -n eurosat python=3.8 -y
```

### 预训练权重下载失败
```
[ASN1: NOT_ENOUGH_DATA] not enough data
```
**解决**：浏览器下载 `resnet18-f37072fd.pth`（约 45MB），保存到本地，在 `configs/config.yaml` 中设置 `model.pretrained_path` 指向该文件。

### `pip install -r requirements.txt` 编码错误
```
UnicodeDecodeError: 'gbk' codec can't decode byte
```
**解决**：requirements.txt 不应含中文注释。已修复为纯 ASCII。

## 训练问题

### CUDA OOM（显存不足）
```
RuntimeError: CUDA out of memory
```
**自动恢复**：程序会 `batch_size //= 2` 并重试（最小 32）。若仍不行：
```yaml
train:
  batch_size: 64    # 手动降低
system:
  num_workers: 2    # 降低内存占用
```

### 训练 loss 出现 NaN
- 检查学习率是否过高
- 检查数据路径是否正确
- 尝试关闭数据增强：`augmentation` 下各参数设为 0

### 训练准确率低（<60%）
- 确认预训练权重已加载：`python -c "from config import load_config; from model import build_model; c=load_config('configs/config.yaml'); m=build_model(c)"` 看日志有无 `已从本地加载预训练权重`
- 确认 `model.freeze_backbone: true` 和 `unfreeze_layers: 1` 均已设置
- 训练不够：增加 `epochs`

### 解冻 layer4 微调时报错
```
ValueError: loaded state dict contains a parameter group that doesn't match
```
- 这是 optimizer 参数数量变化，程序已自动处理（跳过 optimizer 恢复）
- 确保 `config.yaml` 中 `unfreeze_layers` 已改为 1

## 推理问题

### 推理结果全是同一类别
- 确认加载了正确的模型：`--model checkpoints/best_model.pth`
- 检查预训练权重是否已加载

### 灰度图或 RGBA 图推理报错
- 程序已自动处理灰度/RGBA→RGB 转换，不应报错
- 若仍报错，检查图像文件是否损坏

### 推理速度很慢
- GPU 推理：约 30 ms/张，正常
- CPU 推理：约 200 ms/张，正常（慢 5-10 倍）
- 若 GPU 也慢：检查是否有其他程序占用 GPU

## 其他问题

### 内存占用高（12GB+）
- DataLoader `num_workers=4` 每个子进程独立持有一份数据
- 降为 `num_workers: 2`，对训练速度影响很小

### 中文字体在图表中显示为方块
- Windows: 程序自动使用 SimHei 字体
- Linux: 需安装中文字体 `apt install fonts-wqy-microhei`
