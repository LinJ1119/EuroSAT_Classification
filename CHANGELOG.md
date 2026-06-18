# CHANGELOG

## V1.0.0 (2026-06-18) — 首次正式发布

### Added
- ResNet18 迁移学习，EuroSAT 10 类土地覆盖分类
- 8 模块架构：config / data / model / train / evaluate / predict / utils / main
- 参数高效微调 (PEFT)：冻结骨干，仅训练分类头 (~5K 参数)
- 部分微调：解冻 layer4，适配卫星影像特征 (~8.4M 参数)
- 训练：50 epochs + 20 epochs 微调，AdamW + ReduceLROnPlateau + Early Stopping
- 评估：Top-1/2 Acc + P/R/F1 + 混淆矩阵 + 易混淆类别分析
- 推理：单张/批量 + Top-K + 通道自适应 (灰度/RGBA/CMYK)
- 训练监控：TensorBoard + GPU 显存监控 + tqdm 进度条
- 环境检测：check_env.py 一键检测 6 项
- 4 种 CLI 模式：train / evaluate / predict / check
- 完整设计文档：SRS + 技术选型 + ADS + DDS + 研发指南
- 单元测试 28 项 + E2E 集成测试

### Performance
- 测试集 Top-1 Accuracy: 93.07%
- 训练显存峰值: ~1.8 GB (batch=256, FP32)
- 单张推理耗时: ~30 ms (GPU)
- 模型参数量: 11.2M

### Known Limitations
- 仅支持 EuroSAT RGB 版本 (64×64)，不支持全波段 13 通道
- 未启用 AMP (GTX 1050 Ti Pascal 架构无 FP16 Tensor Core)
- 纯 CPU 推理速度较慢 (~200 ms/张)
