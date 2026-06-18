# CHANGELOG

## V1.0.0 (2026-06-18)

### Added
- ResNet18 迁移学习，EuroSAT 10 类土地覆盖分类（测试集 93.07%）
- 8 模块架构：config / data / model / train / evaluate / predict / utils / main
- 参数高效微调 (PEFT)：冻结骨干，仅训练分类头 (~5K)
- 部分微调：解冻 layer4，适配卫星影像特征 (~8.4M)
- 训练：50 epoch PEFT + 20 epoch 微调，AdamW + ReduceLROnPlateau + Early Stopping
- 评估：Top-1/2 Acc + P/R/F1 + 混淆矩阵 + 易混淆类别分析 + CSV 导出
- 推理：单张/批量 + Top-K + 通道自适应 (灰度/RGBA/CMYK)
- 监控：TensorBoard + GPU 显存监控 + tqdm
- CLI：4 种模式 (train/evaluate/predict/check) + 11 个参数
- 环境检测：check_env.py（6 项）
- 测试：29 项单元测试 + E2E 集成测试
- 完整设计文档：SRS + 技术选型 + ADS + DDS + 研发指南
- 5 份用户文档：README + INSTALL + USER_GUIDE + CONFIG_GUIDE + TROUBLESHOOTING

### Performance
- 测试集 Top-1: 93.07%
- 训练显存: ~1.81 GB (batch=256, FP32)
- 推理耗时: ~30 ms (GPU)
- 参数量: 11.18M
