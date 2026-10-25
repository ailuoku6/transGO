# 决策

## 2026-10-25

- 采用可选 Core ML CPU_AND_NE 进行推理，保持 PyTorch/MPS 训练；MPS 是 GPU，不是 NPU。
- 使用运行时 evaluator_options，而不是把后端写进 Config/checkpoint，保证旧训练目录可接续。
- 固定形状/显式注意力绕开 fused kernel 和新版 trace 动态 shape 转换问题，不改训练网络及权重。
- FP16 允许 NPU 部署，保留 float32 I/O 和 CPU 合法 mask，数值误差独立验证。
- 静态设备计划能证明模型有 NE 首选算子，不冒充实时 NPU 利用率或功耗数据。
- 默认 torch；不替用户重启正在运行的训练。
