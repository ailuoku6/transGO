# 技术方案

## 推理图

- `coreml.ExportNetwork` 创建 CPU 推理快照，预计算棋盘坐标和局部 mask；显式 QKV/matmul/softmax 替代 fused MHA，训练网络及权重结构不变。
- Core ML ML Program、macOS13 target、FP16 计算、float32 I/O，保留全部六个预测头。
- CPU_AND_NE 禁用 GPU，允许 CPU 回落；实际分配由系统决定，不保证 NPU 独占。
- Evaluator 将 Core ML 原始结果转为 CPU torch tensors，继续原有合法 mask、softmax 和 WDL 价值组合。
- 固定 batch 不足补零、超过分块，去掉补齐输出；按棋盘尺寸按需导出。并行模式在 worker 启动前 prepare，避免首次编译耗尽 RPC 超时。

## 依赖和约束

- Apple Silicon/macOS13+；可选 extra `.[coreml]`，当前 .venv 已安装 coremltools9.0及其依赖，PyTorch2.14.1未变。
- 当前 PyTorch 超出 coremltools 官方已测范围。初次动态 shape trace 触发 int 转换问题，现改为固定 batch/棋盘常量，真实转换通过。
- FP32 导出图与原模型的全部头严格对齐；ANE FP16 raw logits 在随机9/19路小模型上最大约0.0113，需分别检验 raw、归一化概率及搜索接口，而非逐位一致。
- 系统编译缓存和硬件在执行沙箱中可能不可访问，真实硬件测试在原生环境运行。
- 转换/编译有启动成本，每轮新权重需转换；缓存会随轮次增长。
- MLComputePlan 是静态偏好，不是实时利用率。旧系统不支持时仅返回诊断，不阻断正常预测。
