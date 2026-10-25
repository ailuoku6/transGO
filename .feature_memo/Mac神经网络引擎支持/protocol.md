# 接口与兼容

## CLI

- 所有推理命令：`--inference-backend torch|coreml`，默认 torch；`--coreml-batch-size 16`；`--coreml-cache-dir runs/coreml`。
- train 不增加 NPU 选项；loop 训练仍由 config.device 选择 PyTorch 设备。
- benchmark 新增 `--checkpoint` 和可选 `--coreml-compute-plan`。后者输出 static_compute_plan、preferred_operations、runtime_utilization_measured=false。
- doctor.coreml 报告依赖版本和平台资格；ane_execution_verified=false 仅表示 doctor 没有分析模型执行。

## Python

- `Evaluator(model, device, *, backend='torch', coreml_batch_size=16, coreml_cache_dir='runs/coreml')`。
- predict/predict_arrays 仍返回 outputs/policies/wdl/values；evaluate_arrays 返回 float32 numpy policy/value/score。
- features=[B,21,N,N]，N=2..19；policy=[B,N*N+1]；WDL 顺序 loss/draw/win；ownership=[B,N,N,3]；score_mean/logvar、moves_left=[B]。
- 非法 policy logits 在 CPU 置 -inf，补齐行不计入结果或 positions。prepare(board_size) 不计 calls/positions。
- generate_parallel/evaluate_parallel/run_pipeline/PlayService/serve 通过可选 evaluator_options 传递后端，在父进程加载模型。

## 缓存及记录

- 默认 runs/coreml/HASH-NxN.mlpackage；HASH 包含导出版本、配置、所有权重、固定batch和PyTorch/coremltools版本，棋盘体现在文件名。
- 临时目录完整写入后再发布包。原生 MLModel 加载可能更新包目录 mtime，缓存测试禁止重复 convert，而不以目录时间判定。
- loop iteration JSON 记录 inference_backend/coreml_batch_size；training summary 的 device 是真实训练设备。并行评估 play_settings 和人机 state 也记录后端。
- Config/checkpoint/replay schema 不改；显式 coreml 失败不会静默回退 PyTorch。
- Core ML 来源权重在 CPU 加载；与推理用不到的 MPS 可用性解耦。
- 已运行进程不会热加载新实现，用户在合适边界启动带新参数的进程才能启用。
