# 文件

- `transformer_go/coreml.py`：平台/依赖诊断、固定图导出、快照/缓存/批处理、设备计划。
- `transformer_go/search.py`：Evaluator 可选后端及 prepare。
- `transformer_go/runtime.py`：推理设备与 doctor。
- `transformer_go/cli.py`：运行选项和 benchmark。
- `transformer_go/actors.py`、`evaluate.py`、`pipeline.py`：父进程后端、预热与训练/推理设备分离。
- `transformer_go/play.py`、`web.py`：人机对弈后端与状态。
- `pyproject.toml`：coreml 可选依赖。
- `README.md`、`IMPLEMENTATION.md`：使用说明及实现状态。
- `tests/test_coreml.py`：图数值、批处理、快照、错误、CLI、并行服务、闭环及硬件用例。
- 本 memo 的 benchmark.json / integration.json：原始基准与真实闭环摘要。

未修改 C++、训练网络权重结构、用户原有运行目录和 replay。
