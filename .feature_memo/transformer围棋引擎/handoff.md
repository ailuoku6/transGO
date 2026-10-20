# 接续（2026-10-20）

用户授权的“跨局面集中批量推理 + MCTS 子树复用”已实现，代码尚未提交。先看 README.md、IMPLEMENTATION.md、changes.md 的最新记录，早期 CPU 测试数量/规则实现描述已过时。

- 规则使用 vendored KataGo Board/BoardHistory，模型为 PyTorch；训练仅随机初始化和自身对局。
- 多 worker 是 spawn CPU 搜索进程，父进程集中推理线程每个模型只加载一次；按 model key/几何合批。selfplay 用一个模型，evaluation 的 candidate/baseline 分开批次和两棵树，每手两树都 advance。
- C++ Search 的 reuse=True 持久树按完整着法来源校验。目标预算包含保留的子边访问，返回 new_simulations/reused_simulations；根噪声每次从 base priors 重建。失败回调清树。
- selfplay/evaluate 自动复用；Search.run 默认 fresh，Python search 不传 session 也是 fresh。权重若原地变化，调用方必须 reset 会话。
- --inference-batch-size 默认 256、--inference-wait-ms 默认 2，只用于多 worker，不改 Config/checkpoint；旧 checkpoint 可继续训练。推理统计见 selfplay inference_summary、evaluation inference。
- 已重新编译 native，全套 109 passed、5 skipped；MPS 在当前沙箱 unavailable。真实并行 CLI 自博弈 2 局→训练 1 step→评测 2 局通过，config 一致。未改现有 runs/zero-9x9。
- 基准 `.venv/bin/python -m benchmarks.search_optimizations`：CPU smoke 5x5/8手/64模拟，树复用 NN 518→489；4 worker 集中调用 548→138，平均 batch 3.79→15.04，但搜索 0.281→0.442 s。不要将减少调用次数等同已证实 GPU 提速。

后续：目标设备运行基准，使用 `--config configs/tiny.yaml --device mps --board-size 9 --simulations 200 --workers 4`，调 batch/等待/worker。全套测试 `.venv/bin/python -m pytest -q`；native 改动后 `.venv/bin/python setup.py build_ext --inplace`。没有长期棋力结论。
