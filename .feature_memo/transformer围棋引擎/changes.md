# 实现日志

## 2026-10-16

- 新增 `--min-moves` 自博弈参数：阈值前由 C++ PUCT 屏蔽 pass，只有棋盘无合法非 pass 时例外放行；评测保持标准规则。
- replay metadata、`game_completed` 事件和 iteration JSON 记录 `min_moves` 与异常提前 pass 计数。
- 新增 `tests/test_min_moves.py`，覆盖树内 pass 约束、终局例外、CLI 续跑、并行 actor 和参数校验。
- 验证：`.venv/bin/python -m pytest -q tests/test_min_moves.py`，12 passed。
- 新增可选 score utility：C++ 搜索接受 evaluator 的第三个 score 数组，按 `value + weight*tanh(score_mean)` 回传；CLI 默认 0.05，参数不写入 checkpoint config，现有 run 可续接。
- 验证：全套测试 43 passed、4 skipped；临时 smoke loop 使用 `score_utility=0.05` 完成两轮并从 step 1 续到 step 2。
- 新增 `judgment.py`：终局 `adjudicate` 只使用规则引擎；网络 `position_report` 输出 WDL、score 不确定性、ownership 和 likely-dead 形势预测。RPC `analyze`/终局 `state` 分别暴露预测和正式裁定。
- 验证：全套测试 44 passed、4 skipped。

## 2026-10-15

- 从空目录建立 setuptools/pybind11 工程、可选 CMake、CLI 和 Makefile。
- 实现 C++ 规则、历史编码、批量 PUCT；PyTorch 多头 Transformer、MPS。
- 实现 selfplay/SL/RL/replay/checkpoint、评测晋级、spawn actors、GTP、JSON-RPC。
- CPU：24 passed，4 个 Metal 项在受限环境下跳过；包含独立规则 oracle 随机对照和精确 CPU 续训一致性。
- Metal：4 passed，覆盖 9/13/19 路前向反向和自博弈训练后恢复优化器。
- 完成 CPU 与 Metal 真实自博弈各 2 局、Metal 5 步训练、CPU 两轮 candidate 评测闭环、多 actor 两局。
- 9x9 tiny 4,821,898 参数、200 simulations 单次测量：CPU约0.290秒、MPS约0.839秒；是小 batch 功能性测量，不能推广成稳定吞吐或加速结论。
- 无长期棋力训练。后续研究待办详见 `IMPLEMENTATION.md`。

- 最终统一验证：在可访问 Metal 的环境运行 `.venv/bin/python -m pytest -q`，28 passed（2.75 秒），包含全部 CPU 与 Metal 用例。

- 2026-10-15 纯自博弈更新最终验证：完整 CPU+Metal 测试 34 passed（2.78 秒）；新的 runs/zero-only-smoke 在 MPS 上完成 2 局、39 个训练局面、3 步训练和 2 局候选评测。无长期棋力结论。

## 2026-10-20 完成集中批量推理和 MCTS 子树复用

- 两项优化已接入串行/并行自博弈与评测：串行复用树，多 worker 增加父进程集中推理；候选/基准独立树且每手同步推进。
- `.venv/bin/python setup.py build_ext --inplace` 成功（第三方代码既有警告）；`.venv/bin/python -m pytest -q`：109 passed、5 skipped，10.69 s，跳过项均因当前沙箱 MPS 不可用。
- 真实 CLI 并行闭环：2 worker 自博弈 2 局、训练 1 step、2 worker 评测 2 局；参数 16/1 ms 生效，保存 Config 与原始配置一致。临时运行目录位于系统 tmp，未改已有 runs/zero-9x9。
- `.venv/bin/python -m benchmarks.search_optimizations`：CPU、smoke 网络、5x5、8 手、64 模拟。复用树 NN 位置 518→489、forward 136→128，约 0.110→0.100 s（单次微基准）。4 worker 独立 forward vs 集中：548→138 次，2076 位置一致，平均 batch 3.79→15.04；搜索 0.281→0.442 s。小网络 CPU 未加速，GPU 速度尚未测量。
- 正式棋力和目标 MPS/CUDA 吞吐不属于本次已验证结论。
