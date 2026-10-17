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
