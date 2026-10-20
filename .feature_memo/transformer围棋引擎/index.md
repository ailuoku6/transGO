# Transformer 围棋引擎

- 状态：第一版闭环已实现，研究路线继续迭代。
- 更新：2026-10-20。
- 当前重点：已完成跨局面集中推理与 MCTS 子树复用；下一步在目标 MPS 设备测量正式模型吞吐。

## 文档地图

- `../recording-preferences.md`：记录偏好，目前无额外项。
- `brief.md`：目标与边界。
- `tech.md`：架构和约束。
- `protocol.md`：数据、视角、接口。
- `files.md`：代码入口。
- `decisions.md`：已确定的技术取舍。
- `changes.md`：验证记录。
- `handoff.md`：继续开发入口。

## 最新事实

- 全新空目录已落地规则 -> 搜索 -> 自博弈 -> 训练 -> 评测 -> 晋级闭环。
- CPU 测试 24 项通过；4 项真实 Metal 专项测试另行通过。
- 实现范围和研究待办见 `IMPLEMENTATION.md`；启动命令见 `README.md`。

## 下一步

- 坚持随机初始化和纯自博弈，进行 9x9 冷启动与长周期棋力评测；不引入人类棋谱。
- Profiling 后优化状态复制、增量气数、树复用与中心推理服务。
- 增加 SPRT/按开局对统计、DDP、流式 replay 和优先回放。

## 未确认

- 数据来源已确定仅自博弈；长期算力预算和目标棋力尚未指定。

- 最终统一验证：在可访问 Metal 的环境运行 `.venv/bin/python -m pytest -q`，28 passed（2.75 秒），包含全部 CPU 与 Metal 用例。

- 2026-10-15 路线更新：用户明确要求 AlphaGo Zero 式纯自博弈；已删除 SGF 转训练，新增 replay/checkpoint 来源检查。

- 2026-10-15 纯自博弈更新最终验证：完整 CPU+Metal 测试 34 passed（2.78 秒）；新的 runs/zero-only-smoke 在 MPS 上完成 2 局、39 个训练局面、3 步训练和 2 局候选评测。无长期棋力结论。
- 2026-10-16：新增 `--min-moves`，12 个专项测试通过；默认值 0 保持原行为。
- 2026-10-17：已有 score head 接入可选 PUCT score utility；CLI 默认 0.05，不改 checkpoint config，可续接既有 9x9 run。

## 2026-10-20 最新状态（优先于早期记录）

- 集中推理服务与跨落子子树复用已完成，串行兼容；CLI/loop 支持运行时合批参数，checkpoint 配置不变。
- 全套 109 passed、5 skipped（当前沙箱 MPS 不可用），真实 spawn/完整并行闭环通过。
- CPU smoke 集中推理调用约减少 75%，但低负载搜索墙钟时间上升；GPU 加速幅度未验证。详见 changes.md、protocol.md 和 benchmarks/search_optimizations.py。
- 下一步：正式 9x9 模型的 MPS 2/4/8 worker、batch/等待参数基准及长期棋力对照。
