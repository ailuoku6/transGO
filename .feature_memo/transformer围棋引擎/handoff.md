# 接续

从 `index.md`、`README.md`、`IMPLEMENTATION.md` 开始。用户已授权按方案编码，指定 Python/PyTorch + C++ + Metal。

最新功能：`loop`、`selfplay`（含多 actor）支持 `--min-moves N`。阈值前 C++ PUCT 屏蔽 pass，棋盘没有合法非 pass 时仍可 pass；评测不受约束。参数不进入 checkpoint config，可在同一目录续跑时修改，值会写入 replay metadata/迭代 JSON。专项测试为 `tests/test_min_moves.py`，最近 12 passed。

另有运行时 `--score-utility`（CLI 默认 0.05）：把已有 `score_mean` 以 `weight*tanh(score)` 加入 PUCT 叶价值；不改模型/checkpoint 配置，现有 9x9 run 可直接续训。设为 0 可关闭。

`transformer_go/judgment.py` 已加入正式裁定/形势预测分离：`adjudicate` 只取 native 终局结果，`position_report` 暴露 WDL、score mean/std、ownership、likely-dead、moves-left。RPC `analyze` 返回模型报告，终局 `state` 返回 adjudication。当前仍是中国面积规则 v1，尚未实现 KataGo 全部规则、自动死子协商和完整 score/ownership 搜索 utility。

现有 v0.1 可以从 C++ 规则、批量 PUCT 一直跑到自博弈、训练、评测和晋级。Metal 可真实运行，但工具沙箱可能隐藏 GPU；环境外 `doctor` 返回 mps。不要仅凭沙箱检测认定 Mac 没有 Metal。

运行 `.venv/bin/python -m pytest -q`；改 native 后 `.venv/bin/python setup.py build_ext --inplace`。真实 Metal 项 `.venv/bin/python -m pytest -q tests/test_mps.py`。当前 CPU 24 passed，Metal 4 passed。

`configs/smoke.yaml` 是 3x3，仅验证工程。`tiny.yaml` 是 9x9 约482万参数，200模拟。`runs/closed-loop-v2` 有两轮演示结果，`runs/metal/model.pt` 是微量训练工件，没有棋力保证。

后续优先随机网络自博弈冷启动与固定预算评测，再 profiling 优化气数、状态复制、树复用、中心推理。SPRT、优先回放、DDP、HTTP、完整多规则未实现。不要把研究方案描述为全部完工。

- 最终统一验证：在可访问 Metal 的环境运行 `.venv/bin/python -m pytest -q`，28 passed（2.75 秒），包含全部 CPU 与 Metal 用例。

用户最新约束：不使用人类棋谱，随机初始化，仅自身 MCTS 对局训练。新 checkpoint 记录 training_origin=random_init_selfplay，旧无标记工件不可续入 Zero 训练；新目录启动，不删除旧工件。SGF 仅导出与查看。

- 2026-10-15 纯自博弈更新最终验证：完整 CPU+Metal 测试 34 passed（2.78 秒）；新的 runs/zero-only-smoke 在 MPS 上完成 2 局、39 个训练局面、3 步训练和 2 局候选评测。无长期棋力结论。
