# 文件

- `cpp/native.cpp`：规则和批量 PUCT；修改后需重编译。
- `transformer_go/model.py`、`runtime.py`：Transformer 与 MPS。
- `transformer_go/search.py`：Python 到 C++ evaluator 回调接口。
- `transformer_go/data.py`、`sgf.py`、`selfplay.py`：数据与终局标签。
- `transformer_go/train.py`、`checkpoint.py`：损失和恢复。
- `transformer_go/evaluate.py`、`pipeline.py`：评测与晋级循环。
- `transformer_go/actors.py`、`gtp.py`、`rpc.py`、`cli.py`：并行和交互。
- `tests/test_min_moves.py`：自博弈最少手数、搜索 pass 屏蔽、CLI/并行 actor 传参回归测试。
- `cpp/native.cpp`、`transformer_go/search.py`：可选 score utility，将已有 score head 接入 PUCT；默认 0 时保持旧搜索行为。
- `transformer_go/judgment.py`：规则引擎正式裁定与网络形势预测的分离接口；`rpc.py` 暴露两者。
- `configs/smoke.yaml`：3x3 工程检查；`configs/tiny.yaml`：9x9 正式起点。
- `tests/`：规则参考实现对照、闭环、接口、Metal。
- `README.md`、`IMPLEMENTATION.md`：运行和范围；`requirements-dev.lock`：本机已验证版本。
- `runs/` 与 `.venv/` 均忽略；工作区当前没有 Git 仓库。

2026-10-15 修改：data.py 来源限制；checkpoint.py 来源记录；train/cli/actors/pipeline 检查训练链；sgf.py 移除转换；test_zero_training.py 测试非 selfplay 拒绝与随机初始化闭环；README/TECHNICAL_PLAN/IMPLEMENTATION 改为纯自博弈路线。
