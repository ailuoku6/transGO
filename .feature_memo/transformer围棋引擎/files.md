# 文件

## 2026-10-22 清理脚本

- `scripts/cleanup_old_artifacts.py`：滚动时间筛选、预览/删除和权重保护，独立标准库脚本。
- `tests/test_cleanup_old_artifacts.py`：10 项临时文件与安全边界测试；`README.md`：清理命令及运行范围。

## 2026-10-21 最新修改

- `transformer_go/exploration.py`：katago/legacy profile、温度与审计元数据。
- `cpp/native.cpp`、`transformer_go/search.py`：塑形噪声与回顾性修正；native 已重编。
- `transformer_go/selfplay.py`、`actors.py`、`pipeline.py`、`cli.py`：策略透传、软训练目标及运行时选择。
- `transformer_go/training_metrics.py`、`train.py`：逐步日志、原子汇总、追加 CSV；pipeline 记录每轮汇总，CLI train 可选 metrics-dir。
- `tests/test_exploration.py`、`test_training_metrics.py`：新增策略和 loss 回归；`test_native.py`、`test_pipeline.py`、`test_gating.py` 扩展 native/续训/CLI 验证。
- `README.md`、`IMPLEMENTATION.md`：探索说明、loss 路径、兼容/重启要求及验证范围。早期评估 SGF 和诊断改动保留，均未代替用户提交。

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

## 2026-10-20 修改与状态修正

- 当前目录已有 Git 仓库，旧文档“没有 Git 仓库”已过时；本轮在已有工作区编辑，未提交。
- cpp/native.cpp：持久 Search 树、完整着法来源校验、预算与噪声处理。
- transformer_go/inference.py：新增集中推理服务、同步进程客户端、分模型合批、统计与异常处理。
- transformer_go/search.py、selfplay.py、evaluate.py、actors.py：数组推理、每局会话、CPU worker 接入。
- transformer_go/cli.py、pipeline.py：运行时合批上限/等待时间透传。
- tests/test_tree_reuse.py、test_inference.py：树复用与集中推理专项；test_gating.py、test_min_moves.py 扩展兼容/CLI 验证。
- benchmarks/search_optimizations.py：固定动作序列的树复用/集中推理对照。
- README.md、IMPLEMENTATION.md：新架构、参数、验证数据与 CPU/GPU 边界。

## 2026-10-20 评估棋谱修改

- transformer_go/evaluate.py：携带真实动作/逐手注释；父进程计分后写 SGF；独立固定温度诊断。
- transformer_go/sgf.py：向后兼容扩展双方名称、根/逐手注释、UTF-8 与结果覆盖。
- transformer_go/pipeline.py、cli.py：自动输出目录、模型哈希、观棋参数。
- tests/test_evaluation_sgf.py：新增导出/转义/RNG/温度/真实串并行 CLI 验证；test_gating.py、test_inference.py、test_min_moves.py 扩展提前停止和 loop 续跑验证。
- README.md：棋谱位置、观棋命令与重启要求。此前优化代码现已提交；本轮新增导出功能尚未提交。
