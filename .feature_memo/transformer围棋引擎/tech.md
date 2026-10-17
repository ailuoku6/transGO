# 技术

- Python 3.12 + PyTorch 2.14.1，项目 `.venv`；Apple Clang，C++17 + pybind11。
- Conv stem + 坐标编码 + 局部/全局 Transformer，SwiGLU，policy/WDL/score mean+variance/ownership/moves heads。
- C++ 原生规则，精确位置键防超级劫，确定性 Zobrist，节点缓存合法动作，单树批量叶子 + 预留/虚拟损失。
- `auto` 优先 MPS，其次 CUDA、CPU；MPS 默认 float32。当前沙箱隐藏 GPU，普通环境可用 MPS。
- replay 使用原子 NPZ、均匀近期窗口；spawn 多 actor 各有推理模型。loop 按轮执行，未实现远程异步集群。
- best 与 learner 分离，失败候选不清空 learner 进度；gatekeeper 记录 candidate/baseline SHA、评估种子、累计得分和停止原因。
- 注意力局部 mask 仍用 dense kernel；棋盘与超级劫历史仍完整复制，不能当作最终高性能实现。

2026-10-15：trainer 只接受 source=selfplay；自博弈/恢复训练检查 checkpoint 的 random_init_selfplay 来源。SGF 转换到训练入口已删除。Metal 和辅助目标仍保留，标签仅来自自博弈终局。
