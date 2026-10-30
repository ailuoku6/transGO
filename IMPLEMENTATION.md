# 第一版实现状态

更新：2026-10-25。

| 模块 | 已实现 | 后续工作 |
|---|---|---|
| 规则 | KataGo `Board`/`BoardHistory`/`Rules` 原生核心、2~19 路、提子、四类 ko、禁自杀/自杀规则、button、双 pass 与 territory encore、Benson/pass-alive、独立生命/seki、死子归属、面积/territory 计分、历史特征、Zobrist | 与完整 KataGo 引擎的神经网络搜索、分布式协议和全部赛事级 GTP 行为整合 |
| Transformer | Conv stem、二维坐标特征、局部/全局注意力、Pre-LN、SwiGLU、policy/WDL/score mean+variance/ownership/moves heads | 相对位置偏置、真正窗口分块、位置编码消融 |
| Metal | MPS 自动检测、显式 mps、float32 训练与批量推理、设备同步 benchmark | profiler、实际 batch 吞吐调优、可选 autocast |
| Neural Engine | 可选 Core ML CPU_AND_NE 推理、FP16 固定形状导出、全部预测头、合法 mask、按权重/棋盘/batch 缓存、集中合批与 loop/对弈接口接入、静态设备计划诊断 | 多局面/长时间吞吐与功耗基准；训练仍由 PyTorch/MPS 执行 |
| MCTS | C++ PUCT、批量叶子、虚拟访问、根 Dirichlet、温度落子、精确目标预算、合法 mask、自博弈最少手数 pass 约束、可选 score utility、每局完整历史校验的落子子树复用 | 多线程共享树、FPU、动态 c_puct、playout-cap randomization |
| 数据 | 原子 NPZ、自博弈 SGF 导出/查看、终局标签、视角/D8 增强、近期 replay window、校验 | streaming shards、优先回放、自动隔离坏 shard、多规则数据 |
| 训练 | 仅自博弈 RL trainer、数据来源检查、checkpoint 来源记录、masked policy CE 与 KataGo 风格软策略目标、平滑 WDL/ownership CE、中立 ownership 软目标、Gaussian/Student-t 混合 score NLL、moves Huber、AdamW、裁剪、恢复 RNG | warmup/scheduler、独立验证集、DDP、多随机种子棋力评测 |
| 自博弈 | 单机 actor、spawn CPU 搜索进程、父进程单网络集中合批推理、跨落子子树复用、父进程转发 `game_completed` 与推理批次统计、模型 SHA256、截断局丢弃 | 远程 actor 队列、自博弈开局多样化、规则/贴目随机化、目标设备吞吐调优 |
| 评测 | KataGo gatekeeper match points、胜平负/无结果半分、数学可确定时提前接受/拒绝、配对颜色、空盘、温度衰减、描述性 Wilson 区间与 Elo、best/learner 分离、正式裁定与网络形势报告分离、spawn 并行评估、candidate/baseline 按模型分别合批和独立搜索树、两树随实战每手同步推进、父进程按种子顺序汇总 `evaluation_game` | 按开局对统计、外部 KataGo/GNU Go 比赛、固定战术基准、多棋盘/多规则 gatekeeper、目标 GPU 基准调优 |
| 接口 | GTP 2、本地逐行 JSON-RPC 2.0、CLI；`final_status_list alive/dead/seki` | HTTP 服务、分析可视化、带版本的热切换 |

`configs/smoke.yaml` 是工程验证配置；`configs/tiny.yaml` 是正式 9x9 起点。13x13/19x19 网络与规则尺寸已通过功能验证，尚未完成长期自博弈或棋力评估。高性能实现第一步使用 C++ 消除 Python 树遍历成本，并缓存节点合法动作；当前完整状态复制与精确超级劫历史复制仍可能成为大预算瓶颈，应以 profiler 结果决定优化次序。

并行推理基准入口为 `benchmarks/search_optimizations.py`；它比较 fresh tree 和复用树，并比较每 worker 独立 forward 与父进程跨 worker 合批，报告准确调用数/吞吐与 merge rate。CPU 低负载有 IPC 开销；必须在目标 MPS/CUDA 设备和正式棋盘/模型上测量吞吐。

2026-10-21 增量：默认自博弈使用 `katago` profile，含 0.75→0.15 平滑温度、policy-shaped Dirichlet 和适配简化 PUCT 的回顾性权重修正。实际树访问数不变，训练目标取修正权重归一化；`legacy` 可切回原方案。未实现 LCB、分叉局、廉价搜索等完整 KataGo 训练算法。每次训练保存全部 step 的 JSONL 和原子 summary，loop 在评估前追加 `training-loss.csv`，重跑使用独立子目录；日志不进入 checkpoint schema。评估同时支持 SGF 和独立零温度观棋，详见 README。

2026-10-30：完成两种损失配置各累计十轮的 Core ML 自博弈/评估与 MPS 训练对照。新损失得分率 52.00%（400 局，描述性 95% Wilson 区间 47.11%–56.85%），证据不足以确认棋力提升；完整限制与颜色拆分见 `runs/ab-loss-coreml-20261029/continuation-10rounds/REPORT.md`。正式 `configs/tiny.yaml` 已切换为新损失参数。

历史工程验证（2026-10-21）：native 已重新编译，全套 138 passed、5 skipped（MPS 不可用）；临时真实 CLI 的并行 katago→串行 legacy 续跑两轮通过，4 局自博弈、4 局评估、训练 step 2→4、2 行 loss 记录。验证覆盖中断日志保留、续训不覆盖、原始访问预算/树复用和软策略训练目标。未修改运行中的长训练、已有权重或 replay。

与最初研究方案的明确调整：macOS 用 MPS 而非 TensorRT；v1 采用 float32 而非默认 BF16；训练 uncertainty 使用 Gaussian NLL，避免只输出而不训练方差头；ownership 为三分类，正确保留中立点；按用户要求取消人类棋谱预训练；训练只接受自博弈数据，从随机网络启动。以上均由接口与数值正确性需要决定。

最终完整测试：规则核心切换到 KataGo BoardHistory 后执行 `.venv/bin/python -m pytest -q`，83 passed、4 skipped；跳过原因仅为当前运行环境检测不到 Metal 设备。另用 KataGo 的 9x9 area fixture 和 7x7 dead throw-in fixture 对照了 area/dead/seki 输出，并测试了 gatekeeper 的 match-point 边界、提前停止、平分接受和超手数强制计分。当前 `runs/zero-9x9/replay` 内 2,818 局旧 replay 已生成独立标签缓存，共 258,793 个训练局面；原 NPZ/SGF 文件及 `best.pt`/`learner.pt` 均保留。setuptools 与 CMake 两种原生扩展构建入口均已验证。

2026-10-15 路线更新：删除 SGF 转训练功能；trainer 拒绝非 selfplay/未知来源，训练与 actor 拒绝没有 random_init_selfplay 标记的 checkpoint。旧 checkpoint 可继续用于 GTP/分析和同一配置的续训；新规则标注通过 `adjudication_version` 写入新 shard，原有 replay 不被覆盖。来源标记是工程追踪，不是外部文件认证。

纯自博弈更新验证：完整测试 34 passed（包含 Metal）；新的 `runs/zero-only-smoke` 从随机权重启动，在 MPS 完成两局自博弈、3 步训练与候选评测。

2026-10-20 两项搜索优化验证：原生扩展重新编译成功；全套 109 passed、5 skipped（当前工具沙箱无法使用 MPS，其中新增一项推理服务线程的 MPS 对照测试）。真实 spawn 进程验证了跨 worker 合批、模型隔离、错误传播、丢弃评测局后的服务退出，以及集中自博弈→训练→集中评测的完整 CLI 闭环；checkpoint Config 保持一致。固定 5x5/8 手/64 模拟的 smoke 网络 CPU 对照：树复用的网络局面数 518→489、调用 136→128；4 worker 集中推理调用 548→138、平均 batch 3.79→15.04，但搜索时间 0.281→0.442 秒，不能声称该 CPU 小负载有提速。正式 MPS/CUDA 吞吐尚待目标设备测量。
