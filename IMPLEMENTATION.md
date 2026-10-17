# 第一版实现状态

更新：2026-10-15。

| 模块 | 已实现 | 后续工作 |
|---|---|---|
| 规则 | KataGo `Board`/`BoardHistory`/`Rules` 原生核心、2~19 路、提子、四类 ko、禁自杀/自杀规则、button、双 pass 与 territory encore、Benson/pass-alive、独立生命/seki、死子归属、面积/territory 计分、历史特征、Zobrist | 与完整 KataGo 引擎的神经网络搜索、分布式协议和全部赛事级 GTP 行为整合 |
| Transformer | Conv stem、二维坐标特征、局部/全局注意力、Pre-LN、SwiGLU、policy/WDL/score mean+variance/ownership/moves heads | 相对位置偏置、真正窗口分块、位置编码消融 |
| Metal | MPS 自动检测、显式 mps、float32 训练与批量推理、设备同步 benchmark | profiler、实际 batch 吞吐调优、可选 autocast |
| MCTS | C++ PUCT、批量叶子、虚拟访问、根 Dirichlet、温度落子、精确预算、合法 mask、自博弈最少手数 pass 约束、可选 score utility | 跨落子树复用、多线程共享树、FPU、动态 c_puct、playout-cap randomization |
| 数据 | 原子 NPZ、自博弈 SGF 导出/查看、终局标签、视角/D8 增强、近期 replay window、校验 | streaming shards、优先回放、自动隔离坏 shard、多规则数据 |
| 训练 | 仅自博弈 RL trainer、数据来源检查、checkpoint 来源记录、masked policy CE、WDL CE、score Gaussian NLL、ownership CE、moves Huber、AdamW、裁剪、恢复 RNG | warmup/scheduler、验证集、DDP、训练监控 |
| 自博弈 | 单机 actor、spawn 多进程、worker 内复用模型、父进程转发 `game_completed`、模型 SHA256、截断局丢弃 | 集中推理服务、远程 actor 队列、自博弈开局多样化、规则/贴目随机化 |
| 评测 | KataGo gatekeeper match points、胜平负/无结果半分、数学可确定时提前接受/拒绝、配对颜色、空盘、温度衰减、描述性 Wilson 区间与 Elo、best/learner 分离、正式裁定与网络形势报告分离、spawn 批次并行评估、父进程按种子顺序汇总 `evaluation_game` | 按开局对统计、外部 KataGo/GNU Go 比赛、固定战术基准、多棋盘/多规则 gatekeeper |
| 接口 | GTP 2、本地逐行 JSON-RPC 2.0、CLI；`final_status_list alive/dead/seki` | HTTP 服务、分析可视化、带版本的热切换 |

`configs/smoke.yaml` 是工程验证配置；`configs/tiny.yaml` 是正式 9x9 起点。13x13/19x19 网络与规则尺寸已通过功能验证，尚未完成长期自博弈或棋力评估。高性能实现第一步使用 C++ 消除 Python 树遍历成本，并缓存节点合法动作；当前完整状态复制与精确超级劫历史复制仍可能成为大预算瓶颈，应以 profiler 结果决定优化次序。

与最初研究方案的明确调整：macOS 用 MPS 而非 TensorRT；v1 采用 float32 而非默认 BF16；训练 uncertainty 使用 Gaussian NLL，避免只输出而不训练方差头；ownership 为三分类，正确保留中立点；按用户要求取消人类棋谱预训练；训练只接受自博弈数据，从随机网络启动。以上均由接口与数值正确性需要决定。

最终完整测试：规则核心切换到 KataGo BoardHistory 后执行 `.venv/bin/python -m pytest -q`，83 passed、4 skipped；跳过原因仅为当前运行环境检测不到 Metal 设备。另用 KataGo 的 9x9 area fixture 和 7x7 dead throw-in fixture 对照了 area/dead/seki 输出，并测试了 gatekeeper 的 match-point 边界、提前停止、平分接受和超手数强制计分。当前 `runs/zero-9x9/replay` 内 2,818 局旧 replay 已生成独立标签缓存，共 258,793 个训练局面；原 NPZ/SGF 文件及 `best.pt`/`learner.pt` 均保留。setuptools 与 CMake 两种原生扩展构建入口均已验证。

2026-10-15 路线更新：删除 SGF 转训练功能；trainer 拒绝非 selfplay/未知来源，训练与 actor 拒绝没有 random_init_selfplay 标记的 checkpoint。旧 checkpoint 可继续用于 GTP/分析和同一配置的续训；新规则标注通过 `adjudication_version` 写入新 shard，原有 replay 不被覆盖。来源标记是工程追踪，不是外部文件认证。

纯自博弈更新验证：完整测试 34 passed（包含 Metal）；新的 `runs/zero-only-smoke` 从随机权重启动，在 MPS 完成两局自博弈、3 步训练与候选评测。
