# 接口与数据

- 规则：Chinese 风格面积、禁自杀、位置超级劫，pass 豁免、双 pass 终局；不自动判死棋。
- 动作 `row*N+col`，pass 为 `N*N`。
- 输入 21 通道：8 个历史局面双颜色 + 合法点 + 最后一步 + 黑行棋标志 + 当前方贴目/面积 + 手数/面积。
- 策略 target 为访问分布；WDL 类 `0/1/2=负/和/胜`，归属类 `0/1/2=对方/中立/当前方`。
- 价值与目数归属均采用当前玩家视角；score、moves_left 按面积归一化。
- NPZ schema v1 字段：features/legal/policy/value/score/ownership/moves_left/aux_mask/metadata，allow_pickle=False。
- 终局前截断对局丢弃；不接收外部 SGF 训练样本。
- Checkpoint schema v1：模型配置、权重、优化器、step、torch/NumPy RNG、软件版本。torch.load 使用 weights_only=True。
- GTP 2：编号、坐标跳 I 列、play/genmove/undo/final_score 等。
- 本地逐行 JSON-RPC 2.0：new_game/play/state/analyze，未实现 HTTP。
- 晋级：无截断、至少 40 盘、胜率>=55%、近似 Wilson 95% 下界>50%；后续用按开局对统计/SPRT。
- 自博弈可通过 CLI `--min-moves` 设置最少总手数；阈值前 C++ 搜索屏蔽 pass，若无合法非 pass 则允许 pass；评测不使用该限制。replay metadata 和 iteration 记录保存该值。
- 自博弈和评测可通过运行时 `--score-utility` 使用 `score_mean` 的小权重参与 PUCT utility；不进入 checkpoint config，因此现有 9x9 checkpoint 可以直接续跑。默认 CLI 值 0.05，设置 0 可恢复 WDL-only 搜索。
- RPC `analyze` 返回网络形势报告（WDL、winrate、score mean/std、moves-left、ownership、likely-dead），`state` 在终局返回规则裁定报告；网络预测不会覆盖正式规则结果。

2026-10-15：checkpoint schema v1 新增 training_origin=random_init_selfplay；缺失/非该值的工件在自博弈与续训时拒绝，在 GTP/分析可读。replay 仅采样 source=selfplay，否则 fail fast。来源是工程追踪记录，不是外部文件认证。
