# 接口与数据

## 2026-10-21 策略与日志增量

- 以下新约定优先于旧策略 target 描述。Config/checkpoint schema 不变；selfplay/loop runtime 参数 selfplay_policy 默认 katago，可选 legacy。
- katago 温度 .15+.60×.5^(ply/board_size)，根噪声总 concentration 10.83、半均匀半 log-policy；混合权重取 config.search.dirichlet_epsilon。
- Search.run 增加 shaped_noise/policy_correction 可选开关，均默认 false。结果保持 visits/policy，另含 selection_weights/selection_policy/action_values/root_priors；回顾性修正不改树统计。selfplay katago 开启两者，legacy 关闭；训练 label 为 selection_policy，采样温度只控制实际着法。
- replay metadata 的 exploration(version=1) 记录完整策略设置，policy_correction_stats 记录平均移除质量及原始/目标单访问点质量；旧 replay 标签不转换。评估仍使用原始 visits 和原有温度，正式晋级协议不变。
- train 返回 metrics 新增 summary；每次尝试在 metrics_dir/steps-START-唯一后缀写 steps.jsonl 和成功 summary.json；独立 train 默认 OUTPUT_PARENT/training/OUTPUT_STEM。
- loop 在 root/training/iteration-NNNN 存各次日志，在评估前向 root/training-loss.csv 追加一行；完整 iteration JSON 的 training 字段引用 summary。CSV 行代表完成训练尝试，重试轮次可重复，不是唯一 iteration 键。
- summary schema_version=1：iteration/checkpoint/start_step/end_step/requested_steps/steps/positions/device/loss_weights/mean/min/max/last/metrics_path/summary_path/UTC 时间。分项未加权，total 已加权；grad_norm 是裁剪前值。
- 中断训练保留已写 steps.jsonl，未完成时无 completed summary；已完成训练即使评估中断也保留 CSV。无旧 loss 回填，不写入 checkpoint 配置；运行中的旧 parent 不热更新。

## 早期约定及后续修订

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
- 晋级：采用 `katago-gatekeeper-v1`；默认最多 200 盘，胜局 1 分、和棋/无结果 0.5 分，达到 `--required-candidate-win-prop`（默认 0.5，平分接受）或数学上已确定达到/无法达到时提前停止。评估从空盘开始，配对颜色、关闭根噪声并使用早期高温到后期低温的随机落子；Wilson 区间仅描述性输出。
- 自博弈可通过 CLI `--min-moves` 设置最少总手数；阈值前 C++ 搜索屏蔽 pass，若无合法非 pass 则允许 pass；评测不使用该限制。replay metadata 和 iteration 记录保存该值。
- 自博弈和评测可通过运行时 `--score-utility` 使用 `score_mean` 的小权重参与 PUCT utility；不进入 checkpoint config，因此现有 9x9 checkpoint 可以直接续跑。默认 CLI 值 0.05，设置 0 可恢复 WDL-only 搜索。
- RPC `analyze` 返回网络形势报告（WDL、winrate、score mean/std、moves-left、ownership、likely-dead），`state` 在终局返回规则裁定报告；网络预测不会覆盖正式规则结果。

2026-10-15：checkpoint schema v1 新增 training_origin=random_init_selfplay；缺失/非该值的工件在自博弈与续训时拒绝，在 GTP/分析可读。replay 仅采样 source=selfplay，否则 fail fast。来源是工程追踪记录，不是外部文件认证。

## 2026-10-20 搜索/推理协议

- Search.run 默认仍为新树；reuse=True 配合 advance(action)/reset() 使用持久树。Python search 的 session 参数可选，自博弈和评测自动使用。
- simulations 表示输出根子边访问总数；new_simulations + reused_simulations = simulations。提升节点自身首次网络评估不计入子边预算；已有访问超过新预算时原样保留。
- --inference-batch-size 默认 256、--inference-wait-ms 默认 2；只控制多 worker 的集中推理，不加入 Config/checkpoint。search.batch_size 仍为单树批量。
- 请求/响应含 client、model key、请求序号，按原行序回填；模型失败传播给等待客户端。推理服务生命周期包住进程池，gatekeeper 丢弃局也能完成和退出。
- 自博弈 inference_summary、评测结果 inference 按模型报告 requests/calls/positions/max_batch_size/mean_batch_size/cross_client_batches。评测统计含已启动但未计分的对局工作量。
- replay metadata 与 iteration JSON 添加 tree_reuse，replay/play_settings 的 search_budget 为 retained-plus-new-edge-visits。无已有权重/replay 迁移。

## 2026-10-20 评估记录协议

- evaluate/evaluate_parallel 新增可选 sgf_dir（Python API 默认 None）、temperature（默认 None）和双方显示名称；Config/checkpoint schema 不变。
- sgf_dir 下使用唯一 match-SEED-* 子目录。evaluation_start 与返回结果含 sgf_dir，evaluation_game 含 sgf_path；仅父进程计分的结果会导出，SGF 原子替换，不写 replay。
- 默认 gatekeeper 选点仍为 visit-temperature-sampling，0.5→0.2、无根噪声。temperature=0 为 visit-argmax；固定温度仅独立诊断，不能启用 required_candidate_win_prop。诊断开始日志协议为 diagnostic-fixed-temperature-v1。
- SGF PB/PW 为模型身份，C 根注释为 JSON 元数据，逐手 C 为温度/首选/实际访问数。并列最高访问次数也记 chosen is top=True。
- 不恢复已结束但未记录的历史对局。运行中的旧父进程不会加载新代码，重启后生效；并行 worker 热重载不受支持。
