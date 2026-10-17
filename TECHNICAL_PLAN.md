# Transformer + 蒙特卡洛树搜索强化学习围棋程序技术方案

> 2026-10-15 实现决定：第一版使用 Python + PyTorch 训练、C++17 规则和搜索；macOS 使用 PyTorch MPS/Metal 进行训练与批量推理。当前默认 float32，按实测再考虑混合精度。已落地的范围与后续项见 `IMPLEMENTATION.md`，运行方式见 `README.md`；本文其余部分为完整研究路线，不能视为全部已实现。

## 1. 目标与范围

目标是实现一个 9x9、13x13、19x19 通用的围棋程序，训练路径采用 AlphaGo Zero 式随机初始化、纯自博弈，并吸收 KataGo 的实用增强：

- Transformer 策略价值网络，输出落子策略、胜率、最终目数、棋盘归属和不确定性。
- 基于 PUCT 的蒙特卡洛树搜索（MCTS），支持批量叶子评估、虚拟损失、根节点噪声和温度采样。
- 网络从随机初始化开始，仅通过自博弈强化学习改进，不使用人类棋谱或人类棋谱预训练模型。
- 自博弈生成训练数据；SGF 用于导出对局，GTP 和 JSON-RPC 用于命令行分析与 GUI/服务器接入。
- 首个可交付目标不是立即达到 KataGo 强度，而是先完成闭环：合法落子 -> 搜索 -> 自博弈 -> 训练 -> 新模型评估 -> 模型晋级。

不建议把“复现阿法狗”理解为复刻原论文中的每个工程细节。AlphaGo 使用了多种人工棋谱、多个网络和复杂的分布式硬件；本方案采用 AlphaZero 的单网络闭环作为主干，并加入 KataGo 风格的辅助目标，以降低工程复杂度并提高样本效率。2026-10-15 用户明确要求采用 AlphaGo Zero 式纯自博弈，已取消人类棋谱监督预训练路线。

## 2. 总体架构

```text
随机初始化网络 θ₀ ──> 模型仓库（初始 best/learner）
模型仓库 <── 评估/晋级 <── 训练器 <── 回放缓冲区 <── 自博弈集群
   │                         ▲                         │
   └────────── 推理服务 <────┴──── MCTS 搜索 <──────────┘
```

建议拆成以下进程和包：

```text
transformerGo/
  go/                 # 棋盘、规则、Zobrist、SGF、GTP
  net/                # Transformer trunk 和各输出 head
  search/             # MCTS、批量推理、根并行、树并行
  selfplay/           # 自博弈 worker、随机开局、数据封装
  train/              # 自博弈 RL 训练、优先回放、checkpoint
  eval/               # 基准、对局、Elo、回归测试
  serving/            # GTP、HTTP/JSON-RPC、分析接口
  configs/             # YAML/JSON 超参数
  scripts/             # 数据转换、启动集群、导出模型
  tests/
```

推荐 Python + PyTorch 作为第一版训练栈，C++ 或 Rust 作为棋盘规则和高性能搜索模块；原型期也可以全部 Python，待 profiler 证明瓶颈后再迁移。推理部署使用 TorchScript/ONNX 或 TensorRT，训练使用 BF16/FP16 混合精度。

## 3. 棋盘与规则引擎

### 3.1 规则边界

第一版固定中国规则、19x19 棋盘、贴目 7.5、无让子；同时让状态结构支持日本规则、不同贴目、让子和棋盘尺寸。必须明确处理：自杀禁着、打劫、超级劫、终局停着、提子、数目和认输。

规则引擎是训练正确性的根基，优先级高于网络结构。每一步都应经过纯函数式 `apply(action) -> next_state` 或可回滚的高性能实现验证。为每个状态维护：

- 黑白棋子 bitboard/数组；
- 当前玩家、连续停着数、贴目和规则类型；
- 劫状态（禁止的单点或超级劫哈希集合）；
- Zobrist hash 和局部气数缓存；
- 终局后的双方得分和胜负标签。

### 3.2 动作空间

将动作编码为 `row * board_size + col`，再加一个 `pass`，因此 19x19 是 362 个动作。策略 head 输出固定大小 `N*N+1`，由规则引擎提供 legal mask。训练和搜索都必须在 softmax 前将非法动作设为 `-inf`，并单独监控“非法策略质量”。

### 3.3 编码与对称

输入采用以当前玩家为视角的平面表示，保证颜色交换和轮流落子的一致性。基础平面建议：

1. 当前玩家棋子；
2. 对手棋子；
3. 最近一步；
4. 劫禁着点；
5. 空点/合法点；
6. 当前玩家平面；
7. 规则和贴目广播平面；
8. 最近 8~16 个历史局面的重复堆叠（可选）。

为避免模型把棋盘方向当成先验，训练样本在线随机使用 8 个 dihedral 对称变换；动作标签、最近一步和归属标签同步变换。

## 4. Transformer 策略价值网络

### 4.1 主干设计

直接对 361 个点做全局 self-attention 在计算上可行，但对局部气、连接和眼形不够高效。建议采用“局部卷积 stem + 位置编码 + 局部/全局混合 Transformer”：

```text
输入平面 [B, C, N, N]
  -> 3x3 Conv stem, channels=d_model
  -> N*N 个交错位置 token + pass token
  -> 12~24 个 Transformer block
  -> 2~4 个全局 token / mean pooling
  -> policy/value/score/ownership/uncertainty heads
```

每个 block 可以是：

```text
x = x + LocalWindowAttention(LN(x))
x = x + MLP(LN(x))
x = x + GlobalAttention(LN(x), global_tokens)
```

局部窗口建议 5x5 或 7x7，并每 2~4 层插入一次全局注意力。棋盘大小固定时，使用可学习 2D 相对位置偏置；支持多棋盘尺寸时使用二维正弦位置编码或插值相对偏置。`pass` token 不参与二维位置编码，但需要当前玩家、阶段和规则 embedding。

### 4.2 推荐规模

从小到大训练三档模型，避免一开始消耗过多算力：

| 配置 | d_model | 层数 | 头数 | 参数量（约） | 用途 |
|---|---:|---:|---:|---:|---|
| tiny | 192 | 8 | 6 | 3~5M | 规则、搜索、闭环验收 |
| base | 256 | 12 | 8 | 10~20M | 9x9/13x13、早期 19x19 |
| strong | 384 | 20~24 | 12 | 40~80M | 19x19 主力模型 |

每个模型使用 Pre-LN、RMSNorm 或 LayerNorm、SwiGLU/GEGLU MLP，dropout 设为 0~0.1。强化学习阶段通常关闭 dropout，避免 MCTS 评估噪声过大。

### 4.3 输出 heads

共享 trunk 后配置以下 head：

- `policy_logits`: `[B, N*N+1]`，预测专家/搜索改进后的落子分布。
- `value`: `[B, 3]`，分别预测当前玩家胜、和、负的概率；实际胜率为 `p_win + 0.5*p_draw`。
- `score_mean`, `score_logvar`: 最终目数均值和方差。
- `ownership`: `[B, N, N]`，预测终局每个点属于黑/白/中立的概率或期望归属。
- `moves_left`: 预测距离终局的剩余步数，辅助长局 credit assignment。

KataGo 风格的 score/ownership/moves-left 头不是为了直接替代搜索，而是让网络学习“赢多少、哪里赢、还要下多久”，对中盘判断、贴目和读秒尤其有帮助。

### 4.4 前向接口

```python
outputs = net(
    features,                 # [B, C, N, N]
    legal_mask=legal_mask,    # [B, N*N+1]
    rule_embedding=rules,
)
# outputs.policy_logits, value, score_mean, score_logvar,
# outputs.ownership, moves_left
```

网络不要在 forward 内静默改变棋盘状态；合法性由规则引擎提供，便于单元测试和部署复用。

## 5. MCTS 设计

### 5.1 节点统计量

每个边保存：`N(s,a)` 访问次数、`W(s,a)` 累计价值、`Q=W/N`、先验 `P(s,a)`、合法性和子节点指针。价值统一从“边上行动方”或“根玩家”视角存储，代码中只能选一种并写成不变量，避免符号错误。

选择公式使用 PUCT：

```text
U(s,a) = c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
score(s,a) = Q(s,a) + U(s,a)
```

`c_puct` 采用随根访问数缓慢变化的 schedule，并在 root 与非 root 使用不同系数。每次扩展用网络得到 policy/value/辅助头，叶子 value 反向回传并对当前玩家换符号。

### 5.2 AlphaZero/KataGo 风格增强

- 根节点 Dirichlet 噪声：`P'=(1-epsilon)P+epsilon*Dir(alpha)`；19x19 可从 `alpha=0.03~0.15` 起调。
- 温度采样：开局前 `temperature=1.0` 按 visit count 采样，过渡到 0.1，后期贪心；评估局始终贪心。
- 虚拟损失：并发搜索时临时增加访问并降低 Q，减少多个线程选择同一叶子。
- 批量叶子评估：收集 `batch_size=16~256` 个 leaf，一次 GPU forward；这是 GPU 利用率的主要来源。
- 根并行/树并行：多 worker 可共享模型但不共享 Python 树；单进程内对一棵树批量并行，跨进程采用根并行更容易稳定。
- 首若干手随机化：在自博弈中对 root policy 做受控采样，防止 replay buffer 过早坍缩。
- 可选 KataGo 式 playout cap randomization：训练数据使用不同搜索预算，迫使网络适应从低到高的搜索质量。
- 可选 score utility：根选择时优化 `win + lambda * normalized_score_margin`，lambda 需很小并固定评估基准。

### 5.3 搜索输出

每个根状态保存：

```text
state_features
legal_mask
visit_policy = visit_count / sum(visit_count)
root_value
chosen_action
root score/ownership 预测（可选）
```

强化学习的 policy target 应使用归一化访问次数，而不是只保存最终落子。访问数为零的非法动作始终保持零概率。

## 6. 训练流程

### 阶段 A：规则与搜索验收

先用随机网络和小搜索预算跑通 9x9。建立以下确定性测试：提子、打劫、连续 pass、终局数目、SGF 往返、8 种对称、MCTS visit 总数和价值符号。没有通过这些测试，不进入网络训练。

### 阶段 B：随机网络冷启动

初始化随机策略价值网络 θ₀，由同一个网络执黑/执白，通过 MCTS、根 Dirichlet 噪声和温度采样生成最初一批完整自博弈。无需人类棋谱、开局库或外部预训练模型。

收集 `(s_t, π_t, z)`：π_t 是搜索访问分布，z 是规则引擎在正常终局计算的真实胜负。仅当获得足够终局数据后开始更新 learner；截断局丢弃。记录停着比例、对局长度、策略熵、终局率和与历史随机模型的胜率，定位过早 pass、探索不足或价值头偏置。上述指标属于后续训练监控计划，当前版本已有终局与截断事件日志。

如冷启动陷入固定开局或快速 pass，优先调整搜索预算、探索参数和采样量，并用固定评测检验变化；不引入人类示范补救。9x9 用于工程验证，也可以独立在 19x19 从随机权重开始。

### 阶段 C：自博弈强化学习

每盘自博弈从当前 best checkpoint 采样，得到 `(s_t, pi_t, z, score, ownership, moves_left)`。核心损失：

```text
L = lambda_pi * CE(pi_t, softmax(policy_logits))
  + lambda_z * CE(z, value_distribution)
  + lambda_score * Huber(score_mean, final_score)
  + lambda_own * BCE(ownership, ownership_target)
  + lambda_mvl * Huber(moves_left, moves_left_target)
  + lambda_reg * ||theta||^2
```

建议初始权重：`lambda_pi=1.0`、`lambda_z=1.0`、`lambda_score=0.2~0.5`、`lambda_own=0.1~0.3`、`lambda_mvl=0.05~0.15`。用独立验证集和对局 Elo 调整，不要盲目追求辅助 head 的单项损失。

训练采用 replay window，只保留最近若干百万到数千万局面；按样本新鲜度和 TD/策略误差做轻量优先采样，同时保留均匀采样比例防止分布偏移。每轮训练后让 candidate 与 best 做 gatekeeper 对局：默认最多 200 局，胜局记 1 分、和棋/无结果记 0.5 分，达到要求的累计得分比例（默认 0.5，平分接受）才晋级；如果剩余对局已不可能达到门槛则提前拒绝，已经达到门槛则提前接受。评估从空盘开始，黑白配对、无根噪声并采用早期高温到后期低温的随机落子。

### 阶段 D：课程学习

推荐顺序：

1. 9x9，低 playout，验证闭环和价值符号；
2. 13x13，增加长连接、劫争和分支复杂度；
3. 19x19，独立随机初始化或迁移完全由自博弈训练的模型，先稳定生成数据；
4. 19x19 增加搜索预算、模型宽度和辅助目标；
5. 固定规则和评测协议，停止频繁改变数据分布。

## 7. 自博弈数据管线

### 7.1 Worker 协议

每个 worker 只负责：加载指定模型、维护规则状态、运行 MCTS、写出压缩样本。当前单机实现使用 Python `spawn` 进程池；一个 actor worker 在进程生命周期内复用模型和搜索 evaluator，多盘棋完成后再退出。评估 worker 同样各自加载 candidate/baseline，父进程按预先生成的颜色/随机种子顺序汇总 gatekeeper 结果，避免完成顺序改变晋级决策。中央服务负责模型版本和任务分配。样本文件建议使用 Parquet、LMDB 或 WebDataset shards，字段包括：

```text
model_version, board_size, rules, komi,
features_uint8, legal_mask_bitset, visit_policy_fp16,
value_target, score_target, ownership_uint8,
moves_left, move_index, game_id, symmetry_seed
```

不要在每条样本中重复保存完整浮点平面；平面可量化为 uint8，策略保存为 fp16 或稀疏 `(action, visits)`。

### 7.2 质量控制

每个 shard 写入前检查：

- 所选动作合法且 visit policy 总和为 1；
- 最终棋盘能被规则引擎复盘；
- 黑白视角和结果标签一致；
- score 与 ownership 能由终局复算；
- 没有 NaN、重复 game_id 或损坏压缩块。

训练器拒绝校验失败的 shard，并把错误样本写入隔离目录。训练和自博弈都记录随机种子、代码版本、模型 hash、规则配置和搜索参数，以便复现。

## 8. 分布式训练与部署

### 8.1 最小可运行配置

- 1 台有 1 张 16GB 以上 GPU 的机器；
- 4~16 个 CPU self-play worker；
- 9x9 `d_model=192`、8 层、每步 100~400 playout；
- PyTorch DDP 先不启用，优先确认单机闭环和数据吞吐。

### 8.2 扩展配置

```text
actor nodes -> object storage/replay shards -> learner GPUs
                    ^                         |
                    |                         v
                 evaluator <--- model registry/best pointer
```

模型注册至少保存：权重、配置、git commit、训练数据时间窗、评测 Elo、对局日志。模型发布采用原子版本号，actor 只在完整 checkpoint 校验通过后切换，避免半写入权重。

### 8.3 性能重点

1. 规则引擎的复制、合法动作和提子；
2. MCTS 节点分配与批量 leaf 收集；
3. 网络推理 batch size 和 GPU 利用率；
4. replay shard 压缩/读取；
5. DDP all-reduce 和 checkpoint I/O。

先用 profiler 测量，再决定 C++/Rust 扩展、CUDA kernel、TensorRT 或更大模型。不要在没有基准的情况下提前优化注意力实现。

## 9. 评测、基准与验收门槛

### 9.1 单元和性质测试

- 随机合法落子 10 万局无崩溃；
- 规则引擎与独立参考实现对比；
- 8 个对称变换的 policy/value 一致性；
- MCTS visit 总数、非法动作屏蔽和 value 符号测试；
- 相同 seed、相同 checkpoint、相同 playout 下结果可复现到允许的浮点误差。

### 9.2 强度评测

固定模型和规则，分开报告：

- 9x9/13x13/19x19 的 Elo 和胜率；
- playout 数从 32、128、512、2048 的 scaling 曲线；
- 对随机、纯 roll-out、传统启发式、历史 checkpoint 的胜率；
- 贴目变化和黑白交换的性能；
- 读秒下每步延迟、吞吐、峰值显存。

权重晋级采用 KataGo gatekeeper 的累计 match points 和数学提前停止，避免把尚未完成的比赛误判为通过；长期强度报告仍应保留永不训练的固定基准集、一组隐藏随机种子，并可另行使用 SPRT 或按开局对统计。

### 9.3 阶段验收

1. **P0 闭环**：9x9 规则、MCTS、训练、自博弈、GTP 全部可运行。
2. **P1 可学习**：在固定 9x9 基准上，训练后的模型显著击败随机/启发式基线。
3. **P2 可扩展**：13x13 和 19x19 能以低预算稳定生成合法样本，训练 loss 和 Elo 不发散。
4. **P3 可服务**：模型热切换、分析接口、日志、恢复训练和失败 shard 隔离完成。
5. **P4 强度**：通过扩大 actor、GPU 和 playout 预算得到持续 Elo 增长，且新 checkpoint 没有明显回归。

## 10. 推荐初始超参数

```yaml
board_size: 9
rules: chinese
komi: 7.5
model:
  d_model: 192
  layers: 8
  heads: 6
  window: 5
  global_attention_every: 2
  history_planes: 8
search:
  simulations: 200
  c_puct: 1.5
  root_dirichlet_alpha: 0.15
  root_dirichlet_epsilon: 0.25
  eval_temperature: 0.0
  selfplay_temperature_moves: 20
train:
  batch_size: 256
  optimizer: adamw
  lr: 0.0002
  weight_decay: 0.0001
  warmup_steps: 2000
  amp: bf16
  replay_window_positions: 2000000
  policy_loss_weight: 1.0
  value_loss_weight: 1.0
  score_loss_weight: 0.3
  ownership_loss_weight: 0.2
  moves_left_loss_weight: 0.1
```

19x19 首次运行时可将 `simulations` 降到 64~128，并让多个 actor 并发生成数据；强度提升主要依赖样本量、模型更新稳定性和评测晋级策略，而不是单次搜索预算。

## 11. 常见失败模式与对策

| 现象 | 可能原因 | 对策 |
|---|---|---|
| 模型只会下角落/固定开局 | 根噪声弱、温度过快降为 0、回放过旧 | 增加开局随机化，保留近期样本，监控策略熵 |
| Elo 短期上升后崩溃 | candidate 过快晋级、价值头发散、数据分布漂移 | 固定评测集、SPRT 晋级、梯度裁剪、降低学习率 |
| MCTS 速度极慢 | 逐叶推理、Python 节点对象过重、重复复制棋盘 | 批量 leaf、arena/bitboard、profiling 后迁移 C++ |
| 胜率正常但目数很差 | 没有 score/ownership 辅助目标或标签错误 | 加入 score/ownership head，独立复算终局标签 |
| 训练 loss 正常但棋力不升 | policy target 错位、value 视角反转、非法 mask 错误 | 做单局逐步回放和符号单测，审计样本字段 |
| 9x9 有效、19x19 失效 | 位置编码、模型容量、历史平面或课程切换不当 | 先 13x13 过渡，增加全局注意力和 19x19 warm start |

## 12. 实施顺序

### Sprint 1：基础正确性

完成规则引擎、SGF、Zobrist、对称变换、GTP，以及 9x9 规则测试。

### Sprint 2：网络和单机搜索

实现 Transformer trunk、所有 heads、mask、单叶 MCTS 和批量推理；用随机网络验证搜索统计和速度。

### Sprint 3：随机初始化和第一轮自博弈

完成随机网络初始化、搜索策略/终局标签、样本 shard、单机 actor/learner 闭环，生成第一个可晋级 checkpoint；SGF 仅导出自博弈记录。

### Sprint 4：稳定性与评测

加入 replay window、candidate/best 评估、Elo、回归集、断点恢复、指标面板。

### Sprint 5：规模化

加入多 actor、GPU learner、模型 registry、低/高 playout 混合、13x13 和 19x19 课程。

### Sprint 6：强度增强

调大模型和搜索预算，加入 score utility、ownership/moves-left 校准、规则扩展和服务端分析。

## 13. 关键工程原则

1. 规则、视角、终局标签和 MCTS 价值符号必须有可执行测试；这是整个系统最容易出现“训练正常但棋力为零”的地方。
2. 先闭环再扩容。能从状态生成合法样本、训练一个更新、评测并晋级，才值得投入更大模型和集群。
3. 把搜索参数、模型版本和规则配置作为数据的一部分记录；否则无法解释 Elo 变化，也无法恢复实验。
4. 辅助目标要服务于棋力和稳定性，不能只优化离线 loss。最终以固定评测对局和计算预算下的 Elo 为准。
5. 训练数据与初始权重坚持随机初始化、纯自博弈；自博弈对局可按项目许可证管理，依赖的规则库和推理框架仍需保留归属信息。
