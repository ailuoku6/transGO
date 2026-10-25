# transformerGo

采用 AlphaGo Zero 式纯自博弈训练：网络从随机权重启动，训练数据全部来自自身 MCTS 对局。

Python / PyTorch Transformer 策略价值网络，C++17 围棋规则和批量 PUCT 蒙特卡洛树搜索。macOS 默认通过 **PyTorch MPS（Metal GPU）** 运行网络推理和训练，也可选择 **Core ML（CPU + Apple Neural Engine/NPU）** 进行网络推理；规则和搜索运行在 CPU。`device: auto` 优先选择 MPS，再选择 CUDA、CPU；`device: mps` 会在 Metal 不可用时明确报错。

这是一版可运行的工程闭环，尚未经过足够数据训练，附带的 smoke checkpoint 仅用于验证流程，没有棋力保证。完整研究方案见 [TECHNICAL_PLAN.md](TECHNICAL_PLAN.md)，实现状态见 [IMPLEMENTATION.md](IMPLEMENTATION.md)。

## 安装

推荐 Apple Silicon、macOS 14+、Python 3.12。需要 Xcode Command Line Tools 中的 Apple Clang。也支持 Linux CPU/CUDA。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
transformer-go doctor
```

本工作区已经创建 `.venv` 并编译扩展，可直接运行 `.venv/bin/transformer-go`。修改 `cpp/native.cpp` 后执行：

```bash
.venv/bin/python setup.py build_ext --inplace
.venv/bin/python -m pytest -q
```

也提供 CMake 构建入口：配置时将 `Python_EXECUTABLE` 指向安装了 pybind11 的虚拟环境 Python。日常开发使用上面的 setuptools 命令即可。

`doctor` 应显示 `mps_available: true`、`selected: mps`。某些执行沙箱不允许枚举 GPU，会显示 CPU；在 macOS 普通终端运行可以区分设备权限问题。程序没有默认启用 MPS CPU 算子回退，防止掩盖不支持的算子。

## Mac NPU / Neural Engine 推理

Apple Silicon、macOS 13+ 可安装可选依赖，在现有推理命令后添加 `--inference-backend coreml`：

```bash
.venv/bin/python -m pip install -e '.[coreml]'
.venv/bin/transformer-go doctor

# 先使用既有权重对照 MPS 与 Core ML；第一次转换/编译发生在预热阶段，不计入搜索耗时。
.venv/bin/transformer-go benchmark --config configs/tiny.yaml \
  --checkpoint runs/zero-9x9/best.pt --inference-backend torch
.venv/bin/transformer-go benchmark --config configs/tiny.yaml \
  --checkpoint runs/zero-9x9/best.pt --inference-backend coreml \
  --coreml-batch-size 16 --coreml-compute-plan

# 原有训练目录可继续接续；训练仍由 config.device 选择 PyTorch/MPS 设备。
.venv/bin/transformer-go loop --config configs/tiny.yaml --output runs/zero-9x9 \
  --iterations 1 --games 64 --steps 1000 --eval-games 200 \
  --inference-backend coreml --coreml-batch-size 16
```

`selfplay`、`eval`、`loop`、`play`、`gtp`、`json-rpc` 和 `benchmark` 均支持这些后端选项。默认仍是 `torch`，旧命令及 checkpoint/replay schema 不变；不需要修改 `device` 为 `npu`。自博弈/评估的并行 CPU worker 仍共用父进程中的集中推理服务。`loop` 当前按阶段执行，训练阶段使用 MPS、推理阶段使用 Core ML，并非 GPU/NPU 同时训练。

Core ML 用 `CPU_AND_NE` 允许 CPU 与 Neural Engine 执行，禁用 GPU；这不是 NPU 独占模式，具体分配由 macOS 决定。`doctor.coreml.hardware_eligible` 仅报告平台条件，`ane_execution_verified: false` 表示 doctor 未分析模型执行；可用 `benchmark --coreml-compute-plan` 查看该模型的静态 `preferred_operations`。它也不是实时 NPU 利用率，`runtime_utilization_measured` 明确为 false。旧系统不支持执行计划时返回诊断原因，不影响正常预测。

推理导出为 FP16 计算、float32 输入/输出，保留全部预测头；合法着 mask、softmax、WDL 价值组合仍在 CPU 执行。每种棋盘尺寸按需生成固定形状模型，batch 默认 16，不足补零、超过拆分，返回时去掉补齐行。可以尝试 `--coreml-batch-size 1/4/16`，单局小批请求与多进程合批的最佳设置不同。`--inference-batch-size` 仍是集中服务的跨局面收集上限，含义不变。

转换结果默认缓存到 `runs/coreml/`，由权重、模型结构、棋盘尺寸、batch、导出及软件版本区分；可用 `--coreml-cache-dir` 改路径。首次转换和系统编译有额外启动耗时，每轮 candidate 权重变化需要新的转换。并行模式会在启动 worker 前准备模型，避免首次编译耗尽推理 RPC 超时。显式选择 Core ML 时，依赖、平台或预测失败会报错，不会静默切回 PyTorch。

本机现有 9x9、192 维/8 层、step 327000 的 best 权重，在空盘、200 次搜索、预热后测量 3 次，MPS 搜索耗时中位数为 0.320 秒，Core ML batch 16 为 0.104 秒（约 3.1 倍）；Core ML 静态计划有 225 个算子首选 Neural Engine。batch 1 中位数约 0.205 秒。此为单一局面和本机的局部基准，不代表完整自博弈/训练吞吐或棋力提升，也没有测量功耗。该次空盘的 policy/value/score 最大绝对误差约为 `3.9e-4 / 9.0e-4 / 2.8e-5`。原始结果在 `.feature_memo/Mac神经网络引擎支持/benchmark.json`。

验证命令：

```bash
.venv/bin/python -m pytest -q
# 需要可访问原生 Core ML 服务的 macOS 终端；普通 CI 默认跳过硬件测试。
TRANSFORMER_GO_TEST_COREML=1 .venv/bin/python -m pytest tests/test_coreml.py -q -k real_coreml
```

本机已用 PyTorch 2.14.1 / coremltools 9.0 验证。coremltools 可能提示该 PyTorch 版本超出其官方测试范围；项目使用固定形状、显式注意力导出绕过已发现的转换问题，升级依赖后应重跑硬件测试。执行沙箱可能限制系统编译缓存或硬件访问，在普通 macOS 终端运行可区分权限问题。

## 先运行最小闭环

```bash
.venv/bin/transformer-go selfplay \
  --config configs/smoke.yaml --output runs/demo/replay --games 2

.venv/bin/transformer-go train \
  --config configs/smoke.yaml --data runs/demo/replay \
  --output runs/demo/model.pt --steps 5

.venv/bin/transformer-go train \
  --config configs/smoke.yaml --data runs/demo/replay \
  --resume runs/demo/model.pt --output runs/demo/model.pt --steps 5
```

`smoke.yaml` 使用 3x3、41,610 参数网络、每步 8 次模拟，以快速验证接口。正式实验使用 `configs/tiny.yaml`：9x9、192 维、8 层、每步 200 次模拟。13x13、19x19 可以复制配置修改 `board_size`，同一网络支持动态尺寸；训练回放每次只采样指定棋盘尺寸。

无需手动串联各步骤的闭环命令：

```bash
.venv/bin/transformer-go loop \
  --config configs/smoke.yaml --output runs/loop-smoke \
  --iterations 1 --games 2 --steps 3 --eval-games 2

.venv/bin/transformer-go loop \
  --config configs/tiny.yaml --output runs/9x9 \
  --iterations 5 --games 16 --steps 200 --eval-games 200
```

`loop` 保存独立的 `best.pt` 与 `learner.pt`：actor 使用 best；learner 不因候选模型未晋级而丢失训练进度。每轮保存 candidate、对局 replay 和评测 JSON。晋级现在采用 KataGo gatekeeper 的 match-point 规则：默认最多 200 盘，candidate 得分为胜局 1 分、和棋/无结果 0.5 分，达到 `--required-candidate-win-prop`（默认 0.5）即可接受，平分晋级；如果剩余对局无论结果如何都不能达到门槛，则提前拒绝，达到门槛也会提前接受。评估从空盘开始，候选黑白双方轮换，关闭根 Dirichlet 噪声，并使用 KataGo gatekeeper 示例的 0.5 到 0.2 温度衰减。Wilson 区间仍作为描述性字段输出，不参与晋级。2 盘 smoke 评测也遵循同一累计得分规则。

如果要复现 KataGo 示例 gatekeeper 的默认门槛，可以使用：

```bash
.venv/bin/transformer-go loop \
  --config configs/tiny.yaml --output runs/zero-9x9 \
  --iterations 500 --games 64 --steps 1000 --eval-games 200 \
  --min-moves 70 --required-candidate-win-prop 0.5
```

`--eval-games 50` 仍然可以用于较快评估；在默认 0.5 门槛下，50 盘完整比赛至少需要 25 个 match points，25 胜 25 负也会晋级。由于 gatekeeper 会提前停止，实际完成的对局数可能小于上限，iteration JSON 的 `played`、`candidate_points`、`target_points`、`stop_reason` 和 `promotion_policy` 是晋级审计依据。

当前闭环按轮执行，并非跨机器异步集群；不要同时对同一个 loop 目录启动多个 learner。

## 自博弈探索策略

`selfplay` 和 `loop` 默认使用 `--selfplay-policy katago`。该运行时选项不进入 Config/checkpoint，因此可以接续原来的 `best.pt`、`learner.pt` 和 replay。新策略包含：

- 落子温度从 `0.75` 平滑衰减到 `0.15`，公式为 `T(t)=0.15+0.60×0.5^(t/N)`，`t` 是已落子总手数，`N` 是棋盘边长；9 路半衰期为 9 手。较低温度减少低访问点合计占据大量采样概率的情况，仍保留探索。
- 根节点使用按网络策略塑形的 Dirichlet 噪声，总 concentration 为 `10.83`，一半均匀、一半按截断后的 log policy 分配；混合权重沿用 `search.dirichlet_epsilon`。
- 搜索完成后，参照 KataGo 回顾性修正减少低效探索的选点权重。实际落子从修正权重按温度抽样，训练策略目标则是修正权重直接归一化，**不再按落子温度锐化**。低访问但高价值的候选仍可保留，不设置一刀切的 1% 概率门槛。

这部分对照本地 KataGo `d91ea855` 的 `cpp/configs/training/selfplay1_maxsize9.cfg`、`searchhelpers.cpp`、`searchresults.cpp` 和 `searchexplorehelpers.cpp`。温度/噪声方案取自该示例，权重修正则适配本项目简化 PUCT，是近似移植；尚未移植 LCB、分叉局和廉价搜索等完整训练机制。当前 tiny 配置的 200 次访问预算不变，也尚无棋力提升的 A/B 结论。

原始 `visits`、`policy` 和树复用统计保持搜索的真实访问数；新增 `selection_weights`、`selection_policy` 表示修正后的选点/目标权重。replay 元数据保存 `exploration` 和 `policy_correction_stats`，`game_completed` 也输出平均修正量与单访问点的权重占比。评估/晋级仍用既有无根噪声协议，独立 `eval --temperature 0` 仍选原始最高访问次数。

需要对照旧方案时，在原命令末尾添加 `--selfplay-policy legacy`，恢复前 `search.temperature_moves` 手温度 1、此后 0.1、对称 Dirichlet 噪声和原始访问分布目标。旧 replay 保留原有策略标签，新数据随生成逐步进入 replay window。

本次修改包含 C++ 搜索，需使用重新编译的 native 扩展。运行中的旧 loop 不会热加载新策略；在合适的轮次边界结束后重新启动原命令，默认即使用新策略并接续已有训练状态。

## 每轮训练 loss

`loop` 自动保存 loss，不需要额外参数。例如 `--output runs/zero-9x9`：

```text
runs/zero-9x9/training-loss.csv
runs/zero-9x9/training/iteration-NNNN/steps-起始步数-唯一后缀/steps.jsonl
runs/zero-9x9/training/iteration-NNNN/steps-起始步数-唯一后缀/summary.json
```

- `training-loss.csv`：每次完成训练追加一行，包含轮次、起止训练 step、采样池局面数，以及各项的整轮均值和最后一步值。训练结束即写入，无需等待评估完成。
- `steps.jsonl`：每个训练步记录 `total/policy/value/score/ownership/moves/grad_norm`，逐步 flush；训练中断时已写步骤仍保留。
- `summary.json`：训练和 candidate checkpoint 保存成功后原子写入，包含上述指标的 mean/min/max/last、各损失权重、设备和日志路径；完整轮次的 `iteration-NNNN.json` 也包含该汇总。

续训会创建新的日志子目录。若一轮在评估期间中断、重跑同一轮，CSV 保留不同训练尝试，以起止 step 和 `summary_path` 区分，不覆盖旧日志。训练中途失败的尝试没有 completed summary，也不追加完成行。

单独 `train` 默认写到 `OUTPUT_PARENT/training/OUTPUT_STEM/steps-*`，可通过 `--metrics-dir` 指定目录。终端仍输出训练进度，训练完成时额外输出 `training_summary`。新日志从运行更新后的代码开始记录，旧版本未持久化的逐步 loss 不能从 checkpoint 还原。

各分项 loss 为加权前数值；`total` 是按 summary 中权重求和的优化目标，`grad_norm` 是裁剪前梯度范数。replay 分布会随训练变化，因此 loss 趋势需要结合评估结果判断，不能单凭下降认定棋力提高。

## 清理旧 checkpoint 和 replay

脚本默认预览 `runs/zero-9x9` 中修改时间早于当前时刻 24 小时的文件，添加 `--delete` 才实际删除：

```bash
.venv/bin/python scripts/cleanup_old_artifacts.py
.venv/bin/python scripts/cleanup_old_artifacts.py --delete
```

清理范围仅为运行目录根部的旧 `.pt`。始终保留根部 `best.pt`、`learner.pt`，并跳过符号链接；`replay/` 下的 `.npz`、`.sgf`，以及评估/观棋目录、训练 loss 和 iteration JSON 均保留。删除前会再次检查文件年龄，已更新的文件跳过。

“一天之前”按滚动 24 小时计算，不是当天零点。可用 `--older-than-hours 48` 改为两天前，或用 `--root runs/另一个运行目录` 指定 run；默认目录相对于脚本位置确定，不依赖启动时的当前目录。输出包含截止时间、文件清单及总字节数。训练、评估或对弈正在使用 checkpoint 时，应等当前进程完成后再执行删除，避免它在后续阶段需要重新读取某个待删除文件。

## 并行自博弈与评估

```bash
.venv/bin/transformer-go selfplay \
  --config configs/tiny.yaml --checkpoint runs/9x9/best.pt \
  --output data/replay --games 8 --workers 2 --min-moves 40
```

使用 spawn CPU 搜索进程，父进程只加载一份网络并集中合批推理，树和 Go 规则状态仍留在各 worker。同一局里选中着法的子树会升为下一手根节点；保留访问计入该手预算，只补足到 `search.simulations`。不同局面请求按模型和棋盘尺寸合批。并发数默认 1；多进程时可用 `--inference-batch-size` 限制单次推理上限（默认 256 局面），`--inference-wait-ms` 控制合批等待（默认 2 ms）。父进程转发每盘 `game_completed`，并报告推理批次统计和 `inference_summary`。MPS 共享内存的设备适用性、batch 等待和 worker 数需用目标硬件测量。

完整 `loop` 也支持把自博弈和评估分别放入多进程池：

```bash
.venv/bin/transformer-go loop \
  --config configs/tiny.yaml --output runs/zero-9x9 \
  --iterations 500 --games 64 --steps 1000 --eval-games 200 \
  --min-moves 70 --selfplay-workers 2 --eval-workers 2
```

`--selfplay-workers` 控制 CPU 搜索进程数，`--eval-workers` 控制并行评测局数；两者默认都是 `1`，旧命令保持串行兼容。并行模式下父进程分别持有 self-play 网络，或 candidate 与 baseline 两个独立网络；子进程提交 CPU 特征和合法 mask，不加载 GPU 模型。请求按模型分别合批，评测局每走一步都会同时推进两棵搜索树；candidate 的访问统计不会混入 baseline。父进程预先生成黑白轮换与随机种子，并按该顺序汇总 match points，晋级结果不受完成顺序影响。gatekeeper 已经确定结果时，当前批次已运行棋局仍会完成并丢弃，不计入 `played`；推理服务会一直等到 worker 退出。

`--inference-batch-size` 和 `--inference-wait-ms` 是并行模式的运行时参数，不写入 checkpoint 配置。`evaluate_parallel` 会在结果中记录每个模型的 forward 数、位置数、最大/平均 batch 和跨 worker 合批次数。树只在同一局、同一模型与完整合法历史匹配时复用；变更 `min_moves`、score utility 或评估器会使旧树失效。若旧树已有的访问超过新预算，保留访问会原样返回，不会为了凑预算丢弃搜索统计。

也可以单独并行评估两个 checkpoint：

```bash
.venv/bin/transformer-go eval \
  --config configs/tiny.yaml --candidate runs/new.pt \
  --baseline runs/zero-9x9/best.pt --games 200 --workers 2 \
  --required-candidate-win-prop 0.5
```

集中合批只在多个 worker 时启用；单 worker 直接调用模型，也会自动复用子树。MPS/Metal 可先从 2～4 个 worker 测量，评估中每个模型要有多个请求同时就绪才能合批。`search.batch_size` 仍控制单树叶子批量，`--inference-batch-size` 是跨局面上限，增大上限本身不保证满批。checkpoint 提供完整的自博弈模型及搜索配置，`--config` 的 device 字段决定本次运行设备。训练器继续在父进程中单进程运行。

## 查看评估棋谱

新的 loop 进程会自动把每盘计分的评估对局保存到输出目录下的
evaluation/iteration-NNNN/match-SEED-唯一后缀/game-NNNN.sgf。
例如输出目录为 runs/zero-9x9 时，棋谱就在其 evaluation 子目录中。
每盘结束后立即写入，evaluation_game 日志含 sgf_path，iteration JSON 的
evaluation.sgf_dir 指向本次比赛目录。SGF 可用支持 SGF 的围棋看谱工具打开。
重跑同一轮会新建比赛目录；并行提前停止后被丢弃、未计分的局不导出。
评估棋谱与 replay 分开，不作为训练数据。

SGF 记录双方 checkpoint 名称及 SHA-256、执色、种子、胜负/目差、搜索预算和温度。
每手注释包含 MCTS 首选坐标、首选访问次数、实际着法访问次数、实际着法是否达到最高访问次数（含并列）。
达到手数上限时会标明 forced_scored 和 move_limit，只保存真实着法，不补造 pass；无结果记录为 Void，和棋为 0。

默认晋级评估关闭根噪声，但仍以 0.5 衰减到 0.2 的温度抽样，因此并非每手都下访问次数最多的点。
想看当前 best 在每手都选择 MCTS 最高访问次数时的棋风，可独立运行两盘观棋对局：

~~~bash
.venv/bin/transformer-go eval \
  --config configs/tiny.yaml \
  --candidate runs/zero-9x9/best.pt \
  --baseline runs/zero-9x9/best.pt \
  --games 2 --workers 2 --temperature 0 \
  --score-utility 0.05 --sgf-dir runs/zero-9x9/inspection
~~~

独立 eval 默认保存到 runs/evaluation，可用 --sgf-dir 改位置。
--temperature 0 只表示当前模型和搜索预算下的访问次数首选，不保证客观最优；也不等于原始网络策略概率最高的点。
该参数仅用于独立诊断，不能与 --required-candidate-win-prop 同时使用，loop 的晋级规则不变。
双方用同一模型、空盘和零温度时，对局可能重复，不适合用这种对局的胜率判断棋力提升。

评估棋谱导出本身只涉及 Python；自博弈策略更新还涉及 native 编译。已经运行的 loop 不会自动加载修改，需在合适的轮次边界退出后重新启动原命令；
新进程会接续已有 checkpoint。此前未保存的评估对局不能直接补回，重新对战属于新生成的棋谱。

## 从零开始的训练路线

空的运行目录会随机初始化网络，随后反复执行：

```text
随机初始化 θ₀
  -> 当前 best 网络 + MCTS 自博弈
  -> 保存 (局面 s, 搜索访问分布 π, 终局胜负 z)
  -> learner 用 replay 数据更新网络
  -> candidate 对战 best
  -> 达到门槛后晋级，继续自博弈
```

无需下载人类棋谱或预训练权重。策略目标来自 MCTS 访问权重（默认经过回顾性修正），价值目标来自实际终局；目数、归属和剩余手数也只从同一盘自博弈生成。最早几轮随机网络可能很弱或频繁 pass，这是冷启动阶段需要用探索、更多对局与评测观察的问题。

```bash
.venv/bin/transformer-go loop \
  --config configs/tiny.yaml --output runs/zero-9x9 \
  --iterations 5 --games 16 --steps 200 --eval-games 200 --min-moves 40 \
  --score-utility 0.05
```

`--min-moves` 只约束自博弈：达到指定的总手数前，搜索会屏蔽 `pass`；如果棋盘上已经没有合法的非 `pass` 落子，则仍允许 `pass` 防止对局卡死。达到该手数后恢复正常双 `pass` 终局。它按黑白双方合计的 ply 计数，例如 `--min-moves 40` 表示前 40 手不能主动 pass，通常最早在第 42 手结束。评测对局不使用此限制，仍按正常规则进行。该参数是运行预算，可以在同一 `loop` 目录续跑时修改；它会写入 replay 元数据和迭代记录，便于审计。

`--score-utility` 把网络的 `score_mean` 以一个很小的权重加入 PUCT 叶节点 utility。默认 CLI 值为 `0.05`，用于让搜索同时关注胜负和归一化目数；它不修改模型结构、checkpoint 配置或已有 replay，可以直接在现有 9x9 目录续跑。设为 `0` 可恢复只有 WDL value 的旧搜索行为。自博弈和评估使用同一个值，避免评测协议不一致；每个 replay shard 和 iteration JSON 都会记录该参数。

同一命令再次运行会续接该目录的 learner/best；用新的空目录开始独立实验。这个命令是启动实验的例子，不能把 5 轮当作达到指定棋力的训练预算。

训练器只接受 `source=selfplay` 的 replay shard，拒绝人类棋谱和来源缺失的数据。用于自博弈或续训的 checkpoint 必须带 `training_origin=random_init_selfplay`。此前旧版本没有记录该标记的 checkpoint 仍可用于对弈/查看，但不会自动进入新的 Zero 训练链；请用新的运行目录从随机权重开始。来源字段是工程追踪记录，不是对外部伪造文件的真实性认证。

SGF 仅用于导出和查看自博弈记录，已移除将外部 SGF 转为训练样本的入口。采用 Transformer 与 KataGo 辅助目标是本项目的结构选择；不使用人类数据的训练原则与 AlphaGo Zero 一致，网络和搜索工程仍是本项目自己的实现。

## 对弈与分析接口

直接使用浏览器与本地引擎对弈：

```bash
.venv/bin/transformer-go play
```

默认从 `runs/` 下选择修改时间最新的 `best.pt`，并使用 checkpoint 自带的棋盘、规则和 200 次 MCTS 搜索配置。终端会显示本机地址，浏览器打开后可改选其他 `best.pt`、`candidate-*.pt`、`learner.pt` 或 checkpoint；切换权重在新对局时生效。可选择黑白、9/13/19 路棋盘和每手搜索预算，默认 200 次，也支持悔棋、停一手、认输、局面分析和 SGF 导出。服务仅监听本机回环地址。

例如指定某个 checkpoint 和端口：

```bash
.venv/bin/transformer-go play --checkpoint runs/closed-loop-v2/best.pt --port 8766
```

对局时引擎每手使用最高访问次数落子；增加搜索预算会增加每手搜索量，但网页显示的网络胜率仍只是模型估计，不是正式棋力评级。终局胜负和目差来自现有 KataGo 规则核心。

GTP 2 标准输入/输出接口：

```bash
.venv/bin/transformer-go gtp --config configs/tiny.yaml --checkpoint runs/9x9/best.pt
```

支持 `boardsize`、`komi`、`clear_board`、`play`、`genmove`、`undo`、`showboard`、`final_score`、`final_status_list alive/dead` 等命令，支持编号和注释。坐标跳过 I 列，例如 `play b D4`。`komi` 在开局前设置；`final_score` 只能在当前规则阶段正式结束后查询（日式规则会经过两轮 encore）。

本地 GUI 可使用逐行 JSON-RPC 2.0 接口：

```bash
.venv/bin/transformer-go json-rpc --config configs/tiny.yaml --checkpoint runs/9x9/best.pt
```

请求示例：

```json
{"jsonrpc":"2.0","id":1,"method":"new_game","params":{"size":9,"komi":7.5}}
{"jsonrpc":"2.0","id":2,"method":"play","params":{"action":30}}
{"jsonrpc":"2.0","id":3,"method":"analyze","params":{}}
```

`analyze` 返回各动作 `visits`、归一化 `policy`、当前玩家视角 `value` 和模拟次数，不自动落子。动作编码为行优先 `row*N+col`，`N*N` 是 pass。此接口是本地进程协议，尚未实现 HTTP 服务。

`analyze` 还返回 `model` 形势报告：`wdl`、`winrate`、`score_mean`、`score_std`、`moves_left`、逐点 `ownership`、ownership 置信度和 `likely_dead` 预测。`likely_dead` 只是网络对“继续合理下棋后的归属”的判断，不会修改棋盘，也不会改变正式胜负。棋局结束后，`state` 会附带 `adjudication`，其中 `winner`、`score_black`、`ownership`、`dead_stones`、`territory` 和 `seki` 全部来自 KataGo `BoardHistory` 规则核心；GTP 还支持 `final_status_list alive|dead|seki`。

## 评测与性能测量

```bash
.venv/bin/transformer-go eval \
  --config configs/tiny.yaml --candidate runs/new.pt \
  --baseline runs/old.pt --games 40

.venv/bin/transformer-go benchmark --config configs/tiny.yaml
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q tests/test_mps.py
```

benchmark 包含一次预热并同步 GPU，报告该配置下的搜索速度、推理调用数和参数量。小棋盘、小 batch 的 Metal 调度成本可能高于 CPU；需要在 9x9/19x19 和实际 batch 下测量，不能用 smoke 数值推断正式吞吐。

搜索复用与集中推理可用 `.venv/bin/python -m benchmarks.search_optimizations --config configs/tiny.yaml --device mps --board-size 9 --simulations 200 --workers 4` 做对照；结果含树复用前后的访问/推理量、单机并行的启动与搜索时间、平均推理 batch 和跨进程合批数。当前工具沙箱若报告 MPS 不可用，可使用 `--device cpu` 验证功能，但不能从 CPU 小盘吞吐推断 GPU 加速。CPU smoke 网络（5x5、8 手、每手 64 模拟、4 worker）只观察到调用 548→138、平均 batch 3.8→15.0，搜索阶段 0.281→0.442 秒；它证明请求确实合并，且说明低负载 CPU 下 IPC/同步成本能抵消合批收益。

## 数据、视角与规则约定

- 默认使用 KataGo `Chinese-OGS` 规则：面积计分、禁止自杀、**位置超级劫（PSK）**、双 pass 终局；`rules: japanese` 使用 KataGo 的 territory + seki tax + 三阶段 encore；也可传 KataGo 的 JSON/legacy rules 字符串。
- C++ 原生规则层直接编译 KataGo 的 `Board`、`BoardHistory`、`Rules` 及其依赖。终局裁定会按 pass-alive/Benson、独立生命区域、seki、死子归属、territory encore、ko/suicide/button 配置计分；网络 ownership 只用于分析预测，不参与正式裁定。完整 KataGo 引擎还包含神经网络、分布式搜索和更多赛事协议，本项目只复用规则核心。KataGo 按许可证随项目一并分发所需源文件，版本和归属见 `cpp/vendor/katago/UPSTREAM.md`。
- C++ 通过 KataGo 的精确棋盘/历史状态保存超级劫和 encore 历史，另提供确定性 Zobrist hash。特征含最近 8 个棋盘、合法点、最近落子、黑方行棋标志、视角贴目和进度，共 21 通道。
- 策略为搜索访问权重分布（`katago` 使用修正权重，`legacy` 使用原始访问次数），胜负类顺序为 `负/和/胜 = 0/1/2`；价值、目数与归属均以样本当前行棋方为视角。
- score 和 moves-left 除以棋盘面积归一化；ownership 是 `对手/中立/当前方 = 0/1/2`。自博弈正常终局样本的 `aux_mask=1`。
- 达到最大手数仍未终局的自博弈局丢弃；不会用强制 pass 制造训练标签。
- 评估 gatekeeper 使用 KataGo 的 `maxMovesPerGame` 风格强制计分：超过评估手数上限的局面通过 `BoardHistory` 当前分数计入 match points；无结果局按 0.5 分计入双方。自博弈仍丢弃未正式终局的局，避免把强制评估裁定混入训练标签。
- NPZ shard 不使用 pickle，包含 schema、UUID、模型 SHA256、种子、搜索配置和规则元数据。写入先落临时文件再原子替换。float32 特征保留贴目与进度精度；未来规模化时再量化/迁移为流式 shards。
- 训练使用近期 replay window，并按 shard 新鲜度加权采样：最新选中的 shard 权重为最旧 shard 的 4 倍，同一 shard 内仍均匀；同时使用 D8 对称增强、AdamW、梯度裁剪和 float32。Checkpoint 保存权重、优化器、训练步数、配置、采样 RNG，支持恢复训练。规则核心升级不会删除 `runs/` 下已有 checkpoint；继续同一 9x9 目录时，旧 replay 会通过 SGF 在 KataGo 规则下重放，只重算 value/score/ownership 标签，并写入 `.adjudication-cache/`，原 NPZ/SGF 与权重保持不变。新生成的 shard 会记录 `adjudication_version=katago-boardhistory-v1`。
- 局部注意力当前用 dense attention mask 实现，表达局部先验但没有稀疏窗口算子的性能收益；C++ 单树批量搜索使用叶子预留/虚拟损失，尚未实现多线程共享树。

## 已验证

2026-10-21：KataGo 式自博弈探索和 loss 持久化已编译并通过全套测试，138 passed、5 skipped（当前工具环境 MPS 不可用）。独立临时目录真实 CLI 完成并行 `katago` 一轮，再以串行 `legacy` 续跑一轮；训练 step 2→4，保存 4 局 replay、4 局评估棋谱和 2 行 loss 汇总，原 Config 一致。日志均值/极值/最后值、断点续训不覆盖、失败保留逐步日志已验证。尚未进行棋力 A/B 或本次 GPU 实测。

2026-10-15 在当前 Apple Silicon Mac 上验证了：C++ 编译、独立 Python 规则 oracle 对照、提子/禁入/超级劫/计分、搜索访问数与价值视角、SGF 往返、CPU 自博弈与断点续训、并行 actor、完整候选评测闭环，以及真实 Metal 自博弈训练。Metal 专项测试覆盖 9x9/13x13/19x19 前向/反向和优化器续训。没有进行长期棋力训练，也没有验证达到任何业余段位或专业强度。
