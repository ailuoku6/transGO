# 实现日志

## 2026-10-22 旧文件清理脚本

- 新增 `scripts/cleanup_old_artifacts.py`，纯 Python 标准库，无需重编。默认 run 为项目 runs/zero-9x9；按修改时间早于当前 24 小时筛选，不使用当天零点。
- 范围：根目录 .pt、replay 递归 .npz/.sgf（含标签缓存）；固定保留 best.pt/learner.pt。默认 dry-run，--delete 才删除；--root 和 --older-than-hours 可配置。跳过符号链接，unlink 前复查年龄，错误返回非零。
- 新增 10 项测试通过，覆盖预览、真实临时文件删除、保留权重/其他目录、截止边界、非法参数、符号链接、扫描后被更新的文件。
- 真实目录只预览，列出 7,816 个文件、1,830,374,923 字节（约 1.70 GiB）；截止为上海时间 2026-10-21 20:49:01。此轮没有删除用户数据。
- README 增加命令和运行范围。replay 正在读取时不应并发执行删除；本脚本未接入训练自动清理或计划任务。

## 2026-10-21 KataGo 式探索与训练 loss

- 用户授权对照本地 KataGo 策略优化并记录每轮 loss。参考 d91ea855 的 selfplay1_maxsize9.cfg、searchhelpers/searchresults/searchexplorehelpers；默认 runtime profile=katago，可选 legacy，不改 checkpoint Config。
- C++ 增加总 concentration 10.83 的 policy-shaped Dirichlet，以及适配本项目未加权 PUCT 的回顾性选点权重修正。只改返回的 selection_weights/selection_policy，不覆盖原始 visits/policy，不破坏复用预算。保留高价值单访问点；严格小于 prune threshold 才移除。
- 自博弈温度 .75→.15，半衰期为棋盘边长；实际着法应用温度，策略训练标签保持修正权重直接归一化。replay/game_completed 记录 profile 和修正统计。旧父进程的四参数 actor 调用缺省 legacy，新的 parent 显式传 katago。
- train 每步 flush JSONL；成功保存模型后原子写 mean/min/max/last 汇总；loop 在评估前追加 CSV，每轮 JSON 含 summary。重复/失败尝试不覆盖历史；独立 train 支持 --metrics-dir。
- 修正新增 selfplay 测试的 seed 参数接口；补充汇总数值、逐步全量写出、续训不覆盖、训练中断保留和 CLI profile 传递回归。
- 编译：setuptools native 已成功生成；专项 56 passed；完整 `.venv/bin/python -m pytest -q` 为 138 passed、5 skipped，8.42 秒，跳过项因工具环境 MPS 不可用。
- 临时真实 CLI：并行 katago（2 worker）→串行 legacy 各一轮，训练 step 2→4，4 局 replay、4 局评估、2 行 CSV；目录 /var/folders/rg/ft9nflgn0dz_vx88v7plg4pc0000gn/T/transformer-go-exploration-loss-9ci2l0wa。
- README/IMPLEMENTATION 更新行为、日志位置、旧 checkpoint 兼容和重启要求；未中断用户进程，未改原 run 的权重/replay。未移植完整 KataGo（LCB/分叉局/廉价搜索），尚未做棋力 A/B。

## 2026-10-16

- 新增 `--min-moves` 自博弈参数：阈值前由 C++ PUCT 屏蔽 pass，只有棋盘无合法非 pass 时例外放行；评测保持标准规则。
- replay metadata、`game_completed` 事件和 iteration JSON 记录 `min_moves` 与异常提前 pass 计数。
- 新增 `tests/test_min_moves.py`，覆盖树内 pass 约束、终局例外、CLI 续跑、并行 actor 和参数校验。
- 验证：`.venv/bin/python -m pytest -q tests/test_min_moves.py`，12 passed。
- 新增可选 score utility：C++ 搜索接受 evaluator 的第三个 score 数组，按 `value + weight*tanh(score_mean)` 回传；CLI 默认 0.05，参数不写入 checkpoint config，现有 run 可续接。
- 验证：全套测试 43 passed、4 skipped；临时 smoke loop 使用 `score_utility=0.05` 完成两轮并从 step 1 续到 step 2。
- 新增 `judgment.py`：终局 `adjudicate` 只使用规则引擎；网络 `position_report` 输出 WDL、score 不确定性、ownership 和 likely-dead 形势预测。RPC `analyze`/终局 `state` 分别暴露预测和正式裁定。
- 验证：全套测试 44 passed、4 skipped。

## 2026-10-15

- 从空目录建立 setuptools/pybind11 工程、可选 CMake、CLI 和 Makefile。
- 实现 C++ 规则、历史编码、批量 PUCT；PyTorch 多头 Transformer、MPS。
- 实现 selfplay/SL/RL/replay/checkpoint、评测晋级、spawn actors、GTP、JSON-RPC。
- CPU：24 passed，4 个 Metal 项在受限环境下跳过；包含独立规则 oracle 随机对照和精确 CPU 续训一致性。
- Metal：4 passed，覆盖 9/13/19 路前向反向和自博弈训练后恢复优化器。
- 完成 CPU 与 Metal 真实自博弈各 2 局、Metal 5 步训练、CPU 两轮 candidate 评测闭环、多 actor 两局。
- 9x9 tiny 4,821,898 参数、200 simulations 单次测量：CPU约0.290秒、MPS约0.839秒；是小 batch 功能性测量，不能推广成稳定吞吐或加速结论。
- 无长期棋力训练。后续研究待办详见 `IMPLEMENTATION.md`。

- 最终统一验证：在可访问 Metal 的环境运行 `.venv/bin/python -m pytest -q`，28 passed（2.75 秒），包含全部 CPU 与 Metal 用例。

- 2026-10-15 纯自博弈更新最终验证：完整 CPU+Metal 测试 34 passed（2.78 秒）；新的 runs/zero-only-smoke 在 MPS 上完成 2 局、39 个训练局面、3 步训练和 2 局候选评测。无长期棋力结论。

## 2026-10-20 完成集中批量推理和 MCTS 子树复用

- 两项优化已接入串行/并行自博弈与评测：串行复用树，多 worker 增加父进程集中推理；候选/基准独立树且每手同步推进。
- `.venv/bin/python setup.py build_ext --inplace` 成功（第三方代码既有警告）；`.venv/bin/python -m pytest -q`：109 passed、5 skipped，10.69 s，跳过项均因当前沙箱 MPS 不可用。
- 真实 CLI 并行闭环：2 worker 自博弈 2 局、训练 1 step、2 worker 评测 2 局；参数 16/1 ms 生效，保存 Config 与原始配置一致。临时运行目录位于系统 tmp，未改已有 runs/zero-9x9。
- `.venv/bin/python -m benchmarks.search_optimizations`：CPU、smoke 网络、5x5、8 手、64 模拟。复用树 NN 位置 518→489、forward 136→128，约 0.110→0.100 s（单次微基准）。4 worker 独立 forward vs 集中：548→138 次，2076 位置一致，平均 batch 3.79→15.04；搜索 0.281→0.442 s。小网络 CPU 未加速，GPU 速度尚未测量。
- 正式棋力和目标 MPS/CUDA 吞吐不属于本次已验证结论。

## 2026-10-20 棋力只读诊断

- 只读统计 11,467 局、1,115,521 个局面；最近 128 局占 replay 1.0874%。本轮 1,000 step × batch 128 的抽样预期约 1,392 次来自这批数据，说明旧策略 replay 可能稀释新策略，尚需 A/B。
- 读取 best 权重，三个局面做 200/800 模拟无噪声对照。样本棋谱角点为低概率探索行为；搜索量能改变部分访问分布，但无强引擎裁判，不能据此断定哪一方变好。
- 初判为探索外观、数据新旧比例及搜索预算共同可疑；8 层模型每 2 层已有全局注意力，未发现架构表达范围不足的直接证据。诊断与后续实验顺序写入 `diagnostics-2026-10-20.md`，未改配置/权重/replay。

## 2026-10-20 评估 SGF 与零温度观棋

- 用户希望看评估阶段的棋谱，以区分探索落子和搜索偏好。新增父进程按计分顺序导出 SGF；串行和真实 spawn 并行均接入。
- loop 每轮保存 evaluation/iteration-NNNN/match-SEED-唯一后缀/game-NNNN.sgf；每盘日志含 sgf_path，结果含 sgf_dir。提前停止后丢弃局不保存，同轮重跑不覆盖旧局。
- SGF 包含双方名称/CLI与loop的模型哈希、执色、种子、规则、目差、温度/预算及逐手首选信息。无结果 RE[Void]，和棋 RE[0]；手数上限局标记 forced_scored/move_limit，不造 pass。
- 独立 eval --sgf-dir 可自选目录，--temperature 0 用于搜索首选观棋；与 gatekeeper 门槛互斥，不改 checkpoint 配置、replay 或正式晋级规则。
- .venv/bin/python -m pytest -q：126 passed、5 skipped，10.71 秒。覆盖 SGF 合法回放、转义、无结果/和棋、强制计分、随机数不受导出影响、重跑、串行/并行 CLI、并行提前停止、loop 续跑及模型哈希。
- 原 native 编译产物无需重编。既有长跑不会热更新，需用户在合适轮次边界自行重启原 loop 命令；未中断或重启用户进程。

- 真实 9x9 观棋验证：固定 best 快照（step 141000，SHA-256 da366441a459c5066e86b8ba78855b147f0804907c0d58106b400bbeb818bc28），CPU 单线程、原 200 访问预算、score utility 0.05、温度 0，生成两盘到 runs/zero-9x9/inspection/match-42-_1_10ews/。均 97 手、白胜 8.5，完整合法回放且每手达到最高访问次数。双方同模型、空盘、零温度导致两盘着法相同（True）。这是新生成的诊断对局，不是旧评估恢复；未改训练权重或 replay。

## 2026-10-20 低概率探索源码核对

- 对照本地 KataGo d91ea855 和原始 replay：第 3 手 59 个各占0.5%的落点合计29.5%，当前 T=1 会原样抽样。
- 记录 KataGo 示例温度衰减、回顾性搜索权重修正、LCB、塑形噪声，以及 chosenMovePrune=1 不保证排除单访问点的边界。详见 diagnostics-2026-10-20.md 新节。
- 本轮仅诊断及文档更新，没有修改训练策略/代码/权重/replay。
