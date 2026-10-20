# 最新接续：KataGo 式探索与 loss（2026-10-21）

用户授权的策略优化和逐轮 loss 已完成。先看 README 的“自博弈探索策略”“每轮训练 loss”、protocol.md 最新节和 changes.md。之前评估 SGF、零温度观棋及 diagnostics-2026-10-20.md 均保留；未提交本轮改动，未停止/重启用户训练，未改原 run 权重/replay。

- 默认 selfplay/loop runtime profile 为 katago，支持 --selfplay-policy legacy；不改 Config/checkpoint。温度 .75→.15、半衰期棋盘边长；根噪声采用总 concentration 10.83 的半均匀/半 log-policy 塑形；混合权重沿用 config。
- cpp/native.cpp 回顾性权重修正为简化 PUCT 的近似移植，未包括 LCB、分叉局或廉价搜索。原始 visits/policy/树预算不变，新增 selection_weights/selection_policy；selfplay 以未加温度的 selection_policy 为训练目标，温度只控制落子。保留高价值单访问点，不用概率硬门槛。
- training_metrics.py 每步 flush steps.jsonl，训练/模型保存成功后原子 summary.json。loop 在评估前追加 training-loss.csv，完整 iteration JSON 含 training 汇总。每次尝试独立 steps-START-* 目录，重复轮次可有多条 CSV；失败训练只留已完成步骤。独立 train 支持 --metrics-dir。
- 原 loop 命令可继续用，新进程默认启用新策略和日志；已运行的旧进程不热更新。旧四参数 actor 提交缺省 legacy 保持原 parent 行为，新 parent 显式传 profile。旧日志不能从 checkpoint 还原；旧 replay 策略标签不转换。
- native 已编译；专项 56 passed，全套 138 passed、5 skipped（当前工具环境 MPS 不可用）。临时真实并行 katago→串行 legacy 两轮通过：4 局 replay、4 局评估，训练 step 2→4，CSV 2 行。验证中断保留、续训不覆盖、汇总正确、树预算/软目标。目录见 changes.md。
- 前一需求已支持 evaluation/iteration-NNNN/match-*/ SGF 和独立 eval --temperature 0。默认评估保持 .5→.2、无根噪声、原始访问次数选点；不要把训练 profile 当成完整 KataGo 评估引擎。
- 9x9 诊断棋谱仍在 runs/zero-9x9/inspection/match-42-_1_10ews/，为当时 best 快照新生成的两盘零温度对局，非历史评估恢复。早期集中推理/树复用、CPU 基准与完整 SHA 见 changes.md。

后续仅有实验性工作：固定预算做新旧策略棋力 A/B，目标 MPS/CUDA 基准，监测 loss 与评估而非单凭棋谱观感。没有证实棋力提高或 GPU 提速；测试入口 .venv/bin/python -m pytest -q，native 修改后需 setup.py build_ext --inplace。
