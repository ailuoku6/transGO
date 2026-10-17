# 决策

## 2026-10-15

## 2026-10-16 - 自博弈最少手数

- 通过命令行 `--min-moves` 控制，不加入 checkpoint config，允许同一训练目录续跑时调整。
- 约束放在 C++ 搜索树的 legal mask，而非修改规则引擎；这样训练样本的规则合法 mask 仍保留 pass，网络会学习在阈值前降低 pass 概率。
- 评测不使用最少手数，避免晋级统计偏离标准对局协议。

## 2026-10-17 - 兼容既有 run 的 score utility

- 将已有 `score_mean` 通过 `weight*tanh(score_mean)` 接入叶节点回传；权重作为 CLI 运行时参数，不加入 checkpoint config，避免现有 9x9 checkpoint 因配置不一致无法恢复。
- 自博弈和评测使用同一个 score utility，保证晋级对局的搜索协议一致；权重 0 保留旧 WDL-only 行为。

- 用户指定 Python/PyTorch + C++，不采用 Rust 或纯 Python 搜索。
- Apple GPU 使用 PyTorch MPS，优先 float32，避免未经验证的 BF16/FP16 路径。
- 用中国风格面积 + PSK 作为明确的首版规则；不实现自动死棋判定。
- uncertainty head 用 Gaussian NLL 学习方差；ownership 用三分类保留中立区域。
- 只用真实终局辅助标签，截断局丢弃；认输棋谱只监督 policy/WDL。
- 初期 dense masked attention 易验证，后续再实现真正分块窗口。
- learner 进度独立于 best；候选晋级采用 KataGo gatekeeper 的累计 match points、平分接受和数学可确定时提前停止。
- 依赖首次 PyPI 下载超时，改清华 PyPI 镜像安装公开同名依赖，已完成。

## 2026-10-15 - 用户取消人类棋谱路线

- 替代此前预训练建议：从随机权重开始，仅自博弈。外部 SGF 转训练代码和 CLI 已删除。
- KataGo score/ownership/moves 辅助头仍由自身对局终局监督，符合纯自博弈约束。
- 保留旧工件用于对弈；不自动补写来源，新的 Zero 训练使用空运行目录。
