# Decisions

## 2026-10-25 - 本机浏览器对弈

- Decision: 用 Python 标准库本地 HTTP 服务和包内静态 UI 提供实际可玩的界面；每个进程持有一个加锁的当前棋局。
- Rationale: 本地 GUI 是用户可直接操作的入口，并能原样复用 Python/C++ 棋规和推理，不增加服务框架依赖。
- Alternatives: GTP/JSON-RPC 仅适合连接外部 GUI/客户端，不提供本项目内可即开即用的界面。
- Consequences: 默认只绑定 loopback；单 server 只有一个当前对局；多局窗口隔离留待后续。

## 2026-10-25 - 默认权重和落子协议

- Decision: 选择 `runs/**/*.pt` 中 mtime 最新的 `best.pt` 为默认，额外列出候选、learner 和其他 checkpoint；换权重只在新局生效。AI 采用最高搜索访问数落子，预算可在开局设置。
- Rationale: 与 loop 的 best 晋级产物保持一致，并提供可复现、低随机性的实战体验；用户可提高搜索预算观察预算变化。
- Alternatives: 按训练 step 最大值排序不适用于不同 run；加噪声/温度抽样会让人机测试结果更难比较。
- Consequences: 文件修改时间用于 best 排序；UI 网络胜率仅为估计，不能替代成组棋力评测。
