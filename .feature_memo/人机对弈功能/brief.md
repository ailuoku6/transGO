# Brief

## User Goal

- 提供人机对弈功能，用对局观察项目 checkpoint 的实际表现；默认使用最新 `best.pt`，并允许选择其他权重。

## Scope

- 在本机浏览器对弈，复用现有 C++ 棋规、MCTS 和 PyTorch checkpoint。
- 支持黑白、9/13/19 路、搜索预算、落子、pass、悔棋、认输、局面分析和 SGF 导出。
- 将引擎最新判断和搜索访问数展示在页面。

## Non-Goals

- 本需求不改变训练、评测晋级或权重文件，也不宣称获得正式棋力等级。
- 暂不实现远程多人、网络公开服务或外部围棋 GUI 协议。

## Acceptance Criteria

- `transformer-go play` 可启动只监听回环地址的网页对弈。
- 默认从 `runs/` 中按修改时间选择最新 `best.pt`；对局开始时可切换到其他 checkpoint。
- AI 轮到时用当前 checkpoint 的搜索配置/用户选定预算按最高访问数自动落子。
- 黑白轮换、悔棋、终局计分和 SGF 导出符合规则服务。
- API 与 UI 测试通过，README 有启动说明。

## Open Questions

- None.
