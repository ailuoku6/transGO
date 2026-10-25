# 人机对弈功能

- Status: done
- Last updated: 2026-10-25
- Current focus: 本机网页对弈已完成并通过服务、浏览器、测试和打包检查。
- Workspace: `transformerGo`

## Document Map

- `../recording-preferences.md`: 项目级记录偏好
- `brief.md`: 用户目标与验收条件
- `tech.md`: 服务和 UI 架构
- `protocol.md`: HTTP API 与对局数据契约
- `files.md`: 文件边界
- `decisions.md`: 权重默认值与对局行为决策
- `changes.md`: 实施及验证记录
- `handoff.md`: 新线程续作提示

## Latest Summary

- Python 本地网页 API 与 SVG 围棋 UI 已实现，真实运行默认选择 `runs/zero-9x9/best.pt`。
- 对局复用 checkpoint 配置、Evaluator、MCTS 和 native Board；支持黑白、预算、悔棋、pass、认输、分析和 SGF。
- 全套验证为 151 passed、5 skipped；wheel 含静态资源；本地浏览器验证了真实 best 对局。

## Next Actions

- 用户可运行 `.venv/bin/transformer-go play` 启动页面；当前验证服务留在 `127.0.0.1:8766`。

## Open Questions

- None.
