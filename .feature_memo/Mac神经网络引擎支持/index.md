# Mac 神经网络引擎支持

- Status: done
- Last updated: 2026-10-25
- Current focus: 可选 Core ML 后端及使用说明完成，标准测试、真实硬件及完整并行闭环验证通过。
- Workspace: `transformerGo`

## 文档索引

- `../recording-preferences.md`：项目记录偏好
- `brief.md`：目标、范围和验收
- `tech.md`：实现和约束
- `protocol.md`：CLI、推理接口、缓存和兼容
- `files.md`：改动文件
- `decisions.md`：决策
- `changes.md`：验证与变更
- `handoff.md`：接续入口
- `benchmark.json`：三次搜索原始耗时、数值误差和静态设备计划
- `integration.json`：真实临时并行闭环摘要

## 当前结论

- 原 MPS 路径使用 GPU，现新增 Core ML CPU_AND_NE 推理；训练继续 PyTorch/MPS。
- 默认仍为 torch；Config/checkpoint/replay schema 不变，不重启既有训练。
- 本机 step327000 的 9x9/200 次空盘搜索，中位耗时 MPS 0.320 秒、Core ML batch16 为 0.104 秒；静态计划225个算子首选 NE，未测实时利用率。
- 真实双 worker 自博弈、MPS训练和双 worker Core ML评估已完成一轮，原有运行目录仅被读取。

## 后续

- 3/9/19 路真实转换、全部预测头/归一化概率和缓存测试均通过（3 passed）。
- 长时间多局面吞吐、功耗和棋力 A/B 均未测量。
