# Files

## Recently Modified

| Path | Why it matters | Status |
| --- | --- | --- |
| `transformer_go/play.py` | 权重扫描、单局状态、MCTS 应手、悔棋、分析和 SGF | modified |
| `transformer_go/web.py` | 本地 HTTP API 与静态资源服务 | modified |
| `transformer_go/static/index.html` | 对弈页面结构与操作面板 | modified |
| `transformer_go/static/style.css` | 响应式对弈界面与棋盘样式 | modified |
| `transformer_go/static/app.js` | 棋盘绘制和浏览器交互 | modified |
| `transformer_go/cli.py` | `play` 命令入口 | modified |
| `tests/test_services.py` | 权重选择、对局流程、分析、悔棋、SGF/认输测试 | modified |
| `README.md` | 本地启动和功能说明 | modified |

## Important Context

| Path | Why it matters |
| --- | --- |
| `transformer_go/search.py` | MCTS、树会话和模型分析既有实现 |
| `transformer_go/checkpoint.py` | 安全读取 checkpoint 与配置 |
| `transformer_go/_native` | 原生规则核心；Board immutable play() 返回新状态 |
| `.feature_memo/transformer围棋引擎/` | 既有引擎 memo 有用户未提交改动，本需求未修改 |
