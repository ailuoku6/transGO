# Technical Notes

## Current Approach

- 标准库 `ThreadingHTTPServer` 提供本地静态页面与 JSON API；绑定回环地址，API 访问共享一份加锁的对局状态。
- `PlayService` 从 checkpoint 构造现有 `Evaluator`，用 `SearchSession` 复用每局子树；所有落子经 native `Board` 合法性检查。
- 页面使用 SVG 绘制动态尺寸棋盘；原生 JavaScript/CSS，无新增第三方依赖。
- 默认从 `runs/**/*.pt` 查找 mtime 最新的 `best.pt`；其他已发现 checkpoint 在开新局时切换。

## Key Constraints

- 模型本身允许动态棋盘尺寸；checkpoint 的规则、贴目和搜索配置作为所选模型默认值。
- 对局预算在开局时固定，AI 以访问数最高动作落子；页面胜率是模型预测而非棋力评级。
- HTTP server 与资产服务仅面向本机，不暴露远程监听。

## Dependencies and Patterns

- 复用 `checkpoint.load_checkpoint`、`search`、`SearchSession`、`judgment.adjudicate`、`sgf.dumps` 和 `gtp.to_coord`。
- 静态资源随 `transformer_go` Python package 分发。

## Risks

- 大棋盘或高搜索预算会显著增加单步等待时间；UI 显示搜索中状态，预算上限为 5000。
- 轮流对局只维护一个当前 session；同机同时打开多个页面会共享棋局。
