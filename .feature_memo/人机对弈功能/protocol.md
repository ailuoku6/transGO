# Protocols and Contracts

## API Contracts

- `GET /api/weights` 返回排序后的权重条目 `{id,label,role,modified,external}`；首项为默认最新 best。
- `GET /api/state` 返回棋盘、合法动作、行棋方、终局、所选 checkpoint、预算、最后引擎搜索和是否可悔棋。
- `POST /api/new` 接受 `weight,size,komi,simulations,human_color`；白棋玩家开局时立即生成黑棋引擎应手。
- `POST /api/play` 接受行优先 `action`（`size*size` 为 pass），然后自动执行引擎应手。
- `POST /api/undo` 撤销最近一手人类着法及其后的引擎应手；没有人类着法时拒绝。
- `POST /api/analyze` 仅在人类行棋时返回前五推荐、访问数和当前方网络胜率估计。
- `POST /api/resign` 标记认输；`GET /api/sgf` 导出当前棋谱。

## Data Models

- `human_color`/`player`：黑为 `1`，白为 `-1`；`cells` 为行优先棋盘编码。
- AI 落子记录选中动作、访问数、总模拟数和耗时；终局结果由 native KataGo rules 计算。
- SGF 注释包含模型身份、训练 step、预算以及每手搜索摘要；认输记为 `B+R`/`W+R`。

## Storage and Config

- 对局仅保存在服务进程内存，用户通过导出下载 SGF；无权重和 replay 写入。
- `play` CLI 默认配置文件 `configs/tiny.yaml`，实际模型配置从选中 checkpoint 读取。可用 `--checkpoint` 和 `--port` 覆盖初始权重/端口。

## Compatibility Notes

- 不修改 checkpoint、Config、训练 replay 或 JSON-RPC schema；无新增运行依赖。
