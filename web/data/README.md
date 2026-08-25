# web/data 说明

本目录存放每日 Edition 展示数据，由产出工作流（`scripts/finalize_edition.py`）生成：

- `edition.json` — 最新一期 Edition
- `editions.json` — 目录（date → path 映射）
- `upcoming.json` — 近期日程
- `edition-<YYYY-MM-DD>.json` — 历史各期（由 `editions.json` 引用）

当前为空骨架（无历史内容）。每日产出后由发布流程填充并提交。
