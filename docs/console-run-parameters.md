# Console 运行参数白名单

日期：2026-08-23 ｜ 定位：console（`POST /api/runs`）与 world-agent CLI 之间的参数契约。
前端接线方（codex）以本文档为准；修改任何透传关系须同步本表。

## 可配置（前端可传，经 CreateRunRequest 透传）

| 字段 | 类型/校验 | CLI 参数 | 来源优先级 |
|---|---|---|---|
| `profile_id` | str（必填） | —（路由到 profile） | 请求 |
| `perspective` / `mission` | str ≤20000，互斥，空串→中性 | `--perspective` | 请求（缺省不传） |
| `mode` | `broad` \| `deep` | `--mode` | 请求 → profile.defaults.mode |
| `object_id` | str ≤200，仅 deep | `--object-id` | 请求（缺省不传） |
| `max_turns` | int 1..200 | `--max-turns` | 请求 → profile.defaults.max_turns |
| `max_cost_usd` | float >0 ≤1000 | `--max-cost-usd` | 请求 → profile.defaults.max_cost_usd |
| `adapters` | str \| list[str] | `--adapters` | 请求 → profile.defaults.adapters |
| `thinking` | bool | `--thinking` | 请求 → profile.defaults.thinking |
| `reasoning_effort` | `high` \| `max`（仅 thinking 开启时生效） | `--reasoning-effort` | 请求 → profile.defaults.reasoning_effort |

profile 级默认存于 `RunDefaults`（profile JSON `defaults` 段），前端 GET `/api/profiles` 可见、
PUT 可编辑。请求字段 `null` = 用 profile 默认；显式值 = 覆盖。
当前 Console 默认轮次兜底为 96，成本上限为 `null`（不限制）。主要资源边界仍由 CLI 的
上下文 Token 阈值控制：约 700,000 Token 提醒、800,000 Token 硬停止；轮次与成本仅是额外兜底。

## 内部推导（前端不可传，服务端由 profile 派生）

- `thread_id`：缺省按 profile 生成（`--thread-id`）
- `world_db` / `runtime_db`：profile 路径解析（隔离存储，registry 保证唯一）
- `domain_focus`：profile 域聚焦（snapshot 三方一致校验的一部分）
- `operator_instructions`：profile 字段（走 PUT /api/profiles/{id}，非 run 参数）

## 刻意不暴露（保持 CLI 侧；如前端确需请先过阶段 5 评审）

- `--graph-shell-restore`：恢复/重建类管理动作，要求 operator 意图，console 不提供按钮
  （2026-08-23 决策：仅 finalize 提供 console 手动入口，restore 保持 CLI-only）
- `--deadline-*`、context 阈值（soft reminder / hard cut）：时间/资源边界，CLI 侧默认即可
- `--replay-fixture`：离线重放专用

> 注：`--graph-shell-finalize-wake` 的 console 暴露面见下文"滞留清单与手动收尾 API"；
> 前端按钮接线（codex）以该节契约为准。

## 滞留清单与手动收尾 API（2026-08-23，服务端契约）

Owner 决策（方案 3）：收尾不是 agent 自动行为，而是 operator 从 console **滞留清单**
里**手动启动**的确定性动作（"程序都没有在运行，怎么会自动触发？"）。前端接线方
（codex）以此契约为准；服务端已实现 + 测试覆盖，本任务不碰 web/。

### `GET /api/profiles/{profile_id}/pending-wakes`

返回**可被 finalize 发布**的 wake 清单——与确定性 finalize 入口完全同构的 fail-closed
判据：有 world 侧 writer claim + 至少一行 active staging + 无 published receipt。

```jsonc
{
  "wakes": [
    {
      "wake_id": "wake-<id>",
      "staging": { "staged_objects": 1, "staged_assertions": 0, "staged_inquiries": 0 },
      "staging_total": 1,
      "claimed_by": "<thread-id>"   // 持有该 wake 声明的线程
    }
  ]
}
```

- 语义保证：清单里每一项，POST finalize 都会执行一次真实发布尝试（published 或带原因的
  blocked/compile_failed/commit_rejected）；**清单外的不在 finalize 域内**——无 claim 的
  残留行属于 restore（fail-closed `wake_unknown`），已发布 wake 即使有后续 staging 也排除
  （I1：发布后的新工作属于新 wake）。
- 空/未初始化库 → `200 {"wakes": []}`（只读，绝不创建库）。
- 未知 profile → `404 {"detail": ...}`。

### `POST /api/profiles/{profile_id}/pending-wakes/{wake_id}/finalize`

确定性发布该 wake 的 active staging：与 CLI `--graph-shell-finalize-wake` 同一入口
（`graph_shell_finalize_wake`），无需 model、无需 writer lease，同事务 + 幂等重放。
**这是 console 唯一的收尾写面**（其余全只读）。

| HTTP | body `status` | 含义 |
|---|---|---|
| 200 | `published` | 发布成功；`commit_id`/`committed_at`/`stats`/`item_ids` 完整 receipt |
| 200 | `already_published` | 幂等重放（同 `commit_id`，无重复写入） |
| 200 | `blocked` / `compile_failed` / `commit_rejected` / `nothing_to_finalize` | 发布尝试有因失败，staging 保留，可改后重试 |
| 404 | `wake_unknown` / `no_world` | wake 或 world 库不存在（清单外） |
| 422 | — | `wake_id` 为空/超 200 字符，或库级 IO 错误 |

- 前端建议：按钮 → 乐观触发 → 200 后**重新拉取清单**（发布成功的项消失）；blocked 等
  原因从 body `blockers`/`problems` 展示；429/超时无需 — 服务端单次执行，幂等可重发。
- 交互注意事项：POST 会真实写 world 库（发布提交）。按钮文案应表达"发布到正式记忆"，
  非"删除"。无确认弹窗的服务端不做强制（浏览器侧可自行确认）。

## 一致性契约

1. `_world_namespace`（请求→namespace）与 `_namespace_arguments`（namespace→args）成对，
   改动必须同步；snapshot 持久化 `world_args` 以便 `_default_runner` 重建。
2. `_default_runner` 重建时校验 domain_focus 与 profile_snapshot 三方一致，不一致 fail-closed
   （ValueError → run 失败，不启动 world）。
3. 新增 CLI 参数若需透传：CreateRunRequest + RunDefaults + 两个转换函数 + snapshot 字段
   + 回环测试（`_namespace_arguments` → `_parse_world_arguments` 等价）。
4. 校验失败统一 422（SystemExit 转 ValueError）。
