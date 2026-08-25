# Graph Shell 金标场景离线指标（Slice 7 / G5b-2 Wave E）

- 生成时间：`2026-08-21T00:00:00+00:00`
- implementation_head：`6b317fd1ea1c4d722bc4a37c68c7c649f50d9d9b`
- artifact_generation_head：`6b317fd1ea1c4d722bc4a37c68c7c649f50d9d9b`
- review_head：`None`

## E3 解释边界

Offline scripted-model scenarios prove the tools/feedback support the target behaviors, the state machine and formal-graph results are correct, and the golden paths are reproducible. They do NOT prove a real LLM will take these actions — real agent behavior requires the live canary.

## 场景结果

| 场景 | 标题 | 通过 | 失败项 |
|---|---|---|---|
| G1 | 重复事件复用与更正 | ✅ | — |
| G2 | 同名 concept 显式决策 | ✅ | — |
| G3 | 关系表达 | ✅ | — |
| G4 | 零连接对象处理 | ✅ | — |
| G5 | 跨 wake 连续性 | ✅ | — |
| G6 | 主动更正与自我复查 | ✅ | — |
| G7 | 恢复与人工恢复 | ✅ | — |

## 行为指标

| 指标 | 值 |
|---|---|
| referent reuse count | 16 |
| referent reuse rate | 0.762 |
| duplicate escape count | 0 |
| object_ref assertion count | 8 |
| object_ref assertion rate | 0.667 |
| literal entity-hint count | 0 |
| supersede formed count | 2 |
| inspect use count | 3 |
| diff use count | 1 |
| inspect/diff-then-modify count | 2 |
| blocker reported / fixed | 1 / 1 |
| blocker fix rate | 1.0 |
| staged_unpublished count | 3 |
| resume count | 1 |
| abandon count | 3 |
| unhandled exception count | 0 |
| formal commits per wake | 8 commits / 10 wakes |
| turns (fixture) | 26 |
| tokens | fixture (scripted model; not measured) |
| cost (usd) | fixture (scripted model; 0.0 by construction) |
