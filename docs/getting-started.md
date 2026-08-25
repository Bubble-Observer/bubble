# 上手指南（Getting Started）

本文从零带你跑起来。两条路：

| 路径 | 需要什么 | 花多少钱 | 验证什么 |
|---|---|---|---|
| **离线 demo**（推荐先跑） | Python ≥ 3.11，无网络无 key | 0 | 协议与状态机完整闭环 |
| **真实 wake**（可选进阶） | DeepSeek API key + 可选 cookie | 付费（可设上限） | 真实 LLM 的行为 |

---

## 1. 环境要求

- **Python ≥ 3.11**（已在 3.11 / 3.14 验证）。检查：`python --version`
- 无需 GPU、无需网络（离线 demo）、无需任何外部服务
- 已测试平台：Windows、macOS、Linux（命令差异仅 venv 激活一行）

## 2. 安装

```bash
git clone https://github.com/Bubble-Observer/bubble.git
cd bubble

python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate

pip install -e .        # 普通运行
# 或：pip install -e ".[dev]"   # 额外安装测试工具（可选）
```

安装后会得到 `bubble-world` 命令（真实运行用）。

## 3. 离线 demo（第一次务必先跑）

```bash
python examples/offline_demo.py
```

成功时输出末尾是：

```text
wake 1: published  (commit demo-w1:finalize)
wake 2: published  (commit demo-w2:finalize)
formal graph: 3 objects, 4 assertions
supersede chain: [{'id': 'demo-w1:a3', 'literal': '3-0', 'supersedes_id': ''},
                  {'id': 'demo-w2:a1', 'literal': '3-1', 'supersedes_id': 'demo-w1:a3'}]

DEMO PASSED: four wakes. wake 2 reused the wake-1 event id and superseded its
score; wake 3 stopped without publishing (staging survived); wake 4 resumed
wake 3 and published Rookie.
and superseded the wake-1 score assertion.
```

demo 干了什么（全部使用真实组合根代码）：

1. **wake 1**：全新世界 → 检索记忆 → 创建事件 + 两个参与者 + 比分断言 →
   `graph_inspect` 检查就绪度 → `finalize_graph` 原子发布；
2. **wake 2**：再次检索 → **复用** wake 1 的事件 id（没有创建重复对象）→
   supersede 旧比分断言 → 再次发布。

中途的临时目录用完自动删除，不会在你的仓库留下任何数据库文件。

### demo 失败怎么办

- `command not found: python` → 未安装 Python 或未加入 PATH（用 `py -3.11` 试试）
- 其他报错 → 把完整输出发到仓库 Issue，注明操作系统与 `python --version`。

## 4. 真实 wake（可选进阶）

### 4.1 准备 API key（必填）

1. 到 <https://platform.deepseek.com/api_keys> 注册并创建一个 API key；
2. 复制 `.env.example` 为 `.env`（`.env` 已被 gitignore，永远不会被提交）：

```bash
cp .env.example .env     # Windows: copy .env.example .env
```

3. 编辑 `.env`，把 key 填入：

```text
DEEPSEEK_API_KEY=sk-你的真实key
```

### 4.2 准备 cookie（可选，但强烈建议）

**没有 cookie 也能运行**——匿名只读（B站搜索/视频信息等公开接口可用）。但部分
站点（尤其 NGA）的接口会要求浏览器会话，风控也更容易拦匿名请求。建议用
**专用账号**（不用自己主账号，避免封禁风险与隐私泄露）。

#### B站 SESSDATA（推荐先配这个）

1. 打开 <https://www.bilibili.com>，用你的账号登录；
2. 按 `F12` 打开开发者工具 → 切到 **Application（应用）** 标签；
3. 左侧 **Cookies → https://www.bilibili.com**；
4. 找到名为 **`SESSDATA`** 的那一行，双击 **Value** 列，全选复制；
5. 贴进 `.env`：

```text
BILIBILI_SESSDATA=粘贴在这里
```

> 只读操作（搜索、视频信息）只需要 `SESSDATA` 一个字段。不要把整行 Cookie
> 都贴进来，也不需要其他字段。

#### NGA Cookie（可选）

NGA 部分接口要求浏览器会话，否则报错（"Configure an authorized browser-established
NGA session cookie"）：

1. 用账号登录 <https://bbs.nga.cn>；
2. `F12` → **Network（网络）** 标签 → 刷新页面 → 点任意一个请求；
3. 在 **Request Headers（请求头）** 里找到 `Cookie:` 那一行，**复制整个值**
   （不含开头的 `Cookie:` 字样）；
4. 贴进 `.env`：

```text
NGA_COOKIE=粘贴在这里
```

> cookie 会过期（B站 SESSDATA 通常数月）。报错或素材变少时先重新获取再排查别的。

#### 安全须知（开源贡献者必读）

- `.env` 在 `.gitignore` 中，**绝不提交**真实 key / cookie；
- 使用专用账号，不用你的主账号；
- cookie 相当于你的登录态——不要把 `.env` 发到任何地方；
- 仓库永远只提交 `.env.example`（占位符）。

### 4.3 跑第一次真实 wake

```bash
bubble-world --thread-id my-first \
  --world-db data/demo-world.sqlite3 \
  --runtime-db data/demo-runtime.sqlite3 \
  --mode deep --domain lol_cn
```

- 数据库路径不存在会自动创建（`data/` 也在 gitignore 中）；
- 建议第一次加上成本上限：`--max-cost-usd 0.5`（累计模型成本到 $0.5 停止探索）；
- 正常情况会跑几分钟：检索素材 → 模型决定打补丁 → 可能调用 `finalize_graph`。
  wake 结束会打印运行报告，包括轮次、工具调用、成本与 `terminal_status`。

#### 常用参数

| 参数 | 作用 | 默认 |
|---|---|---|
| `--mode {broad,deep}` | broad=广撒网采样，deep=深挖单个话题 | `broad` |
| `--domain` | 领域焦点（目前唯一注册：`lol_cn`） | `lol_cn` |
| `--max-turns` | 最大轮次 | `96` |
| `--max-cost-usd` | 累计成本上限（推荐） | 无限制 |
| `--adapters` | 采集通道，逗号分隔 | `bilibili,nga,hupu,public-web` |
| `--resume` | 恢复上次未完成的 wake | 关闭 |
| `--thinking` | 开启 provider 推理模式（更贵、更慢） | 关闭 |
| `--live-hard-cap-usd` | 冻结模型的单次硬成本上限（需配 `--live-cost-audit-path`） | 关闭 |
| `--live-deadline-seconds` | 异常 wake 的截止时间（非认知时限） | 关闭 |
| `--graph-shell-status` | 只读管理入口：查看世界的写者租约/暂存/回执 | — |
| `--graph-shell-abandon <wake_id>` | 显式放弃某 wake 的暂存并释放租约 | — |

完整清单：`bubble-world --help`。

### 4.4 wake 状态说明

| terminal_status | 含义 | 怎么办 |
|---|---|---|
| `published` | 已发布正式认知提交 | 无，下次 wake 直接继承 |
| `staged_unpublished` | 模型构建了暂存图但没调用发布（宿主**不会**替你发布） | 用 `--resume` 恢复，或 `--graph-shell-abandon` 放弃 |
| `abandoned` | 已被显式放弃 | 无 |

> 这是刻意设计：宿主绝不在轮次/成本/期限边界替 agent 自动发布。一个 wake
> 没发布不算失败，是"待办"。

## 5. 常见问题（Troubleshooting）

**Q: 真实 wake 里转写/ASR 工具失败？**
本地没有 whisper 模型文件时（默认 `asr_local_files_only=true`，不自动下载），
该工具调用会失败——失败会作为结构化错误返回给模型（可恢复），不影响 wake
整体运行。不需要转写就在 `.env` 加：

```text
ASR_ENABLED=false
```

**Q: 素材变少 / 搜索返回空 / 通道报错？**
第三网站点改版、风控、cookie 过期都会导致。先重新获取 cookie（§4.2），
换网络/等待再试。这类问题在
[docs/limitations.md](limitations.md) 有完整声明。

**Q: 成本怎么看？**
运行报告里有本次成本。`.env` 配置只影响通道，成本上限看 `--max-cost-usd` /
`--live-hard-cap-usd`。离线 demo 永远免费。

**Q: 想清空世界重来？**
直接删除 `--world-db` 指向的文件即可（它只是你的世界库文件）。仓库里任何
`*.sqlite3` 都不会被 git 跟踪。

**Q: Windows 激活不了 venv？**
`python -m venv .venv` 后用 `.\\.venv\\Scripts\\activate`（PowerShell）或
`.venv\\Scripts\\activate.bat`（cmd）。执行策略报错时用
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`。

**Q: Python 版本太旧？**
需要 ≥ 3.11（f-string、typing 语法等）。`py -3.11 -m venv .venv` 可以指定版本。

**Q: 想跑测试套件？**

```bash
pip install -e ".[dev]"
PYTHONPATH=src pytest -q --import-mode=importlib --no-cov
```

（1287 个离线测试，不需要网络/key，约 3–7 分钟。）

## 6. 下一步

- 了解协议细节：[graph-shell.md](graph-shell.md)（英文）
- 了解代码结构：[architecture.md](architecture.md)（英文）
- 了解评测与真实运行证据：[evaluation.md](evaluation.md)（英文）
