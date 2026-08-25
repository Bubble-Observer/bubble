"""Executable contracts for the console's static product shell."""

from __future__ import annotations

import json
import subprocess
from html.parser import HTMLParser
from pathlib import Path

_STATIC = Path(__file__).parents[2] / "src" / "leave_information_bubble" / "console" / "static"


class _Markup(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.elements: list[tuple[str, dict[str, str | None]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.elements.append((tag, dict(attrs)))


def _markup() -> _Markup:
    parser = _Markup()
    parser.feed((_STATIC / "index.html").read_text(encoding="utf-8"))
    return parser


def test_bubble_remains_the_product_and_agent_lobby_is_only_a_page() -> None:
    html = (_STATIC / "index.html").read_text(encoding="utf-8")

    assert "<title>Bubble · Agent 大厅</title>" in html
    assert "<strong>Bubble</strong>" in html
    assert "个人 Agent 控制台" in html
    assert '<span class="brand-mark" aria-hidden="true"></span>' in html
    assert '<p class="eyebrow">Your observers</p>' in html
    assert "<h1>你的 Agent 大厅</h1>" in html
    assert "连接与凭据" in html
    assert "hero-orbit" not in html
    assert html.count('class="expert-settings"') == 3
    assert 'aria-hidden="true">B</span>' not in html
    assert "Agent Space" not in html
    assert "你的专属智能体空间" not in html


def test_product_shell_has_unique_landmarks_graph_and_layered_settings() -> None:
    markup = _markup()
    identified = [(tag, attrs) for tag, attrs in markup.elements if attrs.get("id")]
    ids = [str(attrs["id"]) for _, attrs in identified]
    by_id = {str(attrs["id"]): (tag, attrs) for tag, attrs in identified}

    assert len(ids) == len(set(ids))
    assert by_id["agent-lobby"][0] == "section"
    assert "hidden" not in by_id["agent-lobby"][1]
    assert "hidden" in by_id["agent-workspace"][1]
    assert by_id["workspace-navigation"][0] == "nav"
    assert by_id["memory-graph"][0] == "svg"
    assert by_id["memory-graph"][1]["role"] == "img"
    assert by_id["memory-graph-source"][0] == "small"
    assert by_id["memory-graph-stats"][0] == "div"
    assert "memory-edge-arrow" not in by_id
    assert by_id["memory-graph-window-controls"][0] == "div"
    assert by_id["memory-graph-previous"][0] == "button"
    assert by_id["memory-graph-next"][0] == "button"
    assert by_id["memory-graph-auto"][1]["aria-pressed"] == "true"
    assert any(attrs.get("class") == "graph-stage-footer" for _, attrs in markup.elements)
    assert by_id["local-settings-form"][0] == "form"
    assert by_id["connections-button"][0] == "button"
    profile_adapters = [
        attrs
        for tag, attrs in markup.elements
        if tag == "input" and "profile-adapter" in str(attrs.get("data-role", ""))
    ]
    run_adapters = [
        attrs
        for tag, attrs in markup.elements
        if tag == "input" and "run-adapter" in str(attrs.get("data-role", ""))
    ]
    assert {attrs["value"] for attrs in profile_adapters} == {
        "public-web",
        "bilibili",
        "hupu",
        "nga",
    }
    assert {attrs["value"] for attrs in run_adapters} == {
        "public-web",
        "bilibili",
        "hupu",
        "nga",
    }


def test_quick_create_keeps_generated_id_optional_and_mobile_connections_visible() -> None:
    markup = _markup()
    profile_id = next(attrs for tag, attrs in markup.elements if tag == "input" and attrs.get("name") == "id")
    css = (_STATIC / "app.css").read_text(encoding="utf-8")

    assert "required" not in profile_id
    assert ".top-navigation, .header-actions .ghost" not in css


def test_ui_model_normalizes_adapter_and_connection_payloads_in_node() -> None:
    script = (
        "const ui=require(process.argv[1]);"
        "const result={"
        "adapters:ui.normalizeAdapters([' public-web ', '', 'nga', 'nga']),"
        "settings:ui.nonEmptySettings({deepseek_api_key:'  ',deepseek_model:' deepseek-v4 '})"
        ",current:ui.isCurrentSelection({selectionEpoch:2,profile:{id:'agent-b'}},2,'agent-b')"
        ",stale:ui.isCurrentSelection({selectionEpoch:3,profile:{id:'agent-b'}},2,'agent-b')"
        "};process.stdout.write(JSON.stringify(result));"
    )
    completed = subprocess.run(
        ["node", "-e", script, str(_STATIC / "app.js")],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "adapters": ["public-web", "nga"],
        "settings": {"deepseek_model": "deepseek-v4"},
        "current": True,
        "stale": False,
    }


def test_parameterized_loaders_are_not_registered_as_raw_event_handlers() -> None:
    javascript = (_STATIC / "app.js").read_text(encoding="utf-8")

    assert "addEventListener('click', loadRuns)" not in javascript
    assert "addEventListener('click', loadPrompt)" not in javascript
    assert "addEventListener('change', loadPrompt)" not in javascript


def test_prompt_layer_ui_describes_operator_guidance_as_effective_within_boundaries() -> None:
    html = (_STATIC / "index.html").read_text(encoding="utf-8")
    javascript = (_STATIC / "app.js").read_text(encoding="utf-8")

    assert "操作员长期引导" in html
    assert "可填写长期优先级、排除项与解释偏好" in html
    assert "保存长期引导" in html
    assert "操作员长期引导" in javascript
    assert "高级设置 · 边界内生效" in javascript
    assert "高级设置 · 可忽略" not in javascript


def test_run_controls_match_the_active_console_contract() -> None:
    html = (_STATIC / "index.html").read_text(encoding="utf-8")
    javascript = (_STATIC / "app.js").read_text(encoding="utf-8")
    markup = _markup()
    inputs = [attrs for tag, attrs in markup.elements if tag == "input"]
    selects = [attrs for tag, attrs in markup.elements if tag == "select"]

    for retired in ("wake_protocol", "memory_navigation", "digest_cache_reuse"):
        assert retired not in html
        assert retired not in javascript

    max_turns = next(attrs for attrs in inputs if attrs.get("name") == "max_turns")
    default_max_turns = next(attrs for attrs in inputs if attrs.get("name") == "default_max_turns")
    max_cost = next(attrs for attrs in inputs if attrs.get("name") == "max_cost_usd")
    default_max_cost = next(attrs for attrs in inputs if attrs.get("name") == "default_max_cost_usd")
    assert max_turns["max"] == default_max_turns["max"] == "200"
    assert max_cost["min"] == default_max_cost["min"] == "0.01"
    assert max_cost["max"] == default_max_cost["max"] == "1000"

    assert any(attrs.get("name") == "thinking" for attrs in inputs)
    assert any(attrs.get("name") == "default_thinking" for attrs in inputs)
    assert any(attrs.get("name") == "reasoning_effort" for attrs in selects)
    assert any(attrs.get("name") == "default_reasoning_effort" for attrs in selects)


def test_run_history_explains_cumulative_tokens_and_separates_outcome_fields() -> None:
    html = (_STATIC / "index.html").read_text(encoding="utf-8")
    javascript = (_STATIC / "app.js").read_text(encoding="utf-8")
    css = (_STATIC / "app.css").read_text(encoding="utf-8")

    assert "Token 与成本均按本次运行累计" in html
    assert "输入 + 输出累计，非峰值" in javascript
    assert "当前轮上下文" in javascript
    assert "最后一次调用输入" in javascript
    assert "root.className = 'run-monitor'" in javascript
    assert '<dl class="outcome-written">' in javascript
    assert "<dt>质询</dt>" in javascript
    assert ".outcome-written { display: grid" in css
    assert ".event-list > .empty-state { display: grid" in css
    assert "run-overview-cards" in javascript
    assert "run-token-cards" in javascript
    assert ".run-token-cards { grid-template-columns:" in css
    assert "app.css?v=20260824-5" in html
    assert "app.js?v=20260824-5" in html
    assert "marker-end" not in javascript
    assert "连线表示可从任一端浏览的关联" in javascript
    assert "memory/graph?limit=24&window=" in javascript
    assert "root.classList.toggle('has-selection', Boolean(node))" in javascript
    assert ".graph-detail.has-selection { justify-content: flex-start; }" in css
    assert ".graph-stage-footer { position: relative;" in css


def test_memory_page_exposes_the_manual_pending_publish_surface() -> None:
    markup = _markup()
    by_id = {str(attrs["id"]): (tag, attrs) for tag, attrs in markup.elements if attrs.get("id")}
    javascript = (_STATIC / "app.js").read_text(encoding="utf-8")

    assert by_id["pending-memory"][0] == "section"
    assert by_id["pending-wakes-list"][1]["aria-live"] == "polite"
    assert "/pending-wakes" in javascript
    assert "/finalize" in javascript
