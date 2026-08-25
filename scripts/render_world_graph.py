"""Render / export the formal world graph of a world.sqlite3 database.

Tiny, dependency-free graph visualizer for the formal cognition graph:

- nodes: formal objects (entities / events / concepts), sized by degree
- edges: assertions between objects, colored by predicate, faded when
  superseded (the graph keeps corrections as supersede chains)
- footer: object / assertion counts and the number of formal commits
  (each wake publishes at most one commit)

Optional flags:

- ``--color-by-wake`` colors nodes by the wake that created them, so the
  graph shows how knowledge accumulates run after run
- ``--json out.json`` exports the graph as plain JSON (objects, edges,
  commits) instead of an SVG — the same data a web frontend can load

Usage (from the repository root):

    python scripts/render_world_graph.py data/demo-world.sqlite3 --out my-world.svg
    python scripts/render_world_graph.py data/demo-world.sqlite3 --color-by-wake --out my-world.svg
    python scripts/render_world_graph.py data/demo-world.sqlite3 --json my-world.json

Renders read-only from the given world database — it never writes to it.
SVG output is self-contained and renders inline in the README on GitHub.

"""

# ruff: noqa: T201 — this is a CLI tool; prints are the CLI surface

from __future__ import annotations

import argparse
import math
import random
import sqlite3
from pathlib import Path

CANVAS_W = 1600
CANVAS_H = 1080
MARGIN = 60

KIND_COLORS = {
    "entity": "#3b82f6",  # blue
    "event": "#f59e0b",   # amber
    "concept": "#10b981",  # green
    "place": "#8b5cf6",   # violet
}
DEFAULT_KIND_COLOR = "#64748b"  # slate

PREDICATE_COLORS = {
    "participant": "#3b82f6",
    "part_of": "#8b5cf6",
    "score": "#f59e0b",
    "located": "#10b981",
    "happened_at": "#10b981",
    "related": "#94a3b8",
}
DEFAULT_PREDICATE_COLOR = "#94a3b8"

# Per-wake palette (--color-by-wake): one hue per publishing wake, in commit
# order, so the graph shows how knowledge accumulates wake after wake.
WAKE_COLORS = ["#2563eb", "#f59e0b", "#10b981", "#8b5cf6", "#ec4899", "#0891b2"]

FONT = "'Segoe UI', 'Microsoft YaHei', 'PingFang SC', sans-serif"


def _load(path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        object_cols = {c[1] for c in con.execute("PRAGMA table_info(objects)")}
        provisional = "provisional" in object_cols
        query = (
            "SELECT id, kind, canonical_name FROM objects"
            + (" WHERE provisional = 0" if provisional else "")
        )
        objects = [
            {"id": str(r[0]), "kind": str(r[1] or "other"), "name": str(r[2])}
            for r in con.execute(query)
        ]
        assertions = [
            {
                "subject": str(r[0]),
                "predicate": str(r[1]),
                "object": r[2],
                "literal": r[3],
                "superseded_at": r[4],
                "confidence": float(r[5] or 0.5),
            }
            for r in con.execute(
                "SELECT subject_id, predicate, object_id, literal_json, "
                "superseded_at, confidence FROM assertions"
            )
        ]
        try:
            commits = [
                dict(row)
                for row in con.execute(
                    "SELECT wake_id, commit_id, created_at, receipt_json "
                    "FROM finalize_receipts ORDER BY created_at"
                )
            ]
        except sqlite3.OperationalError:
            commits = []
    finally:
        con.close()
    return objects, assertions, commits


def _layout(objects: list[dict], edges: list[tuple[int, int]]) -> dict[int, tuple[float, float]]:
    """Deterministic force-directed layout (seeded, no external deps)."""
    rng = random.Random(7)
    n = len(objects)
    pos = {i: (rng.uniform(-1, 1), rng.uniform(-1, 1)) for i in range(n)}
    adjacency = {i: set() for i in range(n)}
    for a, b, *_rest in edges:
        if a != b:
            adjacency[a].add(b)
            adjacency[b].add(a)
    for _ in range(500):
        for i in range(n):
            fx = fy = 0.0
            xi, yi = pos[i]
            for j in range(n):
                if i == j:
                    continue
                xj, yj = pos[j]
                dx, dy = xj - xi, yj - yi
                d2 = dx * dx + dy * dy
                if d2 < 1e-6:
                    dx, dy, d2 = rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01), 1e-6
                d = math.sqrt(d2)
                # repulsion
                rep = 0.85 / d2
                fx -= rep * dx / d
                fy -= rep * dy / d
                # spring on linked pairs
                if j in adjacency[i]:
                    pull = 0.05 * (d - 1.35)
                    fx += pull * dx / d
                    fy += pull * dy / d
            # gentle gravity toward center
            fx -= 0.015 * xi
            fy -= 0.015 * yi
            pos[i] = (xi + fx, yi + fy)
    # spread the closest pairs apart once more (label-aware margin)
    for _ in range(80):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j][0] - pos[i][0]
                dy = pos[j][1] - pos[i][1]
                d2 = dx * dx + dy * dy
                if d2 < 0.22:
                    d = math.sqrt(d2) or 1e-6
                    push = 0.04 * (0.22 - d2) / d
                    pos[i] = (pos[i][0] - push * dx, pos[i][1] - push * dy)
                    pos[j] = (pos[j][0] + push * dx, pos[j][1] + push * dy)
                    moved = True
        if not moved:
            break
    # center the laid-out graph on the usable canvas area
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    usable_w = CANVAS_W - 2 * MARGIN - 260
    usable_h = CANVAS_H - 2 * MARGIN - 120
    scale = min(usable_w / span_x, usable_h / span_y)
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    center_x = MARGIN + usable_w / 2
    center_y = MARGIN + usable_h / 2
    return {
        i: (
            center_x + (x - cx) * scale,
            center_y + (y - cy) * scale,
        )
        for i, (x, y) in pos.items()
    }


def _display_name(name: str, max_chars: int = 12) -> str:
    name = name.strip()
    return name if len(name) <= max_chars else name[: max_chars - 1] + "…"


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _wake_colors(
    objects: list[dict], commits: list[dict]
) -> tuple[dict[str, str], list[dict]]:
    """Map wake id prefixes to palette hues, in commit order then appearance.

    ``objects`` gain a ``wake`` field (their id prefix ``{wake_id}:``); returns
    ``(wake -> color, ordered wake list with counts)``.
    """
    ordered: list[str] = []
    seen: set[str] = set()
    for wake in [str(c["wake_id"]) for c in commits] + [
        str(o["id"].split(":", 1)[0]) for o in objects
    ]:
        if wake and wake not in seen:
            seen.add(wake)
            ordered.append(wake)
    counts: dict[str, int] = {}
    for o in objects:
        wake = str(o["id"].split(":", 1)[0])
        o["wake"] = wake
        counts[wake] = counts.get(wake, 0) + 1
    palette = {
        wake: WAKE_COLORS[i % len(WAKE_COLORS)] for i, wake in enumerate(ordered)
    }
    legend = [
        {"wake": wake, "color": palette[wake], "count": counts.get(wake, 0)}
        for wake in ordered
    ]
    return palette, legend


def render(path: Path, out: Path, color_by_wake: bool = False) -> None:
    """Render the world graph of ``path`` to an SVG at ``out`` (read-only)."""
    objects, assertions, commits = _load(path)
    wake_palette: dict[str, str] = {}
    wake_legend: list[dict] = []
    if color_by_wake:
        wake_palette, wake_legend = _wake_colors(objects, commits)
    index = {o["id"]: i for i, o in enumerate(objects)}
    edges = []  # (a_idx, b_idx, predicate, confidence, superseded)
    for a in assertions:
        if a["object"] is None or a["subject"] not in index or a["object"] not in index:
            continue
        edges.append(
            (
                index[a["subject"]],
                index[a["object"]],
                a["predicate"],
                a["confidence"],
                a["superseded_at"] is not None,
            )
        )
    pos = _layout(objects, edges)

    degree = [0] * len(objects)
    for a, b, *_ in edges:
        degree[a] += 1
        degree[b] += 1
    max_deg = max(degree) if degree else 1

    # ── SVG ──────────────────────────────────────────────────────────────
    s = []
    s.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" '
        f'height="{CANVAS_H}" viewBox="0 0 {CANVAS_W} {CANVAS_H}">'
    )
    s.append(
        '<defs><marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" '
        'markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
        '<path d="M0,0 L8,4 L0,8 z" fill="#94a3b8"/></marker></defs>'
    )
    s.append(f'<rect width="{CANVAS_W}" height="{CANVAS_H}" fill="#ffffff"/>')

    # edges
    for a, b, predicate, confidence, superseded in edges:
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        color = PREDICATE_COLORS.get(predicate, DEFAULT_PREDICATE_COLOR)
        opacity = 0.16 if superseded else min(0.2 + 0.55 * confidence, 0.85)
        dash = "6,5" if superseded else "none"
        s.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-opacity="{opacity:.2f}" stroke-width="1.6" '
            f'stroke-dasharray="{dash}" marker-end="url(#arrow)"/>'
        )

    # nodes (draw text last on top)
    for i, o in enumerate(objects):
        x, y = pos[i]
        r = 8 + 10 * (degree[i] / max_deg)
        if color_by_wake:
            color = wake_palette.get(o.get("wake", ""), DEFAULT_KIND_COLOR)
        else:
            color = KIND_COLORS.get(o["kind"], DEFAULT_KIND_COLOR)
        s.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}" '
            f'fill-opacity="0.88" stroke="#ffffff" stroke-width="2"/>'
        )
    for i, o in enumerate(objects):
        x, y = pos[i]
        r = 8 + 10 * (degree[i] / max_deg)
        name = _display_name(o["name"])
        size = 11 if len(o["name"]) <= 8 else 9.5
        s.append(
            f'<text x="{x:.1f}" y="{y + r + 15:.1f}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="{size}" fill="#334155">'
            f"{_esc(name)}</text>"
        )

    # legend
    lx = CANVAS_W - 235
    ly = MARGIN + 20
    s.append(
        f'<text x="{lx}" y="{ly}" font-family="{FONT}" font-size="14" '
        f'font-weight="600" fill="#0f172a">图例</text>'
    )
    ly += 26
    if color_by_wake:
        for entry in wake_legend:
            s.append(
                f'<circle cx="{lx + 6}" cy="{ly - 4}" r="5" fill="{entry["color"]}"/>'
                f'<text x="{lx + 18}" y="{ly}" font-family="{FONT}" font-size="12" '
                f'fill="#334155">{entry["wake"]} · {entry["count"]} 对象</text>'
            )
            ly += 20
    else:
        for kind, color in KIND_COLORS.items():
            s.append(
                f'<circle cx="{lx + 6}" cy="{ly - 4}" r="5" fill="{color}"/>'
                f'<text x="{lx + 18}" y="{ly}" font-family="{FONT}" font-size="12" '
                f'fill="#334155">{kind}</text>'
            )
            ly += 20
    s.append(
        f'<line x1="{lx}" y1="{ly - 2}" x2="{lx + 30}" y2="{ly - 2}" '
        f'stroke="#94a3b8" stroke-dasharray="6,5"/>'
        f'<text x="{lx + 36}" y="{ly}" font-family="{FONT}" font-size="12" '
        f'fill="#334155">已 supersede</text>'
    )
    ly += 26
    s.append(
        f'<text x="{lx}" y="{ly}" font-family="{FONT}" font-size="12" fill="#64748b">'
        f"{len(objects)} 对象 · {len(edges)} 关系断言</text>"
    )
    if commits:
        s.append(
            f'<text x="{lx}" y="{ly + 18}" font-family="{FONT}" font-size="12" '
            f'fill="#64748b">{len(commits)} 次正式提交（wake）</text>'
        )

    s.append("</svg>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(s), encoding="utf-8")
    print(f"rendered {len(objects)} objects / {len(edges)} edges"
          f"{f' / {len(commits)} commits' if commits else ''} -> {out}")


def export_json(path: Path, out: Path) -> None:
    """Export the formal graph of ``path`` as plain JSON (web-ready).

    The export mirrors the SQLite formal graph: objects (with their creating
    wake id prefix), assertions (supersede chains included) and finalize
    receipts. A future web frontend can load this instead of rendering it.
    """
    import json

    objects, assertions, commits = _load(path)
    for o in objects:
        o["wake"] = str(o["id"].split(":", 1)[0])
    index = {o["id"]: i for i, o in enumerate(objects)}
    edges = []
    for a in assertions:
        if a["object"] is None or a["subject"] not in index or a["object"] not in index:
            continue
        edges.append(
            {
                "subject": a["subject"],
                "predicate": a["predicate"],
                "object": a["object"],
                "confidence": a["confidence"],
                "superseded": a["superseded_at"] is not None,
            }
        )
    data = {
        "objects": objects,
        "edges": edges,
        "commits": [
            {
                "wake_id": c["wake_id"],
                "commit_id": c["commit_id"],
                "created_at": c["created_at"],
            }
            for c in commits
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(
        f"exported {len(objects)} objects / {len(edges)} edges / "
        f"{len(commits)} commits -> {out}"
    )


def main() -> None:
    """CLI entry: render a world graph SVG and/or export it as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("world_db", type=Path, help="path to a world.sqlite3")
    parser.add_argument(
        "--out", type=Path, default=Path("world-graph.svg"),
        help="SVG output path (only when --json is not given)",
    )
    parser.add_argument(
        "--color-by-wake",
        action="store_true",
        help="color nodes by the wake that created them (shows accumulation)",
    )
    parser.add_argument(
        "--json", type=Path, default=None,
        help="export the graph as JSON to this path instead of an SVG",
    )
    args = parser.parse_args()
    if not args.world_db.exists():
        parser.error(f"world database not found: {args.world_db}")
    if args.json is not None:
        export_json(args.world_db, args.json)
    else:
        render(args.world_db, args.out, color_by_wake=args.color_by_wake)


if __name__ == "__main__":
    main()
