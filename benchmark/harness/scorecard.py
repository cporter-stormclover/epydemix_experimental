"""Scorecard: pass rate, cost and reproducibility per agent and condition."""

import json
import statistics
from pathlib import Path
from typing import Any, Dict, List


def load_rows(path: Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def scorecard(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (row["agent"], row.get("model"), row["condition"]), []
        ).append(row)
    table = []
    for (agent, model, condition), items in sorted(
        groups.items(), key=lambda kv: str(kv[0])
    ):
        table.append(
            {
                "agent": agent,
                "model": model,
                "condition": condition,
                "sessions": len(items),
                "tasks": len({r["task"] for r in items}),
                "pass_rate": _mean(r["pass_rate"] for r in items),
                "fully_passed": _mean(float(r["passed"]) for r in items),
                "median_cost_usd": _median(r.get("cost_usd") for r in items),
                "median_turns": _median(r.get("turns") for r in items),
                "median_wall_clock_s": _median(r.get("wall_clock_s") for r in items),
                "reproducibility_rate": _mean(r.get("reproduced") for r in items),
            }
        )
    return table


def _fmt(value, pct=False):
    if value is None:
        return "–"
    if pct:
        return f"{value:.0%}"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def render_markdown(table: List[Dict[str, Any]]) -> str:
    lines = [
        "| Agent | Condition | Sessions | Pass rate | Fully passed | Median cost (USD) "
        "| Median turns | Median wall clock (s) | Reproducible |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in table:
        agent = r["agent"] + (f" ({r['model']})" if r.get("model") else "")
        lines.append(
            f"| {agent} | {r['condition']} | {r['sessions']} | {_fmt(r['pass_rate'], True)} "
            f"| {_fmt(r['fully_passed'], True)} | {_fmt(r['median_cost_usd'])} "
            f"| {_fmt(r['median_turns'])} | {_fmt(r['median_wall_clock_s'])} "
            f"| {_fmt(r['reproducibility_rate'], True)} |"
        )
    return "\n".join(lines)
