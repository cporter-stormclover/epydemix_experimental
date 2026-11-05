"""``epydemix sweep``: one config, many points, each a complete bundle.

A ``sweep`` block turns one config into a grid of configs::

    sweep:
      name: school_closure
      reference: baseline.epx            # optional: deltas against this bundle
      metrics: [attack_rate, Infected_total.peak, Infected_total.peak_date]
      over:                              # several entries form a cartesian product
        - path: interventions.0.start_date
          name: start_date               # default: the last path segment
          range: {start: "2026-12-01", stop: "2027-04-01", step: 7d}
        - path: interventions.0.reduction
          values: [0.5, 0.9]
      derive:                            # paths set from the swept values
        interventions.0.end_date: "{{ start_date + 21d }}"

Every point is validated before anything runs, gets its own seed derived from
the sweep's seed and the point key, and is written as a full bundle under
``points/<key>.epx/`` whose ``config.yaml`` reproduces it on its own.
"""

import copy
import datetime
import hashlib
import itertools
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

SWEEP_SCHEMA_VERSION = 1
_DURATION = re.compile(r"^\s*(\d+)\s*([dw])\s*$")
_TEMPLATE = re.compile(r"^\s*\{\{(.*)\}\}\s*$", re.S)
_TERM = re.compile(r"\s*([+-])\s*(\d+\s*[dw])\s*")
_NAME = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)")


class SweepError(ValueError):
    """A malformed sweep block, carrying the point it concerns (if any)."""

    def __init__(self, message: str, point: Optional[str] = None, errors=None):
        super().__init__(message)
        self.point = point
        self.errors = errors or [message]


# ---------------------------------------------------------------------------
# Values, paths and templates
# ---------------------------------------------------------------------------


def parse_duration(text: Any) -> Optional[int]:
    """Days in a duration such as ``7d`` or ``2w``; None if it is not one."""
    if not isinstance(text, str):
        return None
    match = _DURATION.match(text)
    if not match:
        return None
    n, unit = int(match.group(1)), match.group(2)
    return n * (7 if unit == "w" else 1)


def _as_date(value: Any) -> Optional[pd.Timestamp]:
    if isinstance(value, (datetime.date, datetime.datetime)):
        return pd.Timestamp(value)
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()):
        try:
            return pd.Timestamp(value)
        except ValueError:
            return None
    return None


def _date_str(stamp: pd.Timestamp) -> str:
    return stamp.strftime("%Y-%m-%d")


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def expand_range(spec: Dict[str, Any]) -> List[Any]:
    """Values of ``{start, stop, step}``, stop included.

    Dates step by a duration (``7d``, ``2w``); numbers by a positive number.
    """
    if not isinstance(spec, dict) or not {"start", "stop", "step"} <= set(spec):
        raise SweepError("range needs start, stop and step")
    start, stop, step = spec["start"], spec["stop"], spec["step"]
    start_date, stop_date = _as_date(start), _as_date(stop)
    if start_date is not None or stop_date is not None:
        days = parse_duration(step)
        if start_date is None or stop_date is None or days is None or days <= 0:
            raise SweepError(
                f"a date range needs dates for start and stop and a duration "
                f"step such as 7d; got {spec!r}"
            )
        values = []
        current = start_date
        while current <= stop_date:
            values.append(_date_str(current))
            current += pd.Timedelta(days=days)
        return values
    if not (_number(start) and _number(stop) and _number(step)) or step <= 0:
        raise SweepError(
            f"a numeric range needs numbers and a positive step; got {spec!r}"
        )
    decimals = max(
        (len(str(x).split(".")[1]) if "." in str(x) else 0) for x in (start, step)
    )
    values = []
    i = 0
    while True:
        value = round(start + i * step, decimals + 6)
        if value > stop + abs(step) * 1e-9:
            break
        values.append(
            int(value) if isinstance(start, int) and isinstance(step, int) else value
        )
        i += 1
    return values


def _parse_path(path: str) -> List[Any]:
    """``"interventions.0.start_date"`` or ``"interventions[0].start_date"``."""
    if not isinstance(path, str) or not path.strip():
        raise SweepError(f"a sweep path must be a dotted path, got {path!r}")
    parts: List[Any] = []
    for part in path.replace("[", ".").replace("]", "").split("."):
        if not part:
            raise SweepError(f"{path!r} is not a dotted path")
        parts.append(int(part) if part.isdigit() else part)
    return parts


def set_path(config: Dict[str, Any], path: str, value: Any) -> None:
    """Set a dotted path (``interventions.0.start_date``) in a config.

    Missing mapping keys are created; list indices must already exist.
    """
    parts = _parse_path(path)
    node: Any = config
    for i, part in enumerate(parts[:-1]):
        nxt = parts[i + 1]
        if isinstance(part, int):
            if not isinstance(node, list) or part >= len(node):
                raise SweepError(f"{path}: no item {part} in this config")
            node = node[part]
        else:
            if not isinstance(node, dict):
                raise SweepError(f"{path}: '{part}' is not inside a mapping")
            if part not in node or node[part] is None:
                if isinstance(nxt, int):
                    raise SweepError(f"{path}: the config has no list '{part}'")
                node[part] = {}
            node = node[part]
    last = parts[-1]
    if isinstance(last, int):
        if not isinstance(node, list) or last >= len(node):
            raise SweepError(f"{path}: no item {last} in this config")
        node[last] = value
    else:
        if not isinstance(node, dict):
            raise SweepError(f"{path}: '{last}' is not inside a mapping")
        node[last] = value


def render_template(template: Any, variables: Dict[str, Any]) -> Any:
    """Evaluate ``"{{ name + 21d - 1w }}"``: a swept date plus/minus durations.

    Only a variable name followed by ``+`` or ``-`` durations is accepted.
    """
    if not isinstance(template, str):
        raise SweepError(f"a derive value must be a template string, got {template!r}")
    match = _TEMPLATE.match(template)
    if not match:
        raise SweepError(
            f"{template!r} is not a template of the form '{{{{ name + 7d }}}}'"
        )
    body = match.group(1)
    name_match = _NAME.match(body)
    if not name_match:
        raise SweepError(f"{template!r} must start with a sweep variable")
    name = name_match.group(1)
    if name not in variables:
        raise SweepError(
            f"{template!r} uses '{name}', which is not a sweep variable "
            f"({', '.join(variables)})"
        )
    rest = body[name_match.end() :]
    days = 0
    position = 0
    while position < len(rest.rstrip()):
        term = _TERM.match(rest, position)
        if not term:
            raise SweepError(
                f"{template!r}: only '+' and '-' with durations such as 7d are "
                f"allowed, near {rest[position:].strip()!r}"
            )
        sign = 1 if term.group(1) == "+" else -1
        days += sign * parse_duration(term.group(2))
        position = term.end()
    value = variables[name]
    if days == 0:
        return value
    stamp = _as_date(value)
    if stamp is None:
        raise SweepError(
            f"{template!r}: '{name}' is not a date, so durations cannot apply"
        )
    return _date_str(stamp + pd.Timedelta(days=days))


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


def validate_block(sweep: Any) -> List[str]:
    """Problems with the shape of a sweep block (empty if none)."""
    if not isinstance(sweep, dict):
        return [
            "sweep must be a mapping with 'over' (and optionally name, metrics, derive, reference)"
        ]
    errors = []
    over = sweep.get("over")
    if not isinstance(over, list) or not over:
        errors.append(
            "sweep.over must be a non-empty list of {path, values} or {path, range}"
        )
    else:
        for i, entry in enumerate(over):
            if not isinstance(entry, dict) or "path" not in entry:
                errors.append(f"sweep.over[{i}] needs a 'path'")
                continue
            has_values, has_range = "values" in entry, "range" in entry
            if has_values == has_range:
                errors.append(
                    f"sweep.over[{i}] needs exactly one of 'values' or 'range'"
                )
            elif has_values and (
                not isinstance(entry["values"], list) or not entry["values"]
            ):
                errors.append(f"sweep.over[{i}].values must be a non-empty list")
    for key in ("name", "reference"):
        if key in sweep and not isinstance(sweep[key], str):
            errors.append(f"sweep.{key} must be text")
    metrics = sweep.get("metrics")
    if metrics is not None and (
        not isinstance(metrics, list) or not all(isinstance(m, str) for m in metrics)
    ):
        errors.append("sweep.metrics must be a list of metric names")
    derive = sweep.get("derive")
    if derive is not None and not isinstance(derive, dict):
        errors.append("sweep.derive must map config paths to templates")
    return errors


def variable_name(entry: Dict[str, Any]) -> str:
    if entry.get("name"):
        return str(entry["name"])
    parts = [p for p in _parse_path(entry["path"]) if not isinstance(p, int)]
    return str(parts[-1])


def _key_part(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(value))


def expand_grid(sweep: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every point of the sweep: ``{"key", "variables", "paths"}``.

    ``variables`` maps sweep variable names to values, ``paths`` maps config
    paths to values (swept and derived).
    """
    entries = sweep["over"]
    names = [variable_name(e) for e in entries]
    if len(set(names)) != len(names):
        raise SweepError(f"sweep variables must have distinct names: {names}")
    axes = []
    for entry in entries:
        values = (
            expand_range(entry["range"]) if "range" in entry else list(entry["values"])
        )
        if not values:
            raise SweepError(f"sweep over {entry['path']} has no values")
        axes.append(values)

    points, seen = [], {}
    for combo in itertools.product(*axes):
        variables = dict(zip(names, combo))
        paths = {entry["path"]: value for entry, value in zip(entries, combo)}
        for path, template in (sweep.get("derive") or {}).items():
            paths[path] = render_template(template, variables)
        key = "__".join(f"{n}={_key_part(v)}" for n, v in variables.items())
        if key in seen:
            seen[key] += 1
            key = f"{key}__{seen[key]}"
        else:
            seen[key] = 0
        points.append({"key": key, "variables": variables, "paths": paths})
    return points


def point_seed(parent_seed: int, key: str) -> int:
    """A point's seed: the first 32 bits of sha256(parent seed, point key).

    32 bits, like every seed the CLI draws, so that it survives JSON readers
    that parse numbers as doubles.
    """
    digest = hashlib.sha256(f"{int(parent_seed)}:{key}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def point_config(
    config: Dict[str, Any], point: Dict[str, Any], parent_seed: int, sweep_name: str
) -> Dict[str, Any]:
    """The standalone config of one sweep point."""
    cfg = copy.deepcopy(config)
    cfg.pop("sweep", None)
    for path, value in point["paths"].items():
        set_path(cfg, path, value)
    cfg.setdefault("simulation", {})["seed"] = point_seed(parent_seed, point["key"])
    # A swept parameter is a choice of the sweep: declare it as such.
    swept = {
        _parse_path(path)[1]: value
        for path, value in point["paths"].items()
        if _parse_path(path)[0] == "parameters" and len(_parse_path(path)) == 2
    }
    if swept:
        entries = [
            a for a in cfg.get("assumptions") or [] if a.get("parameter") not in swept
        ]
        for name, value in swept.items():
            entries.append(
                {
                    "parameter": name,
                    "value": value,
                    "basis": "agent_choice",
                    "rationale": f"swept by {sweep_name}",
                }
            )
        cfg["assumptions"] = entries
    return cfg


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _lookup(tree: Any, parts: List[str]) -> Any:
    for part in parts:
        if not isinstance(tree, dict) or part not in tree:
            return None
        tree = tree[part]
    return tree


def metric_path(name: str) -> List[str]:
    """Where a metric lives in summary.json.

    ``attack_rate`` → ``attack_rate.0.5``; ``Infected_total.peak`` →
    ``variables.Infected_total.peak.0.5``; any other dotted path is used as is
    (a trailing quantile level may be given: ``Infected_total.peak.0.95``).
    """
    if name == "attack_rate":
        return ["attack_rate", "0.5"]
    match = re.fullmatch(r"(.+_total)\.(peak|peak_date|final)(?:\.(0\.\d+))?", name)
    if match:
        return ["variables", match.group(1), match.group(2), match.group(3) or "0.5"]
    return name.split(".")


def default_metrics(summary: Dict[str, Any]) -> List[str]:
    variables = list(summary.get("variables") or {})
    infected = next((v for v in variables if v.startswith(("Infected", "I"))), None)
    metrics = ["attack_rate"] if summary.get("attack_rate") else []
    if infected:
        metrics += [f"{infected}.peak", f"{infected}.peak_date"]
    return metrics


def metric_value(summary: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Reported (rounded) value, raw value and Monte Carlo error of a metric."""
    parts = metric_path(name)
    return {
        "value": _lookup(summary, parts),
        "raw": _lookup(summary.get("raw") or {}, parts),
        "se": _lookup(summary.get("mc_se") or {}, parts),
    }


def metric_delta(point: Dict[str, Any], reference: Dict[str, Any]) -> Any:
    """Point minus reference: numbers rounded to their combined error, dates in days."""
    from ..io.precision import round_to_se

    a, b = point["raw"], reference["raw"]
    if a is None or b is None:
        return None
    if _as_date(a) is not None and _as_date(b) is not None:
        return int((_as_date(a) - _as_date(b)).days)
    if not (_number(a) and _number(b)):
        return None
    se = None
    if _number(point["se"]) and _number(reference["se"]):
        se = (point["se"] ** 2 + reference["se"] ** 2) ** 0.5
    return round_to_se(a - b, se)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _run_point(job: Tuple) -> str:
    from .execute import run_to_bundle

    key, cfg, output, config_dir, config_path, extra = job
    run_to_bundle(
        cfg,
        output,
        config_dir=config_dir,
        config_path=config_path,
        command="sweep",
        extra_provenance=extra,
    )
    return key


def plan(config: Dict[str, Any], parent_seed: int) -> List[Dict[str, Any]]:
    """Expand and validate every point; raise SweepError at the first invalid one."""
    from .config import validate_config

    sweep = config.get("sweep")
    problems = validate_block(sweep)
    if problems:
        raise SweepError("Invalid sweep block.", errors=problems)
    name = sweep.get("name") or "sweep"
    planned = []
    for point in expand_grid(sweep):
        try:
            cfg = point_config(config, point, parent_seed, name)
        except SweepError as e:
            raise SweepError(str(e), point=point["key"], errors=e.errors)
        report = validate_config(cfg)
        if not report["valid"]:
            raise SweepError(
                f"Sweep point {point['key']} is not a valid config.",
                point=point["key"],
                errors=report["errors"],
            )
        planned.append({**point, "config": cfg, "warnings": report["warnings"]})
    return planned


def run_sweep(
    config: Dict[str, Any],
    output,
    config_dir: Optional[Path] = None,
    config_path: Optional[str] = None,
    workers: int = 1,
) -> Dict[str, Any]:
    """Validate, run and aggregate a sweep. Returns the parent summary.

    Nothing runs unless every point is valid.
    """
    from ..io.json_utils import NumpySafeEncoder
    from ..io.manifest import MANIFEST_SCHEMA_VERSION, _get_version
    from ..io.provenance import provenance_block, seal_bundle
    from ..io.summary import load_summary
    from .config import resolve_seed

    output = Path(output)
    config = copy.deepcopy(config)
    parent_seed = resolve_seed(config.get("simulation") or {})
    config.setdefault("simulation", {})["seed"] = parent_seed
    sweep = config["sweep"]
    name = sweep.get("name") or output.stem
    planned = plan(config, parent_seed)

    points_dir = output / "points"
    points_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for point in planned:
        extra = {
            "sweep": name,
            "point_key": point["key"],
            "parent_seed": parent_seed,
            "parent_bundle": str(output.resolve()),
        }
        jobs.append(
            (
                point["key"],
                point["config"],
                str(points_dir / f"{point['key']}.epx"),
                config_dir,
                config_path,
                extra,
            )
        )
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_run_point, jobs))
    else:
        for job in jobs:
            _run_point(job)

    # Aggregate the points' summaries.
    summaries = {
        p["key"]: load_summary(str(points_dir / f"{p['key']}.epx")) for p in planned
    }
    metrics = sweep.get("metrics") or default_metrics(next(iter(summaries.values())))
    reference = None
    if sweep.get("reference"):
        ref_path = Path(sweep["reference"])
        if not ref_path.is_absolute() and config_dir is not None:
            ref_path = Path(config_dir) / ref_path
        ref_summary = load_summary(str(ref_path))
        reference = {
            "bundle": str(ref_path.resolve()),
            "metrics": {m: metric_value(ref_summary, m) for m in metrics},
        }
    rows = []
    for point in planned:
        values = {m: metric_value(summaries[point["key"]], m) for m in metrics}
        row = {
            "key": point["key"],
            "variables": point["variables"],
            "bundle": f"points/{point['key']}.epx",
            "seed": point["config"]["simulation"]["seed"],
            "metrics": {m: v["value"] for m, v in values.items()},
        }
        if reference:
            row["deltas"] = {
                m: metric_delta(values[m], reference["metrics"][m]) for m in metrics
            }
        rows.append(row)
    summary = {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "type": "sweep",
        "name": name,
        "n_points": len(rows),
        "variables": [variable_name(e) for e in sweep["over"]],
        "metrics": metrics,
        "points": rows,
    }
    if reference:
        summary["reference"] = {
            "bundle": reference["bundle"],
            "metrics": {m: v["value"] for m, v in reference["metrics"].items()},
        }
    with open(output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=NumpySafeEncoder)

    import yaml

    with open(output / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    import uuid
    from datetime import datetime as _dt

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "id": uuid.uuid4().hex,
        "type": "SweepResults",
        "epydemix_version": _get_version(),
        "created_at": _dt.utcnow().isoformat(timespec="seconds") + "Z",
        "sweep": {
            "name": name,
            "variables": summary["variables"],
            "metrics": metrics,
            "n_points": len(rows),
            "parent_seed": parent_seed,
        },
        "points": [
            {
                "key": r["key"],
                "bundle": r["bundle"],
                "variables": r["variables"],
                "seed": r["seed"],
            }
            for r in rows
        ],
        "files": {
            "summary": {
                "path": "summary.json",
                "description": "One row per point: swept values and metrics "
                "(with deltas against the reference, if any).",
                "schema_version": SWEEP_SCHEMA_VERSION,
            },
            "config": {"path": "config.yaml", "description": "The sweep config."},
        },
        "figures": {},
        "provenance": provenance_block("sweep", config_path, config),
    }
    (output / "figures").mkdir(exist_ok=True)
    seal_bundle(output, manifest)
    return summary
