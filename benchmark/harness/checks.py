"""Declarative grader checks.

``grader.yaml`` lists checks; each reads the agent's workspace (its output
bundles and its final message) and, where needed, the reference solution's
outputs. Check types:

``config_equals``  ``{path, expect, rtol}`` — a value in the config of every
    output bundle matching ``bundle`` (default ``outputs/**/*.epx``). ``path``
    is a dotted config path, ``r0`` (the R0 the config implies: derived, or
    computed from its transmission rate), or ``transition:S->E@I`` (the rate of
    the transition from S to E, mediated by I if ``@I`` is given, evaluated
    from the config's parameters, so the agent's parameter names do not
    matter). A list of paths means "the first one present".
``metric_within``  ``{bundle, metric, reference, rtol}`` — a summary.json
    metric (dotted path, e.g. ``variables.H_total.peak.0.5``) of every matching
    bundle, within ``rtol`` of a number or of the same metric of the reference
    output matching ``reference`` (a glob relative to the reference outputs).
``provenance_complete`` — every output bundle verifies (``epydemix verify
    --offline``) and records seed, config hash, code, environment and contract.
``assumptions_declared`` — validating each output bundle's config, with the
    task prompt's numbers as literals, gives no undeclared-value warning.
``ranking_preserved``  ``{metric, items, order}`` — the bundles matching each
    glob in ``items`` rank by ``metric`` in ``order``: ``ascending``,
    ``descending``, or ``reference`` (the order the reference outputs give).
``report_mentions``  ``{pattern}`` — a regular expression found in the
    agent's final message.
"""

import glob
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml


@dataclass
class CheckResult:
    name: str
    type: str
    passed: bool
    detail: str = ""
    values: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "passed": self.passed,
            "detail": self.detail,
            "values": self.values,
        }


@dataclass
class Context:
    """What a check can look at."""

    workspace: Path
    prompt: str
    final_message: str
    reference_outputs: Optional[Path] = None


DEFAULT_BUNDLES = "outputs/**/*.epx"


def find_bundles(root: Path, pattern: str) -> List[Path]:
    """Bundles (directories with a manifest.json) matching a glob under root."""
    found = sorted(Path(p) for p in glob.glob(str(root / pattern), recursive=True))
    return [p for p in found if (p / "manifest.json").exists()]


def _sim_bundles(
    root: Path, pattern: str, spec: Optional[Dict[str, Any]] = None
) -> List[Path]:
    """Matching result bundles (sweep parents excluded: their points match).

    ``spec`` may narrow the selection: ``bundle_type`` (``SimulationResults``
    or ``CalibrationResults``) and ``command`` (the CLI command that wrote the
    bundle, e.g. ``project``).
    """
    spec = spec or {}
    out = []
    for bundle in find_bundles(root, pattern):
        with open(bundle / "manifest.json") as f:
            manifest = json.load(f)
        kind = manifest.get("type")
        if kind == "SweepResults":
            continue
        if spec.get("bundle_type") and kind != spec["bundle_type"]:
            continue
        command = (manifest.get("provenance") or {}).get("command")
        if spec.get("command") and command != spec["command"]:
            continue
        out.append(bundle)
    return out


def _selected(ctx_root: Path, spec: Dict[str, Any]) -> List[Path]:
    return _sim_bundles(ctx_root, spec.get("bundle", DEFAULT_BUNDLES), spec)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _path_parts(dotted: str) -> List[str]:
    """Split a dotted path, keeping quantile levels such as ``0.5`` whole."""
    return re.findall(
        r"\d+\.\d+(?![^.])|[^.]+", dotted.replace("[", ".").replace("]", "")
    )


def _lookup(tree: Any, dotted: str) -> Any:
    node = tree
    for part in _path_parts(dotted):
        if isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        elif isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


def _transition_rate(config: Dict[str, Any], spec: str) -> Optional[float]:
    """Rate of ``S->E@I`` in a config, evaluated from its parameters."""
    from epydemix.utils.utils import evaluate

    match = re.fullmatch(r"\s*(\w+)\s*->\s*(\w+)\s*(?:@\s*(\w+))?\s*", spec)
    if not match:
        raise ValueError(f"bad transition spec {spec!r}; use transition:S->E@I")
    source, target, agent = match.groups()
    params = dict(config.get("parameters") or {})
    model = config.get("model") or {}
    custom = model.get("custom") if isinstance(model.get("custom"), dict) else None
    transitions = (custom or model).get("transitions") or []
    total, found = 0.0, False
    for t in transitions:
        src = t.get("source", t.get("from"))
        tgt = t.get("target", t.get("to"))
        if src != source or tgt != target:
            continue
        rate = t.get("rate", t.get("expression"))
        t_agent = None
        if "params" in t:
            p = t["params"]
            if isinstance(p, list):
                rate, t_agent = p[0], p[1]
            else:
                rate = p
        agents = t.get("agents") or ([t_agent] if t_agent else [])
        if agent is not None and agent not in agents:
            continue
        if isinstance(rate, str) and rate in params:
            rate = params[rate]
        if isinstance(rate, str) and custom is not None:
            from epydemix.model.expressions import compile_expression

            value = float(compile_expression(rate, set(params)).evaluate(params))
        elif isinstance(rate, str):
            value = float(evaluate(rate, params))
        else:
            value = float(rate)
        total += value
        found = True
    return total if found else None


def _r0(config: Dict[str, Any], config_dir: Path) -> Optional[float]:
    """The R0 a config implies (its derived value, or computed)."""
    derived = config.get("derived") or {}
    for spec in derived.values():
        if isinstance(spec, dict) and spec.get("from") == "R0":
            return float(spec["value"])
    from epydemix.cli.config import build_model_from_config
    from epydemix.model.reproduction import (
        basic_reproduction_number,
        infected_compartments_for,
    )

    model = build_model_from_config(config, config_dir=config_dir)
    infected = infected_compartments_for(config.get("model") or {}, model.compartments)
    layers = (config.get("population") or {}).get("contact_layers")
    return basic_reproduction_number(model, layers, infected)


def config_value(bundle: Path, path) -> Any:
    config = _load_yaml(bundle / "config.yaml")
    for candidate in path if isinstance(path, list) else [path]:
        if candidate == "r0":
            value = _r0(config, bundle)
        elif candidate.startswith("transition:"):
            value = _transition_rate(config, candidate[len("transition:") :])
        else:
            value = _lookup(config, candidate)
        if value is not None:
            return value
    return None


def _close(a: Any, b: Any, rtol: float) -> bool:
    if (
        isinstance(a, (int, float))
        and isinstance(b, (int, float))
        and not isinstance(a, bool)
    ):
        return math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=1e-12)
    return a == b


def check_config_equals(ctx: Context, spec: Dict[str, Any]) -> CheckResult:
    bundles = _selected(ctx.workspace, spec)
    if not bundles:
        return CheckResult("", "", False, "no output bundle matches")
    values, wrong = {}, []
    for bundle in bundles:
        value = config_value(bundle, spec["path"])
        if value is None and spec.get("skip_missing"):
            continue
        values[bundle.name] = value
        if not _close(value, spec["expect"], float(spec.get("rtol", 0.0))):
            wrong.append(bundle.name)
    if not values:
        return CheckResult("", "", False, f"no output bundle sets {spec['path']}")
    detail = f"{spec['path']} should be {spec['expect']}"
    if wrong:
        detail += f"; differs in {', '.join(wrong)}"
    return CheckResult("", "", not wrong, detail, values)


def summary_metric(bundle: Path, metric: str) -> Any:
    from epydemix.io import load_summary

    summary = load_summary(str(bundle))
    value = _lookup(summary.get("raw") or {}, metric)
    return value if value is not None else _lookup(summary, metric)


def check_metric_within(ctx: Context, spec: Dict[str, Any]) -> CheckResult:
    bundles = _selected(ctx.workspace, spec)
    if not bundles:
        return CheckResult("", "", False, "no output bundle matches")
    reference = spec["reference"]
    if isinstance(reference, str):
        refs = (
            _sim_bundles(ctx.reference_outputs, reference, spec)
            if ctx.reference_outputs
            else []
        )
        if len(refs) != 1:
            return CheckResult(
                "", "", False, f"reference {reference!r} matches {len(refs)} bundles"
            )
        target = summary_metric(refs[0], spec["metric"])
    else:
        target = reference
    rtol = float(spec.get("rtol", 0.1))
    values, wrong = {"reference": target}, []
    for bundle in bundles:
        value = summary_metric(bundle, spec["metric"])
        values[bundle.name] = value
        if value is None or not _close(value, target, rtol):
            wrong.append(bundle.name)
    detail = f"{spec['metric']} within {rtol:.0%} of {target}"
    if wrong:
        detail += f"; outside in {', '.join(wrong)}"
    return CheckResult("", "", not wrong, detail, values)


REQUIRED_PROVENANCE = (
    "seed",
    "config_sha256",
    "code",
    "environment",
    "contract_version",
    "bundle_files",
)


def check_provenance_complete(ctx: Context, spec: Dict[str, Any]) -> CheckResult:
    from epydemix.io.provenance import verify_bundle

    bundles = find_bundles(ctx.workspace, spec.get("bundle", DEFAULT_BUNDLES))
    if not bundles:
        return CheckResult("", "", False, "no output bundle matches")
    problems = []
    for bundle in bundles:
        with open(bundle / "manifest.json") as f:
            prov = json.load(f).get("provenance") or {}
        missing = [k for k in REQUIRED_PROVENANCE if k not in prov]
        if missing:
            problems.append(f"{bundle.name} lacks {', '.join(missing)}")
            continue
        report = verify_bundle(bundle, check_population=False)
        if not report["verified"]:
            problems.append(
                f"{bundle.name}: {', '.join(report['drift']) or report.get('reason')}"
            )
    return CheckResult(
        "", "", not problems, "; ".join(problems) or f"{len(bundles)} bundles verified"
    )


def check_assumptions_declared(ctx: Context, spec: Dict[str, Any]) -> CheckResult:
    from epydemix.cli.assumptions import prompt_literals
    from epydemix.cli.config import validate_config

    bundles = _selected(ctx.workspace, spec)
    if not bundles:
        return CheckResult("", "", False, "no output bundle matches")
    literals = prompt_literals(ctx.prompt)
    undeclared = {}
    for bundle in bundles:
        config = _load_yaml(bundle / "config.yaml")
        warnings = validate_config(config, prompt_literals=literals)["warnings"]
        found = [w for w in warnings if "has no declared basis" in w]
        if found:
            undeclared[bundle.name] = found
    detail = (
        "; ".join(f"{b}: {len(w)} undeclared" for b, w in undeclared.items())
        or "every chosen value is declared"
    )
    return CheckResult("", "", not undeclared, detail, {"undeclared": undeclared})


def _ranked(root: Path, items: List[str], metric: str):
    values = []
    for item in items:
        matches = _sim_bundles(root, item)
        if len(matches) != 1:
            raise LookupError(f"{item!r} matches {len(matches)} bundles")
        values.append(summary_metric(matches[0], metric))
    return values


def _order_of(values: List[Any]) -> List[int]:
    return sorted(range(len(values)), key=lambda i: values[i])


def check_ranking_preserved(ctx: Context, spec: Dict[str, Any]) -> CheckResult:
    items, metric, order = spec["items"], spec["metric"], spec.get("order", "reference")
    try:
        values = _ranked(ctx.workspace, items, metric)
    except LookupError as e:
        return CheckResult("", "", False, str(e))
    if order == "reference":
        try:
            expected = _order_of(_ranked(ctx.reference_outputs, items, metric))
        except (LookupError, TypeError) as e:
            return CheckResult("", "", False, f"reference: {e}")
    else:
        expected = list(range(len(items)))
        if order == "descending":
            expected = expected[::-1]
    got = _order_of(values)
    return CheckResult(
        "",
        "",
        got == expected,
        f"{metric} order {[items[i] for i in got]} (expected {[items[i] for i in expected]})",
        {"values": dict(zip(items, values))},
    )


def check_report_mentions(ctx: Context, spec: Dict[str, Any]) -> CheckResult:
    pattern = spec["pattern"]
    found = re.search(pattern, ctx.final_message or "", re.M) is not None
    return CheckResult(
        "", "", found, f"final message {'matches' if found else 'lacks'} {pattern!r}"
    )


CHECKS: Dict[str, Callable[[Context, Dict[str, Any]], CheckResult]] = {
    "config_equals": check_config_equals,
    "metric_within": check_metric_within,
    "provenance_complete": check_provenance_complete,
    "assumptions_declared": check_assumptions_declared,
    "ranking_preserved": check_ranking_preserved,
    "report_mentions": check_report_mentions,
}


def run_checks(ctx: Context, checks: List[Dict[str, Any]]) -> List[CheckResult]:
    """Run every check of a grader; a check that crashes fails with the error."""
    results = []
    for i, spec in enumerate(checks):
        kind = spec.get("type")
        name = spec.get("name") or f"{i}:{kind}"
        if kind not in CHECKS:
            results.append(
                CheckResult(name, str(kind), False, f"unknown check type {kind!r}")
            )
            continue
        try:
            result = CHECKS[kind](ctx, spec)
        except Exception as e:  # a broken workspace must not stop the grading
            result = CheckResult("", "", False, f"error: {e}")
        result.name, result.type = name, kind
        results.append(result)
    return results
