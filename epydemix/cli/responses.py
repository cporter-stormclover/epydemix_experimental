"""Typed shapes of what the CLI commands print to stdout.

These dataclasses document the JSON each command prints. The agent contract
(AGENT.md) renders them, and the tests check that the commands' output
conforms to them, so the documentation cannot drift from the code. Fields
typed ``Optional`` may be absent.
"""

from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, get_args, get_origin


@dataclass
class ValidateResult:
    """``epydemix validate``."""

    valid: bool
    errors: List[str]
    warnings: List[str]
    n_points: Optional[int] = None  # sweep configs: points checked
    point: Optional[str] = None  # sweep configs: the first invalid point
    upgraded: Optional[Dict[str, Any]] = None  # --upgrade: {path, from, to}


@dataclass
class ManifestResult:
    """``epydemix run`` / ``calibrate`` / ``project``: the bundle's manifest."""

    schema_version: int
    id: str
    type: str  # SimulationResults | CalibrationResults
    files: Dict[str, Any]  # {stem: {path, description, columns?, ...}}
    provenance: Dict[str, Any]
    parameters_used: Dict[str, Any]
    epydemix_version: str
    created_at: str
    model: Optional[Dict[str, Any]] = (
        None  # {compartments, type, infected_compartments}
    )
    simulation: Optional[Dict[str, Any]] = None  # {n_simulations, start/end, seed}
    population: Optional[Dict[str, Any]] = None  # {demographic_groups}
    calibration: Optional[Dict[str, Any]] = None
    assumptions: Optional[List[Dict[str, Any]]] = None
    derived: Optional[Dict[str, Any]] = None
    figures: Optional[Dict[str, Any]] = None
    usage_hints: Optional[Dict[str, Any]] = None


@dataclass
class SummaryFile:
    """``<bundle>/summary.json``."""

    schema_version: int
    type: str
    quantiles: List[str]  # ["0.05", "0.5", "0.95"]
    precision: Dict[str, Any]
    raw: Dict[str, Any]  # the same values, unrounded
    mc_se: Dict[str, Any]  # their bootstrap standard errors
    bundle_id: Optional[str] = None
    n_simulations: Optional[int] = None
    variables: Optional[Dict[str, Any]] = None  # {X_total: {peak, peak_date, final}}
    attack_rate: Optional[Dict[str, Any]] = None  # {0.05, 0.5, 0.95, units}
    attack_rate_method: Optional[str] = None  # transition_flows | legacy
    attack_rate_flows: Optional[List[str]] = None
    posterior: Optional[Dict[str, Any]] = None  # {param: {mean, std, ci95, median}}
    n_particles: Optional[int] = None


@dataclass
class DeriveResult:
    """``epydemix derive``: ``derived`` is {parameter: {value, from, R0, rho,
    gamma, formula, inputs, contact_layers}}."""

    derived: Dict[str, Any]
    warnings: List[str]


@dataclass
class SweepResult:
    """``epydemix sweep`` (also ``<sweep>/summary.json``); ``points`` rows
    are {key, variables, bundle, seed, metrics, deltas?}."""

    bundle: str
    schema_version: int
    type: str  # "sweep"
    name: str
    n_points: int
    variables: List[str]
    metrics: List[str]
    points: List[Dict[str, Any]]
    reference: Optional[Dict[str, Any]] = None


@dataclass
class CompareResult:
    """``epydemix compare``: {_assumptions, <scenario>: {<metric>: {median,
    ci90, ...}}, _deltas_vs_<baseline>?}."""

    _assumptions: Dict[str, Any]  # {differing, same, undeclared_differences}


@dataclass
class InspectResult:
    """``epydemix inspect`` (keys depend on the bundle's variables and the
    command): ``quantiles`` {dates: [str], <variable>: {<q>: [number]}};
    ``peak`` {<variable>: {peak_date: {0.05, 0.50, 0.95, reported}, peak_value:
    {0.05, 0.50, 0.95}}}; ``summary`` {<variable>: {peak_date_median,
    peak_date_reported, peak_value, final_value, mean_across_time}};
    ``posterior`` {<parameter>: {mean, std, ci95, median}}; ``fit``
    {<variable>: {<q>: [number]}, observed: {<variable>: [number]}};
    ``manifest`` the manifest."""


@dataclass
class VerifyResult:
    """``epydemix verify`` (exit 1 when ``verified`` is false)."""

    bundle: str
    drift: List[str]
    verified: bool
    files: Optional[Dict[str, Any]] = None  # {modified, missing, added, unchanged}
    manifest: Optional[str] = None
    config: Optional[str] = None
    population_data: Optional[Dict[str, Any]] = None
    environment_differences: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None


@dataclass
class ReplayResult:
    """``epydemix replay`` (exit 1 when ``reproduced`` is false)."""

    bundle: str
    command: str
    max_rel_diff: float
    n_compared: int
    tol: float
    reproduced: bool
    seed: Optional[int] = None
    replayed_to: Optional[str] = None
    worst: Optional[str] = None


@dataclass
class ErrorResult:
    """Any command on failure, printed to stderr with exit status 1."""

    error: bool
    code: str
    message: str
    details: Optional[List[str]] = None
    point: Optional[str] = None


RESPONSES = [
    ValidateResult,
    ManifestResult,
    SummaryFile,
    DeriveResult,
    SweepResult,
    InspectResult,
    CompareResult,
    VerifyResult,
    ReplayResult,
    ErrorResult,
]


def _type_name(tp) -> str:
    origin = get_origin(tp)
    if origin is list or origin is List:
        return f"[{_type_name(get_args(tp)[0])}]"
    if origin is dict or origin is Dict:
        return "{...}"
    if origin is not None and type(None) in get_args(tp):  # Optional[X]
        return _type_name([a for a in get_args(tp) if a is not type(None)][0])
    return {bool: "bool", int: "int", float: "number", str: "str"}.get(tp, "any")


def _is_optional(tp) -> bool:
    return get_origin(tp) is not None and type(None) in get_args(tp)


def render(model) -> str:
    """One-line shape: ``{valid: bool, errors: [str], n_points?: int}``."""
    if not fields(model):
        return ""
    parts = [
        f"{f.name}{'?' if _is_optional(f.type) else ''}: {_type_name(f.type)}"
        for f in fields(model)
    ]
    return "{" + ", ".join(parts) + "}"


def conforms(data: Dict[str, Any], model) -> List[str]:
    """Problems with a printed JSON object against its model (empty if none)."""
    problems = []
    known = {f.name: f for f in fields(model)}
    for name, f in known.items():
        if name not in data and not _is_optional(f.type):
            problems.append(f"missing {name}")
    if model is not CompareResult:
        problems += [f"unexpected {k}" for k in data if k not in known]
    return problems
