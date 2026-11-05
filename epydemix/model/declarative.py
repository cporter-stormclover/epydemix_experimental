"""Declarative custom models: ``model.custom`` in a config.

::

    model:
      type: custom
      custom:
        compartments: [S, {name: E, role: infected}, {name: I, role: infected},
                       {name: H, role: infected}, R, D]
        parameters:                      # specs; values go in `parameters:`
          beta: {kind: rate, range: [0, 2], unit: 1/days, precision: 2}
        transitions:
          - {from: S, to: E, kind: contact, rate: beta, agents: [I]}
          - {from: I, to: H, expression: "gamma * p_hosp"}
          - {from: H, to: D, expression: "delta * min(1, H / (capacity * pop))"}

Transition kinds: ``spontaneous`` (the default: leave at a rate), ``contact``
(infection through the contact matrix by each compartment in ``agents``) and
``scheduled`` (daily doses, as in the original grammar). A rate is a
parameter (``rate``) or a whitelisted expression (``expression``; see
:mod:`epydemix.model.expressions`) of parameters, compartment sizes and
``pop``. ``role: infected`` marks the compartments the attack rate and R0
count.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..parameters.spec import PARAMETER_KINDS, ParameterSpec
from .expressions import Expression, ExpressionError, compile_expression

ROLES = ("susceptible", "infected", "recovered", "dead", "vaccinated", "other")
KINDS = ("spontaneous", "contact", "scheduled")
EXPRESSION_KIND = "expression"
CONTACT_EXPRESSION_KIND = "contact_expression"


def compartments(custom: Dict[str, Any]) -> List[Tuple[str, Optional[str]]]:
    """``[(name, role)]`` of a ``model.custom`` block (malformed entries skipped)."""
    out = []
    for entry in custom.get("compartments") or []:
        if isinstance(entry, str):
            out.append((entry, None))
        elif isinstance(entry, dict) and isinstance(entry.get("name"), str):
            out.append((entry["name"], entry.get("role")))
    return out


def compartment_names(custom: Dict[str, Any]) -> List[str]:
    return [name for name, _ in compartments(custom)]


def inline_specs(custom: Dict[str, Any]) -> List[ParameterSpec]:
    """Parameter specs declared inline (for the model's registry)."""
    specs = []
    for name, spec in (custom.get("parameters") or {}).items():
        if not isinstance(spec, dict):
            continue
        rng = spec.get("range") or [None, None]
        if not isinstance(rng, list):
            rng = [None, None]
        try:
            specs.append(
                ParameterSpec(
                    name=name,
                    description=spec.get("description", ""),
                    kind=spec.get("kind", "rate"),
                    min=rng[0] if len(rng) > 0 else None,
                    max=rng[1] if len(rng) > 1 else None,
                    units=spec.get("unit"),
                    precision=spec.get("precision"),
                    default=spec.get("default"),
                    tags=["custom"],
                )
            )
        except (ValueError, TypeError):
            continue  # reported by validate()
    return specs


def _allowed_names(custom, parameter_names):
    return set(parameter_names) | set(compartment_names(custom)) | {"pop"}


def validate(
    custom: Any,
    parameter_names: set,
    initial_conditions: Optional[Dict[str, Any]],
    errors: list,
) -> None:
    """Check a ``model.custom`` block.

    Args:
        custom: The block.
        parameter_names: Parameters with a value (``parameters``, ``derived``
            or calibration priors).
        initial_conditions: The config's initial conditions, to find the
            compartments a run starts with.
        errors: Problems are appended here.
    """
    if not isinstance(custom, dict):
        errors.append(
            "model.custom must be a mapping with compartments and transitions"
        )
        return
    entries = custom.get("compartments")
    if not isinstance(entries, list) or not entries:
        errors.append("model.custom.compartments must be a non-empty list")
        return
    names = []
    for i, entry in enumerate(entries):
        if isinstance(entry, str):
            names.append(entry)
            continue
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            errors.append(
                f"model.custom.compartments[{i}] must be a name or {{name, role}}"
            )
            continue
        names.append(entry["name"])
        role = entry.get("role")
        if role is not None and role not in ROLES:
            errors.append(
                f"model.custom.compartments[{i}].role {role!r} must be one of {', '.join(ROLES)}"
            )
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        errors.append(f"model.custom.compartments repeats {', '.join(duplicates)}")
    if "pop" in names:
        errors.append(
            "'pop' is reserved (the group sizes) and cannot name a compartment"
        )

    specs = custom.get("parameters") or {}
    if not isinstance(specs, dict):
        errors.append(
            "model.custom.parameters must map names to {kind, range, unit, precision}"
        )
        specs = {}
    for name, spec in specs.items():
        label = f"model.custom.parameters.{name}"
        if name in names or name == "pop":
            errors.append(f"{label}: '{name}' is also a compartment (or reserved)")
        if not isinstance(spec, dict):
            errors.append(f"{label} must be a mapping with kind, range, unit")
            continue
        if spec.get("kind", "rate") not in PARAMETER_KINDS:
            errors.append(f"{label}.kind must be one of {', '.join(PARAMETER_KINDS)}")
        rng = spec.get("range")
        if rng is not None and not (
            isinstance(rng, list)
            and len(rng) == 2
            and all(
                isinstance(x, (int, float)) and not isinstance(x, bool) for x in rng
            )
            and rng[0] <= rng[1]
        ):
            errors.append(f"{label}.range must be [min, max]")
        precision = spec.get("precision")
        if precision is not None and (not isinstance(precision, int) or precision < 1):
            errors.append(f"{label}.precision must be a positive integer")
        if name not in parameter_names:
            errors.append(f"{label} is declared but has no value in parameters")

    # Rates may use only parameters declared (with their specs) above.
    known_params = set(specs)
    allowed = _allowed_names(custom, known_params)
    transitions = custom.get("transitions")
    if not isinstance(transitions, list) or not transitions:
        errors.append("model.custom.transitions must be a non-empty list")
        return
    edges = []
    for i, tr in enumerate(transitions):
        label = f"model.custom.transitions[{i}]"
        if not isinstance(tr, dict):
            errors.append(f"{label} must be a mapping")
            continue
        for end in ("from", "to"):
            if tr.get(end) not in names:
                errors.append(
                    f"{label}.{end} {tr.get(end)!r} is not a declared compartment. "
                    f"Compartments: {', '.join(names)}"
                )
        if tr.get("from") == tr.get("to") and tr.get("from") is not None:
            errors.append(f"{label}: from and to are the same compartment")
        kind = tr.get("kind", "spontaneous")
        if kind not in KINDS:
            errors.append(f"{label}.kind {kind!r} must be one of {', '.join(KINDS)}")
            continue
        if kind == "scheduled":
            if "schedule" not in tr:
                errors.append(f"{label}: kind scheduled needs a schedule")
            for c in tr.get("eligible") or []:
                if c not in names:
                    errors.append(
                        f"{label}.eligible {c!r} is not a declared compartment"
                    )
        else:
            has_rate, has_expr = "rate" in tr, "expression" in tr
            if has_rate == has_expr:
                errors.append(f"{label} needs exactly one of rate or expression")
            elif has_rate:
                rate = tr["rate"]
                if (
                    not (isinstance(rate, (int, float)) and not isinstance(rate, bool))
                    and rate not in known_params
                ):
                    errors.append(
                        f"{label}.rate {rate!r} is not a declared parameter. "
                        f"Parameters: {', '.join(sorted(known_params)) or 'none'}"
                    )
            else:
                try:
                    compile_expression(tr["expression"], allowed)
                except ExpressionError as e:
                    errors.append(f"{label}.expression: {e}")
            if kind == "contact":
                agents = tr.get("agents")
                if not isinstance(agents, list) or not agents:
                    errors.append(
                        f"{label}: a contact transition needs agents: [compartments]"
                    )
                else:
                    for a in agents:
                        if a not in names:
                            errors.append(
                                f"{label}.agents {a!r} is not a declared compartment"
                            )
            elif "agents" in tr:
                errors.append(f"{label}: agents only apply to kind: contact")
        if tr.get("from") in names and tr.get("to") in names:
            edges.append((tr["from"], tr["to"]))

    for name in unreachable(names, edges, initial_conditions):
        errors.append(
            f"model.custom: compartment '{name}' is unreachable: no transition leads "
            "to it and it starts empty"
        )


def _starts_populated(names, initial_conditions) -> set:
    if not isinstance(initial_conditions, dict):
        return set()
    populated = set()
    for name, value in initial_conditions.items():
        values = list(value.values()) if isinstance(value, dict) else [value]
        if any(
            isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0
            for v in values
        ):
            populated.add(name)
    return populated & set(names)


def unreachable(names, edges, initial_conditions=None) -> List[str]:
    """Compartments no run can populate.

    A compartment is reachable when it starts with people or a transition
    leads to it from a reachable one. Without initial conditions, the
    compartments without incoming transitions are taken as the start.
    """
    start = _starts_populated(names, initial_conditions)
    if not start:
        targets = {t for _, t in edges}
        start = {n for n in names if n not in targets}
    reached, frontier = set(start), list(start)
    while frontier:
        node = frontier.pop()
        for s, t in edges:
            if s == node and t not in reached:
                reached.add(t)
                frontier.append(t)
    return [n for n in names if n not in reached]


def graph(custom: Dict[str, Any]) -> Dict[str, Any]:
    """The compartment graph: nodes with roles, edges with kinds and rates."""
    edges = []
    for tr in custom.get("transitions") or []:
        if not isinstance(tr, dict):
            continue
        edge = {
            "from": tr.get("from"),
            "to": tr.get("to"),
            "kind": tr.get("kind", "spontaneous"),
        }
        for key in ("rate", "expression", "agents", "schedule"):
            if key in tr:
                edge[key] = tr[key]
        edges.append(edge)
    lines = ["graph LR"]
    for e in edges:
        label = e.get("rate") or e.get("expression") or e["kind"]
        via = f" via {', '.join(e['agents'])}" if e.get("agents") else ""
        lines.append(f'  {e["from"]} -->|"{label}{via}"| {e["to"]}')
    return {
        "compartments": [{"name": n, "role": r} for n, r in compartments(custom)],
        "edges": edges,
        "mermaid": "\n".join(lines),
    }


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


def _environment(expr: Expression, data: Dict[str, Any]) -> Dict[str, Any]:
    t = data["t"]
    env = {}
    for name in expr.names:
        if name in data["parameters"]:
            env[name] = data["parameters"][name][t]
        elif name in data["comp_indices"]:
            env[name] = data["pop"][data["comp_indices"][name]]
        elif name == "pop":
            env[name] = data["pop_sizes"]
    return env


def compute_expression_rate(params, data):
    """Per-group rate of a transition written as a whitelisted expression."""
    expr = params[0]
    value = np.asarray(expr.evaluate(_environment(expr, data)), dtype=float)
    return np.nan_to_num(np.broadcast_to(value, np.shape(data["pop_sizes"])), nan=0.0)


def compute_contact_expression_rate(params, data):
    """Contact-mediated rate whose per-contact rate is an expression."""
    rate = compute_expression_rate((params[0],), data)
    agent_idx = data["comp_indices"][params[1]]
    interaction = np.sum(
        data["contact_matrix"]["overall"] * data["pop"][agent_idx] / data["pop_sizes"],
        axis=1,
    )
    return rate * interaction


def add_transitions(
    model, custom: Dict[str, Any], parameter_names, schedule_loader=None
) -> None:
    """Add a ``model.custom`` block's transitions to an EpiModel.

    ``rate: <parameter>`` uses the built-in kinds; expressions use the
    ``expression`` and ``contact_expression`` kinds registered here; a
    ``contact`` transition becomes one mediated transition per agent.
    ``schedule_loader(spec)`` turns a schedule spec into a dose array.
    """
    if EXPRESSION_KIND not in model.transition_functions:
        model.register_transition_kind(EXPRESSION_KIND, compute_expression_rate)
        model.register_transition_kind(
            CONTACT_EXPRESSION_KIND, compute_contact_expression_rate
        )
    allowed = _allowed_names(custom, parameter_names)
    for tr in custom["transitions"]:
        kind = tr.get("kind", "spontaneous")
        source, target = tr["from"], tr["to"]
        if kind == "scheduled":
            doses = schedule_loader(tr["schedule"])
            eligible = tr.get("eligible")
            model.add_transition(
                source,
                target,
                kind="scheduled",
                params=(doses, eligible) if eligible else (doses,),
            )
            continue
        rate = (
            tr["rate"]
            if "rate" in tr
            else compile_expression(tr["expression"], allowed)
        )
        is_expr = isinstance(rate, Expression)
        if kind == "contact":
            for agent in tr["agents"]:
                model.add_transition(
                    source,
                    target,
                    kind=CONTACT_EXPRESSION_KIND if is_expr else "mediated",
                    params=(rate, agent),
                )
        else:
            model.add_transition(
                source,
                target,
                kind=EXPRESSION_KIND if is_expr else "spontaneous",
                params=(rate,) if is_expr else rate,
            )
