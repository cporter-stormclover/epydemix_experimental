"""Basic reproduction number and attack rate, computed from a model's structure.

The next-generation matrix of a compartmental model with contact matrix ``C``
factorizes into a contact part and a disease part (rates do not depend on the
demographic group): ``R0 = rho(C) * rho(F V^-1)``, where ``F`` holds the rates
at which each infected compartment generates new infections and ``V`` the
transitions out of, and between, the infected compartments. ``rho`` is the
spectral radius. ``1 / rho(F V^-1)`` at unit transmission rate is the model's
effective removal rate ``gamma``, so ``transmission_rate = R0 * gamma / rho(C)``;
for SEIR, ``gamma`` is the recovery rate.

R0 here is for a fully susceptible population: only infections of the
susceptible compartment count (e.g. breakthrough infections of vaccinated
people, absent at the start, do not).
"""

import ast
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np

# Compartments of the predefined models (and their modules) that hold
# infected people. Custom models declare theirs (``role: infected``).
INFECTED_LIKE = frozenset({"Exposed", "Infected", "Infectious", "Asymptomatic"})
SUSCEPTIBLE_LIKE = frozenset({"Susceptible"})
# Transition kinds by role in the next-generation matrix (declarative custom
# models add the expression kinds).
MEDIATED_KINDS = ("mediated", "contact_expression")
SPONTANEOUS_KINDS = ("spontaneous", "expression")


def spectral_radius(matrix) -> float:
    """Largest absolute eigenvalue of a square matrix."""
    matrix = np.asarray(matrix, dtype=float)
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def combined_contact_matrix(population, layers: Optional[Sequence[str]] = None):
    """Sum of a population's contact matrices over the given layers (all by default)."""
    layers = list(layers) if layers else list(population.contact_matrices)
    missing = [layer for layer in layers if layer not in population.contact_matrices]
    if missing:
        raise ValueError(
            f"Population has no contact layer {missing}; layers: "
            f"{list(population.contact_matrices)}"
        )
    return sum(np.asarray(population.contact_matrices[layer]) for layer in layers)


def _scalar_parameters(parameters: Dict[str, Any]) -> Dict[str, float]:
    scalars = {}
    for name, value in parameters.items():
        array = np.asarray(value, dtype=float) if not isinstance(value, str) else None
        if array is None:
            continue
        if array.ndim == 0 or np.all(array == array.flat[0]):
            scalars[name] = float(array.flat[0])
    return scalars


def rate_parameters(expr: Any) -> Set[str]:
    """Names used in a rate (a parameter name or an expression)."""
    from .expressions import Expression

    if isinstance(expr, Expression):
        return set(expr.names)
    if not isinstance(expr, str):
        return set()
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return set()
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def _rate_value(expr: Any, parameters: Dict[str, float]) -> float:
    from .expressions import Expression

    if isinstance(expr, Expression):
        state = expr.names - set(parameters)
        if state:
            raise ValueError(
                f"rate {expr.source!r} depends on {sorted(state)}, which change "
                "during a run; R0 is defined for constant rates only"
            )
        return float(np.asarray(expr.evaluate(dict(parameters))).flat[0])
    if not isinstance(expr, str):
        return float(expr)
    if expr in parameters:
        return parameters[expr]
    from ..utils.utils import evaluate

    missing = rate_parameters(expr) - set(parameters)
    if missing:
        raise ValueError(
            f"rate {expr!r} needs {sorted(missing)}, which have no constant value"
        )
    return float(np.asarray(evaluate(expr=expr, env=dict(parameters))).flat[0])


def infected_compartments(model) -> List[str]:
    """Infected-type compartments of a model built from predefined pieces."""
    return [c for c in model.compartments if c in INFECTED_LIKE]


def infected_compartments_for(
    model_cfg: Dict[str, Any], compartments: Iterable[str]
) -> Optional[List[str]]:
    """Infected-type compartments of the model a config describes.

    Predefined models use the fixed names in :data:`INFECTED_LIKE`; custom
    models must tag theirs ``role: infected``. Returns None when the config
    does not say (a custom model without roles).
    """
    compartments = list(compartments)
    model_type = (model_cfg or {}).get("type", "custom")
    if model_type != "custom":
        return [c for c in compartments if c in INFECTED_LIKE]
    roles = compartment_roles(model_cfg)
    infected = [c for c in compartments if roles.get(c) == "infected"]
    return infected or None


def compartment_roles(model_cfg: Dict[str, Any]) -> Dict[str, str]:
    """``{compartment: role}`` declared by a custom model config."""
    roles = {}
    model_cfg = model_cfg or {}
    block = (
        model_cfg.get("custom")
        if isinstance(model_cfg.get("custom"), dict)
        else model_cfg
    )
    for entry in block.get("compartments") or []:
        if isinstance(entry, dict) and "name" in entry and entry.get("role"):
            roles[str(entry["name"])] = str(entry["role"])
    return roles


def disease_ngm(
    model,
    infected: Optional[Iterable[str]] = None,
    susceptible: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The disease part of the next-generation matrix, ``F V^-1``.

    Args:
        model: An EpiModel with ``mediated`` and ``spontaneous`` transitions.
        infected: Infected compartments (default: :data:`INFECTED_LIKE`).
        susceptible: The compartment the population starts in (default: the
            source of the first mediated transition into an infected one).
        parameters: Parameter values to use (default: the model's). Every
            rate involved must have a constant value.

    Returns:
        ``{"matrix", "compartments", "R0_per_contact"}``; the last is
        ``rho(F V^-1)``, the reproduction number for a contact matrix of
        spectral radius 1.
    """
    params = _scalar_parameters(model.parameters if parameters is None else parameters)
    infected = list(infected) if infected is not None else infected_compartments(model)
    if not infected:
        raise ValueError("the model has no infected compartments")
    index = {c: i for i, c in enumerate(infected)}
    n = len(infected)
    F = np.zeros((n, n))
    V = np.zeros((n, n))

    mediated_sources = [
        t.source
        for t in model.transitions_list
        if t.kind in MEDIATED_KINDS and t.target in index and t.source not in index
    ]
    if susceptible is None:
        if not mediated_sources:
            raise ValueError("the model has no transmission (mediated) transition")
        susceptible = mediated_sources[0]

    for t in model.transitions_list:
        if t.kind in MEDIATED_KINDS:
            rate, agent = t.params[0], t.params[1]
            if t.source == susceptible and t.target in index and agent in index:
                F[index[t.target], index[agent]] += _rate_value(rate, params)
        elif t.kind in SPONTANEOUS_KINDS and t.source in index:
            rate = _rate_value(
                t.params[0] if t.kind == "expression" else t.params, params
            )
            i = index[t.source]
            V[i, i] += rate
            if t.target in index:
                V[index[t.target], i] -= rate
    try:
        K = F @ np.linalg.inv(V)
    except np.linalg.LinAlgError:
        raise ValueError(
            "infected compartments have no way out (singular V); R0 is infinite"
        )
    return {
        "matrix": K,
        "compartments": infected,
        "R0_per_contact": spectral_radius(K),
        "susceptible": susceptible,
    }


def basic_reproduction_number(
    model,
    layers: Optional[Sequence[str]] = None,
    infected: Optional[Iterable[str]] = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> float:
    """R0 of a model in its population: ``rho(C) * rho(F V^-1)``."""
    rho = spectral_radius(combined_contact_matrix(model.population, layers))
    return rho * disease_ngm(model, infected, parameters=parameters)["R0_per_contact"]


def solve_rate_for_r0(
    model,
    R0: float,
    parameter: str = "transmission_rate",
    layers: Optional[Sequence[str]] = None,
    infected: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """The value of a transmission parameter that gives a target R0.

    When R0 is proportional to the parameter (it scales every infection rate,
    as ``transmission_rate`` does in the predefined models), ``value = R0 *
    gamma / rho``. When other routes also transmit (R0 grows with the
    parameter but does not vanish without it), the value is found by root
    finding and ``gamma`` is None.

    Returns:
        ``{"value", "rho", "gamma", "formula", "inputs"}``: ``rho`` the
        spectral radius of the contact matrix, ``gamma`` the effective removal
        rate, ``inputs`` the parameters the result depends on.
    """
    if not R0 > 0:
        raise ValueError(f"R0 must be positive, got {R0}")
    params = _scalar_parameters(model.parameters)
    infected = list(infected) if infected is not None else infected_compartments(model)

    def per_contact_at(value):
        params[parameter] = value
        return disease_ngm(model, infected, parameters=params)["R0_per_contact"]

    rho = spectral_radius(combined_contact_matrix(model.population, layers))
    at_one, at_two = per_contact_at(1.0), per_contact_at(2.0)
    if not at_two > at_one:
        raise ValueError(
            f"R0 does not grow with '{parameter}', so it cannot be derived from R0 "
            "(derive a transmission rate)"
        )
    at_zero = per_contact_at(0.0)
    ngm = disease_ngm(model, infected, parameters=params)
    if at_zero == 0 and np.isclose(at_two, 2 * at_one, rtol=1e-9):
        # R0 proportional to the parameter: the closed form.
        gamma = 1.0 / at_one
        value = R0 * gamma / rho
        formula = f"{parameter} = R0 * gamma / rho(C)"
    else:
        # Other transmission routes contribute too (e.g. nosocomial spread);
        # R0 still grows with the parameter, so solve for it.
        from scipy.optimize import brentq

        target = R0 / rho
        if at_zero >= target:
            raise ValueError(
                f"R0 is {at_zero * rho:.4g} even with {parameter} = 0, so no value "
                f"of {parameter} gives R0 = {R0}"
            )
        high = 1.0
        while per_contact_at(high) < target:
            high *= 2
            if high > 1e9:
                raise ValueError(f"no value of {parameter} reaches R0 = {R0}")
        value = brentq(lambda x: per_contact_at(x) - target, 0.0, high, xtol=1e-14)
        gamma = None
        formula = f"{parameter} solves rho(C) * rho(F V^-1) = R0"
    infected_set = set(infected)
    inputs = set()
    for t in model.transitions_list:
        if t.source in infected_set and t.kind in SPONTANEOUS_KINDS:
            inputs |= rate_parameters(
                t.params[0] if t.kind == "expression" else t.params
            )
        if (
            t.kind in MEDIATED_KINDS
            and t.target in infected_set
            and t.source == ngm["susceptible"]
        ):
            inputs |= rate_parameters(t.params[0])
    inputs = sorted((inputs & set(params)) - {parameter})
    return {
        "value": float(value),
        "rho": rho,
        "gamma": gamma,
        "formula": formula,
        "inputs": inputs,
    }


# ---------------------------------------------------------------------------
# Attack rate
# ---------------------------------------------------------------------------


def split_transition(column: str, compartments: Iterable[str]):
    """``(source, target)`` of a transition column such as ``S_to_E_total``."""
    name = column[: -len("_total")] if column.endswith("_total") else column
    known = set(compartments)
    parts = name.split("_to_")
    for i in range(1, len(parts)):
        source, target = "_to_".join(parts[:i]), "_to_".join(parts[i:])
        if source in known and target in known:
            return source, target
    return None


def infection_flow_columns(
    columns: Iterable[str], compartments: Iterable[str], infected: Iterable[str]
) -> List[str]:
    """``_total`` transition columns that move people into infection.

    Their target is an infected-type compartment and their source is not.
    """
    infected = set(infected)
    compartments = list(compartments)
    selected = []
    for col in columns:
        if not col.endswith("_total"):
            continue
        pair = split_transition(col, compartments)
        if pair and pair[1] in infected and pair[0] not in infected:
            selected.append(col)
    return selected
