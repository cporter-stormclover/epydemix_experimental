"""The ``assumptions`` block: which parameter values were chosen, and why.

An agent (or a person) turning a question into a config reads some values
from the prompt and chooses others. ``assumptions`` declares the basis of each
value so that the choice travels with the bundle::

    assumptions:
      - parameter: vaccination_rate
        value: 0.0015
        basis: agent_choice          # prompt_literal | literature_default |
                                     # agent_choice | derived
        rationale: "'low' band of the registry for a slow campaign"

Validation warns (never errors) about a parameter set to a value other than
its registry default without a declared basis.
"""

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Set

BASES = ("prompt_literal", "literature_default", "agent_choice", "derived")
_REQUIRED_FIELDS = ("parameter", "value", "basis", "rationale")

_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?|\.\d+)(?:[eE]([+-]?\d+))?(\s*%)?")


def prompt_literals(text: str) -> Set[float]:
    """The numbers written in a prompt, as the values they could stand for.

    ``80%`` yields both 80 and 0.8; ``2.5e-3`` yields 0.0025.
    """
    values: Set[float] = set()
    for mantissa, exponent, percent in _NUMBER.findall(text):
        number = float(mantissa) * (10 ** int(exponent) if exponent else 1)
        values.add(number)
        if percent:
            values.add(number / 100)
    return values


def _is_literal(value: Any, literals: Iterable[float]) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    return any(
        math.isclose(value, lit, rel_tol=1e-9, abs_tol=1e-12) for lit in literals
    )


def declared(config: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Assumption entries by parameter name (malformed entries skipped)."""
    entries: Dict[str, List[Dict[str, Any]]] = {}
    for entry in config.get("assumptions") or []:
        if isinstance(entry, dict) and isinstance(entry.get("parameter"), str):
            entries.setdefault(entry["parameter"], []).append(entry)
    return entries


def validate_block(config: Dict[str, Any], errors: list) -> None:
    """Check the shape of the ``assumptions`` block."""
    block = config.get("assumptions")
    if block is None:
        return
    if not isinstance(block, list):
        errors.append(
            "assumptions must be a list of {parameter, value, basis, rationale}"
        )
        return
    for i, entry in enumerate(block):
        if not isinstance(entry, dict):
            errors.append(f"assumptions[{i}] must be a mapping")
            continue
        missing = [k for k in _REQUIRED_FIELDS if k not in entry]
        if missing:
            errors.append(f"assumptions[{i}] is missing {', '.join(missing)}")
        basis = entry.get("basis")
        if basis is not None and basis not in BASES:
            errors.append(
                f"assumptions[{i}].basis {basis!r} must be one of {', '.join(BASES)}"
            )
        rationale = entry.get("rationale")
        if "rationale" in entry and (
            not isinstance(rationale, str) or not rationale.strip()
        ):
            errors.append(f"assumptions[{i}].rationale must be a non-empty text")


def check_declared(
    uses: list,
    config: Dict[str, Any],
    registry=None,
    literals: Optional[Iterable[float]] = None,
    no_value: Any = None,
) -> List[str]:
    """Warnings for chosen parameter values that no assumption declares.

    Args:
        uses: ``(section, name, value, value_label)`` tuples, as built by the
            config validator.
        config: The config.
        registry: The model's parameter registry, for defaults and bands.
        literals: Numbers from the prompt; a value equal to one of them is
            read, not chosen.
        no_value: Marker of a use that carries no value (a prior).
    """
    entries = declared(config)
    literals = list(literals or [])
    warnings: List[str] = []
    values_by_name: Dict[str, list] = {}
    for section, name, value, label in uses:
        if value is no_value or not (
            section == "parameters" or section.startswith("overrides")
        ):
            continue
        values_by_name.setdefault(name, []).append(value)
        if name in entries:
            continue
        spec = registry.get(name) if registry is not None and name in registry else None
        if spec is not None and spec.default is not None and value == spec.default:
            continue
        if _is_literal(value, literals):
            continue
        hint = ""
        if (
            spec is not None
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            band = spec.band_of(value)
            if band:
                hint = f" (inside the '{band}' band of {name})"
        warnings.append(
            f"{label} = {value!r} has no declared basis{hint}; add an "
            f"assumptions entry for '{name}' (basis: {', '.join(BASES)})"
        )

    for name, items in entries.items():
        if name not in values_by_name:
            warnings.append(
                f"assumptions: '{name}' is not a parameter this config sets"
            )
            continue
        for entry in items:
            if "value" in entry and not any(
                _same(entry["value"], v) for v in values_by_name[name]
            ):
                shown = ", ".join(repr(v) for v in values_by_name[name])
                warnings.append(
                    f"assumptions: '{name}' is declared as {entry['value']!r} but "
                    f"the config sets {shown}"
                )
    return warnings


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def merge(base: list, overlay: list) -> list:
    """Merge two assumption lists by parameter; the overlay's entries win.

    Lists normally replace each other in config inheritance, but assumptions
    are keyed by parameter, so an overlay that declares one new value keeps
    the base config's declarations for the rest.
    """
    merged: Dict[Any, Any] = {}
    order: List[Any] = []
    for entry in list(base) + list(overlay):
        key = entry.get("parameter") if isinstance(entry, dict) else id(entry)
        if key not in merged:
            order.append(key)
        merged[key] = entry
    return [merged[k] for k in order]


def compare(manifests: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Differences between the declared assumptions of several bundles.

    Returns ``{"differing": [...], "same": [...], "undeclared_differences":
    [...]}``. Each differing entry gives, per scenario, the declared value and
    basis (``None`` where a bundle declares nothing for that parameter).
    ``undeclared_differences`` lists parameters whose values differ between
    bundles although no bundle declares them.
    """
    by_scenario = {
        name: {
            e["parameter"]: e for e in m.get("assumptions") or [] if isinstance(e, dict)
        }
        for name, m in manifests.items()
    }
    parameters = sorted({p for decl in by_scenario.values() for p in decl})
    differing, same = [], []
    for param in parameters:
        values = {
            name: (
                {"value": decl[param].get("value"), "basis": decl[param].get("basis")}
                if param in decl
                else None
            )
            for name, decl in by_scenario.items()
        }
        distinct = {repr(v) for v in values.values()}
        if len(distinct) > 1:
            differing.append({"parameter": param, "values": values})
        else:
            same.append(param)

    undeclared = []
    used = {name: m.get("parameters_used") or {} for name, m in manifests.items()}
    names = sorted({p for u in used.values() for p in u} - set(parameters))
    for param in names:
        values = {name: u.get(param) for name, u in used.items()}
        if len({repr(v) for v in values.values()}) > 1:
            undeclared.append({"parameter": param, "values": values})
    return {"differing": differing, "same": same, "undeclared_differences": undeclared}
