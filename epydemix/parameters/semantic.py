"""Cross-parameter rules a config must satisfy (``semantic_checks``).

Range checks look at one parameter at a time. Semantic checks relate values
to each other and to the rest of the config, for example "a positive
``vaccination_rate`` requires ``model.vaccination: true``". They are plain
data, so a registry can list them in its JSON Schema and agents can read them
before writing a config. Three rules are supported:

``requires``
    When ``parameter`` is set to a value greater than ``greater_than``, every
    dotted config path in ``requires`` must have the given value.
``ordered_dates``
    In every item of the list ``section``, ``start`` must come before ``end``.
``range``
    Every value at ``quantity`` (a dotted path; ``*`` matches any key) must lie
    in ``[min, max]``.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

RULES = ("requires", "ordered_dates", "range")
SEVERITIES = ("error", "warning")


@dataclass
class SemanticCheck:
    """A rule relating parameters to each other or to the config."""

    name: str
    rule: str
    message: str
    severity: str = "error"
    parameter: Optional[str] = None
    greater_than: float = 0.0
    requires: Dict[str, Any] = field(default_factory=dict)
    section: Optional[str] = None
    start: str = "start_date"
    end: str = "end_date"
    quantity: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None

    def __post_init__(self) -> None:
        if self.rule not in RULES:
            raise ValueError(f"Unknown semantic rule '{self.rule}'. Rules: {RULES}")
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}")

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, {}, [])}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SemanticCheck":
        return cls(**d)

    def evaluate(self, config: Dict[str, Any]) -> List[str]:
        """Messages for every violation of this rule in a config."""
        if self.rule == "requires":
            return self._requires(config)
        if self.rule == "ordered_dates":
            return self._ordered_dates(config)
        return self._range(config)

    def _requires(self, config):
        params = config.get("parameters")
        value = params.get(self.parameter) if isinstance(params, dict) else None
        if not _is_number(value) or value <= self.greater_than:
            return []
        problems = []
        for path, expected in self.requires.items():
            actual = _get(config, path)
            if actual != expected:
                problems.append(
                    f"{self.message} (parameters.{self.parameter} = {value} needs "
                    f"{path}: {_yaml(expected)}, the config has "
                    f"{_yaml(actual) if actual is not None else 'none'})"
                )
        return problems

    def _ordered_dates(self, config):
        import pandas as pd

        problems = []
        items = config.get(self.section)
        for i, item in enumerate(items if isinstance(items, list) else []):
            if not isinstance(item, dict):
                continue
            start, end = item.get(self.start), item.get(self.end)
            try:
                ordered = pd.Timestamp(start) < pd.Timestamp(end)
            except (TypeError, ValueError):
                continue  # missing or malformed dates are reported elsewhere
            if not ordered:
                problems.append(
                    f"{self.section}[{i}]: {self.start} ({start}) must be before "
                    f"{self.end} ({end}). {self.message}"
                )
        return problems

    def _range(self, config):
        problems = []
        for path, value in _find(config, self.quantity.split(".")):
            if not _is_number(value):
                continue
            if (self.min is not None and value < self.min) or (
                self.max is not None and value > self.max
            ):
                problems.append(f"{path} = {value}: {self.message}")
        return problems


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _yaml(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _get(config, dotted: str):
    node = config
    for part in dotted.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def _find(node, parts, prefix=""):
    if not parts:
        yield prefix, node
        return
    head, rest = parts[0], parts[1:]
    if not isinstance(node, dict):
        return
    keys = list(node) if head == "*" else [head] if head in node else []
    for key in keys:
        yield from _find(node[key], rest, f"{prefix}.{key}" if prefix else str(key))
