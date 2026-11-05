"""Rate expressions for declarative custom models, with a whitelisted grammar.

An expression such as ``"gamma * p_hosp"`` or
``"beta_h * min(1, H / (capacity * pop))"`` may use:

* names: parameters, compartments (their current size) and ``pop`` (the
  size of each demographic group);
* numeric constants;
* ``+``, ``-``, ``*``, ``/`` and unary ``-``;
* calls to ``min`` and ``max`` (two or more arguments, no keywords).

Nothing else is accepted: no attribute access, subscripts, comprehensions,
lambdas, comparisons, other calls or keyword arguments, and at most 200
characters. The expression is parsed once into a tree of the allowed nodes
and evaluated by walking that tree; it is never passed to ``eval``.
Evaluation is element-wise on numpy arrays (``min``/``max`` become
``np.minimum``/``np.maximum``).
"""

import ast
from typing import Any, Dict, FrozenSet, Iterable

import numpy as np

MAX_LENGTH = 200
FUNCTIONS = {"min": np.minimum, "max": np.maximum}
_BINARY = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: np.divide,
}


class ExpressionError(ValueError):
    """An expression outside the whitelisted grammar, or one that cannot run."""


def _check(node: ast.AST, source: str, allowed: FrozenSet[str]) -> None:
    if isinstance(node, ast.Expression):
        _check(node.body, source, allowed)
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in _BINARY:
            raise ExpressionError(
                f"operator {type(node.op).__name__} is not allowed in {source!r}; "
                "use + - * / and min/max"
            )
        _check(node.left, source, allowed)
        _check(node.right, source, allowed)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, ast.USub):
            raise ExpressionError(f"only unary minus is allowed in {source!r}")
        _check(node.operand, source, allowed)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ExpressionError(f"only numeric constants are allowed in {source!r}")
    elif isinstance(node, ast.Name):
        if node.id in FUNCTIONS:
            raise ExpressionError(f"{node.id} must be called, in {source!r}")
        if node.id not in allowed:
            raise ExpressionError(
                f"unknown symbol {node.id!r} in {source!r}; allowed: "
                f"{', '.join(sorted(allowed)) or 'none'}"
            )
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            raise ExpressionError(
                f"only min(...) and max(...) may be called, in {source!r}"
            )
        if node.keywords:
            raise ExpressionError(f"keyword arguments are not allowed in {source!r}")
        if len(node.args) < 2:
            raise ExpressionError(
                f"{node.func.id} needs at least two arguments, in {source!r}"
            )
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                raise ExpressionError(
                    f"starred arguments are not allowed in {source!r}"
                )
            _check(arg, source, allowed)
    else:
        raise ExpressionError(
            f"disallowed syntax ({type(node).__name__}) in {source!r}"
        )


class Expression:
    """A parsed, checked rate expression."""

    def __init__(self, source: str, allowed_names: Iterable[str]):
        if not isinstance(source, str):
            raise ExpressionError(f"an expression must be text, got {source!r}")
        if len(source) > MAX_LENGTH:
            raise ExpressionError(f"expression longer than {MAX_LENGTH} characters")
        if not source.strip():
            raise ExpressionError("empty expression")
        try:
            tree = ast.parse(source.strip(), mode="eval")
        except (SyntaxError, ValueError) as e:
            raise ExpressionError(f"{source!r} is not a valid expression: {e}")
        self.allowed = frozenset(allowed_names)
        _check(tree, source, self.allowed)
        self.source = source
        self.tree = tree.body
        self.names = frozenset(
            n.id
            for n in ast.walk(tree)
            if isinstance(n, ast.Name) and n.id not in FUNCTIONS
        )

    def __repr__(self) -> str:
        return f"Expression({self.source!r})"

    def __deepcopy__(self, memo):
        return self  # immutable

    def depends_on(self, names: Iterable[str]) -> bool:
        return bool(self.names & set(names))

    def evaluate(self, env: Dict[str, Any]):
        """Value of the expression with the names bound in ``env``."""
        missing = self.names - set(env)
        if missing:
            raise ExpressionError(f"{self.source!r} needs {sorted(missing)}")
        with np.errstate(divide="ignore", invalid="ignore"):
            return _evaluate(self.tree, env)


def _evaluate(node: ast.AST, env: Dict[str, Any]):
    if isinstance(node, ast.BinOp):
        return _BINARY[type(node.op)](
            _evaluate(node.left, env), _evaluate(node.right, env)
        )
    if isinstance(node, ast.UnaryOp):
        return np.negative(_evaluate(node.operand, env))
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.Call):
        values = [_evaluate(a, env) for a in node.args]
        result = values[0]
        for v in values[1:]:
            result = FUNCTIONS[node.func.id](result, v)
        return result
    raise ExpressionError(f"cannot evaluate {type(node).__name__}")  # pragma: no cover


def compile_expression(source: str, allowed_names: Iterable[str]) -> Expression:
    """Parse and check an expression; raise :class:`ExpressionError` if refused."""
    return Expression(source, allowed_names)
