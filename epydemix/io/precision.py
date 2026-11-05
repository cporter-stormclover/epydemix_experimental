"""Precision-aware reporting: numbers rounded to what the data support.

A quantile of a Monte Carlo sample is only known up to its sampling error, so
reporting it with six decimals invites readers (and agents) to repeat digits
that mean nothing. The reporting commands round each statistic to the first
significant digit of its bootstrap standard error; parameters are rounded to
the significant figures their registry entry declares (``ParameterSpec.precision``).

The bootstrap is deterministic (fixed seed), so the same bundle always reports
the same digits: rounding never makes a result irreproducible.
"""

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

N_BOOT = 200
BOOT_SEED = 0
# Significant figures kept when a statistic has no sampling error (every
# simulation agrees, or there is a single one).
FALLBACK_SIGNIFICANT_FIGURES = 6
# Significant figures for parameters whose registry entry declares none.
DEFAULT_PARAMETER_PRECISION = 3
# Elements of a bootstrap array evaluated at once (time series are chunked).
_CHUNK_ELEMENTS = 4_000_000

PRECISION_NOTE = {
    "method": "bootstrap_se",
    "n_boot": N_BOOT,
    "seed": BOOT_SEED,
    "rule": "each statistic is rounded to the first significant digit of its "
    "bootstrap standard error; see raw for unrounded values",
}


def _quantiles_last_axis(values: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    """Linear-interpolation quantiles along the last axis (as pandas/numpy).

    Sorting once and interpolating is several times faster than
    ``np.nanquantile`` on bootstrap arrays; arrays with NaNs fall back to it.
    Returns an array with a leading axis of length ``len(quantiles)``.
    """
    if np.isnan(values).any():
        return np.nanquantile(values, list(quantiles), axis=-1)
    ordered = np.sort(values, axis=-1)
    n = ordered.shape[-1]
    out = []
    for q in quantiles:
        pos = q * (n - 1)
        lo = int(np.floor(pos))
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        out.append(ordered[..., lo] + frac * (ordered[..., hi] - ordered[..., lo]))
    return np.stack(out)


def _resample_indices(n: int, n_boot: int = N_BOOT) -> np.ndarray:
    return np.random.default_rng(BOOT_SEED).integers(0, n, size=(n_boot, n))


def bootstrap_se(samples: Iterable[float], quantiles: Sequence[float]) -> np.ndarray:
    """Bootstrap standard error of each quantile of a sample.

    Args:
        samples: One value per simulation.
        quantiles: Quantile levels.

    Returns:
        One standard error per quantile level (0 for a sample of one).
    """
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.zeros(len(quantiles))
    boot = x[_resample_indices(len(x))]  # (n_boot, n)
    est = _quantiles_last_axis(boot, quantiles)  # (Q, n_boot)
    return est.std(axis=1, ddof=1)


def bootstrap_se_statistic(
    samples: Iterable[float], statistic: Callable[[np.ndarray], np.ndarray]
) -> float:
    """Bootstrap standard error of a statistic (``statistic(x, axis=1)``)."""
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return 0.0
    boot = x[_resample_indices(len(x))]
    return float(np.std(statistic(boot), ddof=1))


def bootstrap_se_series(
    matrix: np.ndarray, quantiles: Sequence[float]
) -> Dict[float, np.ndarray]:
    """Bootstrap standard errors of quantiles across simulations, per time step.

    Args:
        matrix: Shape ``(T, n_simulations)``.
        quantiles: Quantile levels.

    Returns:
        ``{q: array of T standard errors}``.
    """
    matrix = np.asarray(matrix, dtype=float)
    T, n = matrix.shape
    out = {q: np.zeros(T) for q in quantiles}
    if n < 2:
        return out
    idx = _resample_indices(n)
    step = max(1, _CHUNK_ELEMENTS // (N_BOOT * n))
    for start in range(0, T, step):
        block = matrix[start : start + step][:, idx]  # (t, n_boot, n)
        est = _quantiles_last_axis(block, quantiles)  # (Q, t, n_boot)
        se = est.std(axis=2, ddof=1)
        for i, q in enumerate(quantiles):
            out[q][start : start + step] = se[i]
    return out


def bootstrap_se_mean_of_median(matrix: np.ndarray) -> float:
    """Bootstrap error of the time average of the median trajectory.

    Args:
        matrix: Shape ``(T, n_simulations)``.
    """
    matrix = np.asarray(matrix, dtype=float)
    T, n = matrix.shape
    if n < 2:
        return 0.0
    idx = _resample_indices(n)
    step = max(1, _CHUNK_ELEMENTS // max(1, T * n))
    means = []
    for start in range(0, N_BOOT, step):
        block = matrix[:, idx[start : start + step]]  # (T, b, n)
        means.append(_quantiles_last_axis(block, [0.5])[0].mean(axis=0))
    return float(np.std(np.concatenate(means), ddof=1))


def round_significant(value: float, figures: int) -> float:
    """Round to a number of significant figures."""
    if value == 0 or not math.isfinite(value):
        return value
    return float(f"{value:.{figures}g}")


def round_to_se(value: Any, se: Any) -> Any:
    """Round a value to the first significant digit of its standard error.

    A standard error of 12.3 rounds 1532.7 to 1530; one of 0.004 rounds
    0.03597 to 0.036. With no error (``se`` 0 or missing) the value keeps
    :data:`FALLBACK_SIGNIFICANT_FIGURES` significant figures. Values rounded to
    a whole number or coarser are returned as ``int``.
    """
    if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(
        value, bool
    ):
        return value
    value = float(value)
    if not math.isfinite(value):
        return value
    if se is None or not math.isfinite(float(se)) or float(se) <= 0:
        return round_significant(value, FALLBACK_SIGNIFICANT_FIGURES)
    digits = -int(math.floor(math.log10(float(se))))
    rounded = round(value, digits)
    if digits <= 0:
        return int(rounded)
    return float(rounded)


def round_by_se(tree: Any, se_tree: Any) -> Any:
    """Round every number of ``tree`` that has a standard error in ``se_tree``.

    The two trees have the same shape (dicts by key, lists by position).
    Numbers without a matching standard error are left as they are.
    """
    if isinstance(tree, dict):
        se_tree = se_tree if isinstance(se_tree, dict) else {}
        return {k: round_by_se(v, se_tree.get(k)) for k, v in tree.items()}
    if isinstance(tree, list):
        if isinstance(se_tree, list) and len(se_tree) == len(tree):
            return [round_by_se(v, s) for v, s in zip(tree, se_tree)]
        return list(tree)
    if isinstance(se_tree, (int, float, np.integer, np.floating)):
        return round_to_se(tree, se_tree)
    return tree


def reported_date(low: Optional[str], median: Optional[str], high: Optional[str]):
    """How to report a date known as a 90% interval.

    The median date when the interval spans at most one day, otherwise the
    interval itself as ``"YYYY-MM-DD/YYYY-MM-DD"`` (ISO 8601).
    """
    if low is None or high is None:
        return median
    if (pd.Timestamp(high) - pd.Timestamp(low)).days > 1:
        return f"{low}/{high}"
    return median


def add_reported_dates(result: Dict[str, Any]) -> Dict[str, Any]:
    """Add a ``reported`` entry to every peak-date quantile dict in a result."""
    for value in result.values():
        if not isinstance(value, dict):
            continue
        for key in ("peak_date",):
            dates = value.get(key)
            if isinstance(dates, dict):
                lo = dates.get("0.05")
                med = dates.get("0.5", dates.get("0.50"))
                hi = dates.get("0.95")
                dates["reported"] = reported_date(lo, med, hi)
    return result


def parameter_precision(name: str, registry=None, default=None) -> int:
    """Significant figures declared for a parameter, or the default."""
    if registry is not None and name in registry:
        precision = registry.get(name).precision
        if precision is not None:
            return int(precision)
    return int(default if default is not None else DEFAULT_PARAMETER_PRECISION)


def format_parameter(
    name: str,
    value: float,
    registry=None,
    inputs: Optional[List[str]] = None,
    input_precision: Optional[Dict[str, int]] = None,
) -> float:
    """Round a parameter value to the precision its inputs support.

    A parameter set directly keeps the significant figures of its registry
    entry. A derived parameter keeps the minimum precision of the inputs it
    was computed from (``inputs`` names them; ``input_precision`` gives the
    precision of inputs that are not registry parameters, such as R0).
    """
    if inputs:
        figures = []
        for i in inputs:
            if input_precision and i in input_precision:
                figures.append(int(input_precision[i]))
            else:
                figures.append(parameter_precision(i, registry))
        sig = min(figures)
    else:
        sig = parameter_precision(name, registry)
    return round_significant(float(value), sig)
