"""Inspection engine for querying .epx bundles.

Each function reads from the on-disk Parquet files, computes an answer,
and returns a compact dictionary suitable for JSON serialization to stdout.
"""

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pandas.tseries.frequencies import to_offset

from . import precision as _precision
from .bundle import load_bundle, load_bundle_dataframe

REPORT_MODES = ("raw", "rounded")


def inspect_bundle(
    path: str,
    command: str,
    report: str = "raw",
    **kwargs,
) -> Dict[str, Any]:
    """Dispatch an inspect command against a bundle.

    Args:
        path: Path to the .epx bundle directory.
        command: One of ``"manifest"``, ``"quantiles"``, ``"summary"``,
            ``"peak"``, ``"posterior"``, ``"fit"``.
        report: ``"raw"`` (default) returns full-precision floats.
            ``"rounded"`` rounds every statistic to the first significant
            digit of its bootstrap standard error and reports peak dates as an
            interval when their 90% interval spans more than one day (the
            CLI's default). The manifest is never rounded.
        **kwargs: Command-specific arguments.

    Returns:
        A dict suitable for JSON serialization.
    """
    commands = {
        "manifest": _cmd_manifest,
        "quantiles": _cmd_quantiles,
        "summary": _cmd_summary,
        "peak": _cmd_peak,
        "posterior": _cmd_posterior,
        "fit": _cmd_fit,
    }

    if command not in commands:
        raise ValueError(
            f"Unknown inspect command '{command}'. Available: {list(commands.keys())}"
        )

    if report not in REPORT_MODES:
        raise ValueError(f"report must be one of {REPORT_MODES}, got {report!r}")
    if report == "raw" or command == "manifest":
        return commands[command](path, **kwargs)
    result, se = commands[command](path, with_se=True, **kwargs)
    return _precision.add_reported_dates(_precision.round_by_se(result, se))


# ---------------------------------------------------------------------------
# Time-slice helper
# ---------------------------------------------------------------------------


def _apply_time_slice(
    df: pd.DataFrame,
    start: Optional[str] = None,
    end: Optional[str] = None,
    date_col: str = "date",
) -> pd.DataFrame:
    """Filter a DataFrame to a time window.

    Args:
        df: DataFrame with a date column.
        start: Start date (inclusive), ISO format string.
        end: End date (inclusive), ISO format string.
        date_col: Name of the date column.
    """
    if start is not None:
        start_dt = pd.to_datetime(start)
        df = df[df[date_col] >= start_dt]
    if end is not None:
        end_dt = pd.to_datetime(end)
        df = df[df[date_col] <= end_dt]
    return df


# Period-end aliases that pandas 2.2 renamed ("M" -> "ME", ...). pandas 3 only
# accepts the new names, and pandas < 2.2 only the old ones.
_PERIOD_END_ALIASES = {"M": "ME", "Q": "QE", "Y": "YE", "A": "YE"}


def _resolve_frequency(freq: str) -> str:
    """Accept both names of a period-end alias, whichever pandas is installed."""
    match = re.fullmatch(r"(\d*)([A-Za-z]+)", freq)
    if not match:
        return freq
    count, unit = match.group(1), match.group(2).upper()
    if unit in _PERIOD_END_ALIASES:
        new, old = _PERIOD_END_ALIASES[unit], unit
    elif unit in _PERIOD_END_ALIASES.values():
        new, old = unit, unit[:-1]
    else:
        return freq
    try:
        to_offset(count + new)
    except ValueError:
        return count + old
    return count + new


def _resample_grouped(
    df: pd.DataFrame,
    freq: str,
    agg: Dict[str, str],
) -> pd.DataFrame:
    """Resample time-series data grouped by sim_id.

    Args:
        df: DataFrame with sim_id, date, and value columns.
        freq: Pandas frequency string (e.g. 'W', 'M').
        agg: Aggregation method for each value column.
    """
    value_cols = list(agg)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    freq = _resolve_frequency(freq)
    resampled = (
        df.groupby("sim_id")[["date"] + value_cols]
        .apply(lambda g: g.set_index("date")[value_cols].resample(freq).agg(agg))
        .reset_index()
    )
    return resampled


def _load_variables(
    path: str,
    manifest: Dict,
    variables: List[str],
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load variables from whichever file has them, compartments or transitions.

    Returns the variables merged on (sim_id, date), and for each one how to
    aggregate it when resampling: compartments are levels, so a period keeps
    its last value, while transitions are flows, so a period sums them.
    """
    files = manifest.get("files", {})
    sources = {}
    for var in variables:
        for file_key in ("compartments", "transitions"):
            if var in files.get(file_key, {}).get("columns", {}):
                sources[var] = file_key
                break
        else:
            available = [
                c
                for file_key in ("compartments", "transitions")
                for c in files.get(file_key, {}).get("columns", {})
                if c.endswith("_total")
            ]
            raise ValueError(
                f"Unknown variable '{var}'. Available: {', '.join(available)} "
                "(and the per-group columns listed in the manifest)"
            )

    df = None
    for file_key in ("compartments", "transitions"):
        cols = [v for v in variables if sources[v] == file_key]
        if not cols:
            continue
        part = load_bundle_dataframe(path, file_key, columns=["sim_id", "date"] + cols)
        df = part if df is None else df.merge(part, on=["sim_id", "date"])

    agg = {v: "last" if sources[v] == "compartments" else "sum" for v in variables}
    return df, agg


def _resolve_variables(
    manifest: Dict,
    variables: Optional[List[str]],
    file_key: str = "compartments",
) -> List[str]:
    """Resolve variable names from manifest, defaulting to _total vars."""
    if variables:
        return variables

    files = manifest.get("files", {})
    if file_key in files:
        cols = files[file_key].get("columns", {})
        # Default to _total variables (most useful for agents)
        total_vars = [c for c in cols if c.endswith("_total")]
        if total_vars:
            return total_vars
        # Fall back to all non-index columns
        return [c for c in cols if c not in ("sim_id", "date")]
    return []


# ---------------------------------------------------------------------------
# Per-simulation statistics shared by inspect, compare and summary.json
# ---------------------------------------------------------------------------

# The quantile levels reported for per-simulation statistics (peak, final
# value, ...). ``inspect summary`` and ``inspect peak`` label the median
# "0.50"; summary.json uses the plain "0.5" that ``inspect quantiles`` uses.
SUMMARY_QUANTILES = (0.05, 0.5, 0.95)


def _quantile_key(q: float, key_format: str) -> str:
    if key_format == "legacy":
        return f"{q:.2f}"
    return str(q)


def _per_simulation_stats(pivoted: pd.DataFrame) -> Dict[str, pd.Series]:
    """Peak value, peak date and final value of each simulation.

    Args:
        pivoted: One row per date, one column per simulation.
    """
    return {
        "peak": pivoted.max(axis=0),
        "peak_date": pd.to_datetime(pivoted.idxmax(axis=0)),
        "final": pivoted.iloc[-1],
    }


def _value_quantiles(
    values: pd.Series,
    quantiles=SUMMARY_QUANTILES,
    key_format: str = "plain",
) -> Dict[str, float]:
    """Quantiles of a per-simulation statistic, keyed by quantile level."""
    return {_quantile_key(q, key_format): float(values.quantile(q)) for q in quantiles}


def _value_quantiles_se(
    values: pd.Series,
    quantiles=SUMMARY_QUANTILES,
    key_format: str = "plain",
) -> Dict[str, float]:
    """Bootstrap standard errors matching :func:`_value_quantiles`."""
    ses = _precision.bootstrap_se(values.values, list(quantiles))
    return {_quantile_key(q, key_format): float(se) for q, se in zip(quantiles, ses)}


def _series_quantiles(pivoted: pd.DataFrame, quantiles, with_se: bool):
    """Quantiles across simulations at each time step, and their errors."""
    values = {
        str(q): [float(v) for v in pivoted.quantile(q, axis=1).values]
        for q in quantiles
    }
    if not with_se:
        return values, None
    ses = _precision.bootstrap_se_series(pivoted.values, list(quantiles))
    return values, {str(q): [float(v) for v in ses[q]] for q in quantiles}


def _cached_se(path: str, var: str, stat: str) -> Optional[Dict[str, float]]:
    """Standard errors cached in summary.json, keyed like inspect's output.

    Only valid for statistics over the whole simulation window; callers do
    not use it when a time slice is requested.
    """
    summary_path = Path(path) / "summary.json"
    if not summary_path.exists():
        return None
    try:
        with open(summary_path) as f:
            cached = json.load(f)["mc_se"]["variables"][var][stat]
    except (KeyError, TypeError, ValueError):
        return None
    return {_quantile_key(float(q), "legacy"): v for q, v in cached.items()}


def _date_quantiles(
    dates: pd.Series,
    quantiles=SUMMARY_QUANTILES,
    key_format: str = "plain",
) -> Dict[str, Optional[str]]:
    """Quantiles of per-simulation dates, by position in the sorted dates.

    Dates cannot be interpolated like numbers, so the q-quantile is the date at
    position ``int(q * n)`` of the sorted dates (clamped to the valid range).
    """
    sorted_dates = pd.to_datetime(dates).sort_values()
    n = len(sorted_dates)
    result = {}
    for q in quantiles:
        key = _quantile_key(q, key_format)
        if n == 0:
            result[key] = None
            continue
        pos = min(n - 1, max(0, int(q * n)))
        result[key] = str(sorted_dates.iloc[pos])[:10]
    return result


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _cmd_manifest(path: str, **kwargs) -> Dict[str, Any]:
    """Return the manifest."""
    return load_bundle(path)


def _cmd_quantiles(
    path: str,
    variables: Optional[List[str]] = None,
    quantiles: Optional[List[float]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    resample: Optional[str] = None,
    with_se: bool = False,
    **kwargs,
):
    """Compute quantiles across simulations for selected variables.

    Args:
        variables: Variable names to compute quantiles for. Defaults to _total vars.
        quantiles: Quantile levels. Defaults to [0.05, 0.25, 0.5, 0.75, 0.95].
        start: Start date for time slice.
        end: End date for time slice.
        resample: Temporal resampling frequency (e.g. 'W', 'M').
        with_se: Also return the bootstrap standard error of every number.
    """
    if quantiles is None:
        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]

    manifest = load_bundle(path)
    variables = _resolve_variables(manifest, variables)

    df, agg = _load_variables(path, manifest, variables)
    df = _apply_time_slice(df, start, end)

    if resample:
        df = _resample_grouped(df, resample, agg)

    # Compute quantiles
    dates = sorted(df["date"].unique())
    result: Dict[str, Any] = {
        "dates": [str(d)[:10] for d in dates],
    }
    se: Dict[str, Any] = {}

    for var in variables:
        pivoted = df.pivot(index="date", columns="sim_id", values=var)
        result[var], se[var] = _series_quantiles(pivoted, quantiles, with_se)

    return (result, se) if with_se else result


def _cmd_summary(
    path: str,
    variables: Optional[List[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    with_se: bool = False,
    **kwargs,
):
    """Compute summary statistics for selected variables.

    Returns peak date/value, cumulative totals, and final state
    as quantiles across simulations.
    """
    manifest = load_bundle(path)
    variables = _resolve_variables(manifest, variables)

    df, _ = _load_variables(path, manifest, variables)
    df = _apply_time_slice(df, start, end)
    whole_window = start is None and end is None

    result = {}
    se = {}
    for var in variables:
        pivoted = df.pivot(index="date", columns="sim_id", values=var)
        stats = _per_simulation_stats(pivoted)
        peak_dates_sorted = stats["peak_date"].sort_values()
        n_peaks = len(peak_dates_sorted)

        var_summary = {
            "peak_date_median": str(peak_dates_sorted.iloc[n_peaks // 2])[:10]
            if n_peaks > 0
            else None,
            "peak_value": _value_quantiles(stats["peak"], key_format="legacy"),
            "final_value": _value_quantiles(stats["final"], key_format="legacy"),
            "mean_across_time": {
                "0.50": float(pivoted.quantile(0.5, axis=1).mean()),
            },
        }
        if with_se:
            dates = _date_quantiles(stats["peak_date"], key_format="legacy")
            var_summary["peak_date_reported"] = _precision.reported_date(
                dates["0.05"], dates["0.50"], dates["0.95"]
            )
            cached = {
                stat: _cached_se(path, var, stat) if whole_window else None
                for stat in ("peak", "final")
            }
            se[var] = {
                "peak_value": cached["peak"]
                or _value_quantiles_se(stats["peak"], key_format="legacy"),
                "final_value": cached["final"]
                or _value_quantiles_se(stats["final"], key_format="legacy"),
                "mean_across_time": {
                    "0.50": _precision.bootstrap_se_mean_of_median(pivoted.values)
                },
            }
        result[var] = var_summary

    return (result, se) if with_se else result


def _cmd_peak(
    path: str,
    variables: Optional[List[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    with_se: bool = False,
    **kwargs,
):
    """Find peak timing and magnitude for selected variables."""
    manifest = load_bundle(path)
    variables = _resolve_variables(manifest, variables)

    df, _ = _load_variables(path, manifest, variables)
    df = _apply_time_slice(df, start, end)
    whole_window = start is None and end is None

    result = {}
    se = {}
    for var in variables:
        pivoted = df.pivot(index="date", columns="sim_id", values=var)
        stats = _per_simulation_stats(pivoted)
        result[var] = {
            "peak_date": _date_quantiles(stats["peak_date"], key_format="legacy"),
            "peak_value": _value_quantiles(stats["peak"], key_format="legacy"),
        }
        if with_se:
            cached = _cached_se(path, var, "peak") if whole_window else None
            se[var] = {
                "peak_value": cached
                or _value_quantiles_se(stats["peak"], key_format="legacy")
            }

    return (result, se) if with_se else result


def _posterior_stats(vals: pd.Series, with_se: bool):
    """Mean, standard deviation, 95% interval and median of a sample."""
    stats = {
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "ci95": [
            float(vals.quantile(0.025)),
            float(vals.quantile(0.975)),
        ],
        "median": float(vals.median()),
    }
    if not with_se:
        return stats, None
    x = vals.values
    q_se = _precision.bootstrap_se(x, [0.025, 0.5, 0.975])
    se = {
        "mean": _precision.bootstrap_se_statistic(x, lambda b: b.mean(axis=1)),
        "std": _precision.bootstrap_se_statistic(x, lambda b: b.std(axis=1, ddof=1)),
        "ci95": [float(q_se[0]), float(q_se[2])],
        "median": float(q_se[1]),
    }
    return stats, se


def _cmd_posterior(
    path: str,
    generation: Optional[int] = None,
    with_se: bool = False,
    **kwargs,
):
    """Summarize posterior distributions from calibration results."""
    df = load_bundle_dataframe(path, "posterior")

    if generation is not None:
        df = df[df["generation"] == generation]
    else:
        # Use last generation
        max_gen = df["generation"].max()
        df = df[df["generation"] == max_gen]

    param_cols = [c for c in df.columns if c != "generation"]
    result = {}
    se = {}
    for col in param_cols:
        result[col], se[col] = _posterior_stats(df[col].dropna(), with_se)

    return (result, se) if with_se else result


def _cmd_fit(
    path: str,
    variables: Optional[List[str]] = None,
    quantiles: Optional[List[float]] = None,
    with_se: bool = False,
    **kwargs,
):
    """Get calibration fit trajectories (observed vs. simulated quantiles).

    This reads the trajectories from the calibration bundle and computes
    quantiles of the accepted simulations.
    """
    if quantiles is None:
        quantiles = [0.05, 0.5, 0.95]

    bundle_path = Path(path)
    manifest = load_bundle(path)

    # Try to load observed data
    observed_path = bundle_path / "observed_data.parquet"
    observed = None
    if observed_path.exists():
        observed = pd.read_parquet(observed_path)

    # Load trajectories
    traj_path = bundle_path / "trajectories.parquet"
    if not traj_path.exists():
        raise FileNotFoundError(
            f"No trajectories.parquet found in calibration bundle at {bundle_path}"
        )

    traj_df = pd.read_parquet(traj_path)

    # The trajectory "variable" column may be "data" (the internal key used by
    # ABCSampler) rather than the user-facing variable name (e.g. "Infected_total").
    # Build a mapping from user-facing name → trajectory key so that both work.
    traj_vars = set(traj_df["variable"].unique())
    target_variable = manifest.get("calibration", {}).get("target_variable")

    if variables is None:
        # Default: expose the target variable name if known, else raw keys
        if target_variable and "data" in traj_vars:
            variables = [target_variable]
        else:
            variables = list(traj_vars)

    # Map user-requested variable names to actual keys in the parquet.
    # "data" is the internal fallback key written by ABCSampler when only one
    # target variable is calibrated.
    def _resolve_traj_key(var: str) -> str:
        if var in traj_vars:
            return var
        if "data" in traj_vars and var == target_variable:
            return "data"
        return var  # will produce empty df → skipped below

    result: Dict[str, Any] = {}
    se: Dict[str, Any] = {}
    for var in variables:
        traj_key = _resolve_traj_key(var)
        var_df = traj_df[traj_df["variable"] == traj_key]
        if var_df.empty:
            continue

        pivoted = var_df.pivot(index="timestep", columns="sim_id", values="value")
        result[var], se[var] = _series_quantiles(pivoted, quantiles, with_se)

    if observed is not None:
        # observed_data.parquet is in long format: [timestep, variable, value].
        # Reshape to {"var_name": [values...]} so agents can do fit["observed"][VAR].
        obs_dict: Dict[str, Any] = {}
        for var_name, grp in observed.groupby("variable"):
            grp_sorted = grp.sort_values("timestep")
            obs_dict[str(var_name)] = [float(v) for v in grp_sorted["value"].values]
        result["observed"] = obs_dict

    return (result, se) if with_se else result


# ---------------------------------------------------------------------------
# Cross-bundle comparison
# ---------------------------------------------------------------------------

# Built-in metric registry — each metric is a function that takes
# (comp_df, manifest, **kwargs) and returns a dict with "value" and
# optionally "ci90" (a two-element list).


def _legacy_attack_rate_samples(comp) -> Optional[pd.Series]:
    """Per-simulation attack rate (fraction) from susceptible depletion.

    ``1 - S_final / S_initial``. This counts everyone who left the susceptible
    compartment, so it overstates the attack rate of models in which people
    leave it without being infected (vaccination). Returns None when the bundle
    has no susceptible compartment.
    """
    total_cols = [
        c for c in comp.columns if c.endswith("_total") and c not in ("sim_id", "date")
    ]
    s_col = None
    for c in total_cols:
        if c.startswith("S") or c.startswith("Susceptible"):
            s_col = c
            break
    if s_col is None:
        return None

    first_day = comp.groupby("sim_id")[s_col].first()
    last_day = comp.groupby("sim_id")[s_col].last()
    return 1.0 - last_day / first_day


def _sampled_metric(samples: pd.Series, **extra) -> Dict[str, Any]:
    """Median and 90% interval of a per-simulation metric.

    The samples travel with the result under ``_samples`` so that
    :func:`compare_bundles` can compute bootstrap errors; it removes them
    before returning.
    """
    return {
        "median": float(samples.median()),
        "ci90": [float(samples.quantile(0.05)), float(samples.quantile(0.95))],
        **extra,
        "_samples": samples,
    }


def infected_compartments_of(manifest: Dict[str, Any]) -> Optional[List[str]]:
    """Infected-type compartments of the model that wrote a bundle."""
    from ..model.reproduction import INFECTED_LIKE

    model = manifest.get("model") or {}
    if model.get("infected_compartments"):
        return list(model["infected_compartments"])
    compartments = model.get("compartments") or []
    if "Susceptible" in compartments and model.get("type", "predefined") != "custom":
        return [c for c in compartments if c in INFECTED_LIKE]
    return None


def infection_flow_columns_of(manifest: Dict[str, Any]) -> List[str]:
    """Transition columns that move people into infection, from the manifest."""
    from ..model.reproduction import infection_flow_columns

    infected = infected_compartments_of(manifest)
    if not infected:
        return []
    columns = manifest.get("files", {}).get("transitions", {}).get("columns", {})
    compartments = (manifest.get("model") or {}).get("compartments") or []
    return infection_flow_columns(columns, compartments, infected)


def attack_rate_samples(comp, flows, manifest) -> Tuple[Optional[pd.Series], Dict]:
    """Per-simulation attack rate (fraction of the initial population).

    Model-aware: the sum of every flow into an infected-type compartment from
    a compartment that is not one (for SEIR with a leaky vaccine,
    ``Susceptible → Exposed`` plus ``Vaccinated → Exposed``), divided by the
    initial population. Predefined models use fixed infected-type names;
    custom models need ``role: infected`` on their compartments. Without that
    information (or without flows) it falls back to susceptible depletion,
    marked ``"method": "legacy"``.

    Returns:
        ``(samples, info)``; ``info`` names the method and the flows used.
    """
    cols = infection_flow_columns_of(manifest)
    if flows is not None and cols and all(c in flows.columns for c in cols):
        compartments = (manifest.get("model") or {}).get("compartments") or []
        totals = [f"{c}_total" for c in compartments if f"{c}_total" in comp.columns]
        population = (
            comp.sort_values("date").groupby("sim_id")[totals].first().sum(axis=1)
        )
        infections = flows.groupby("sim_id")[cols].sum().sum(axis=1)
        return infections / population, {
            "method": "transition_flows",
            "flows": cols,
            "infected_compartments": infected_compartments_of(manifest),
        }
    legacy = _legacy_attack_rate_samples(comp)
    info = {
        "method": "legacy",
        "note": "susceptible depletion; overstates the attack rate when people "
        "leave the susceptible compartment uninfected. Custom models give the "
        "exact attack rate when their compartments carry role: infected",
    }
    return legacy, info


def _metric_attack_rate(comp, manifest, flows=None, **kw):
    """Percentage of the population infected during the simulation."""
    attack, info = attack_rate_samples(comp, flows, manifest)
    if attack is None:
        return {"value": None, "note": "no susceptible compartment found"}
    return _sampled_metric(attack * 100, units="percent", method=info["method"])


def _metric_peak(comp, manifest, variable=None, **kw):
    """Peak value of a variable across simulations."""
    var = variable or _default_var(comp, prefix="I")
    if var not in comp.columns:
        return {"value": None, "note": f"{var} not found"}
    peaks = comp.groupby("sim_id")[var].max()
    return _sampled_metric(peaks, variable=var)


def _metric_peak_date(comp, manifest, variable=None, **kw):
    """Date of peak value."""
    var = variable or _default_var(comp, prefix="I")
    if var not in comp.columns:
        return {"value": None, "note": f"{var} not found"}
    comp_sorted = comp.sort_values("date")
    idx = comp_sorted.groupby("sim_id")[var].idxmax()
    peak_dates = comp_sorted.loc[idx, "date"]
    peak_dates = pd.to_datetime(peak_dates).sort_values()
    n = len(peak_dates)
    return {
        "median": str(peak_dates.iloc[int(0.5 * n)])[:10] if n > 0 else None,
        "ci90": [
            str(peak_dates.iloc[max(0, int(0.05 * n))])[:10],
            str(peak_dates.iloc[min(n - 1, int(0.95 * n))])[:10],
        ]
        if n > 0
        else None,
        "variable": var,
    }


def _metric_total_deaths(comp, manifest, **kw):
    """Total deaths at end of simulation."""
    d_col = None
    for c in comp.columns:
        if c.endswith("_total") and (c.startswith("D") or c.startswith("Dead")):
            d_col = c
            break
    if d_col is None:
        return {"value": None, "note": "no death compartment found"}
    final = comp.groupby("sim_id")[d_col].last()
    return _sampled_metric(final, variable=d_col)


def _metric_days_over(comp, manifest, variable=None, threshold=None, **kw):
    """Number of days the median of a variable exceeds a threshold."""
    if threshold is None:
        return {"value": None, "note": "threshold required (e.g. days_over:500)"}
    threshold = float(threshold)
    var = variable or _default_var(comp, prefix="H")
    if var not in comp.columns:
        return {"value": None, "note": f"{var} not found"}
    median_ts = comp.groupby("date")[var].median()
    days = int((median_ts > threshold).sum())
    return {
        "value": days,
        "threshold": threshold,
        "variable": var,
    }


def _metric_final_value(comp, manifest, variable=None, **kw):
    """Final value of a variable across simulations."""
    var = variable or _default_var(comp, prefix="I")
    if var not in comp.columns:
        return {"value": None, "note": f"{var} not found"}
    final = comp.groupby("sim_id")[var].last()
    return _sampled_metric(final, variable=var)


def _default_var(comp, prefix="I"):
    """Find the first _total column matching a prefix."""
    for c in comp.columns:
        if c.endswith("_total") and c.split("_")[0].startswith(prefix):
            return c
    # Fall back to first _total column that isn't S
    for c in comp.columns:
        if c.endswith("_total") and not c.startswith("S"):
            return c
    return None


COMPARE_METRICS = {
    "attack_rate": _metric_attack_rate,
    "peak": _metric_peak,
    "peak_date": _metric_peak_date,
    "total_deaths": _metric_total_deaths,
    "days_over": _metric_days_over,
    "final_value": _metric_final_value,
}


def _metric_se(metric: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Bootstrap errors of a metric's median and 90% interval."""
    samples = metric.get("_samples")
    if samples is None:
        return None
    se = _precision.bootstrap_se(samples.values, [0.05, 0.5, 0.95])
    return {"median": float(se[1]), "ci90": [float(se[0]), float(se[2])]}


def compare_bundles(
    bundles: Dict[str, str],
    metrics: Optional[List[str]] = None,
    variables: Optional[List[str]] = None,
    with_se: bool = False,
):
    """Compare multiple bundles on a set of metrics.

    Args:
        bundles: Mapping of scenario name → bundle path.
        metrics: List of metric names to compute. Each can be a plain name
            (e.g. ``"attack_rate"``) or include a parameter after a colon
            (e.g. ``"days_over:500"``).  Defaults to all standard metrics
            (excluding ``days_over`` which requires a threshold).
        variables: Optional list of variable names to pass to variable-specific
            metrics (peak, peak_date, final_value).  If not provided, metrics
            will auto-detect the most relevant variable.
        with_se: Also return the bootstrap standard errors of every median
            and interval bound, as a second dict of the same shape.

    Returns:
        A dict with structure ``{scenario_name: {metric_name: result}}``,
        or ``(result, standard_errors)`` when ``with_se`` is true.
    """
    if metrics is None:
        metrics = ["attack_rate", "peak", "peak_date", "total_deaths"]

    # Parse metric specs (name or name:param)
    parsed_metrics = []
    for m in metrics:
        if ":" in m:
            name, param = m.split(":", 1)
            parsed_metrics.append((name, param))
        else:
            parsed_metrics.append((m, None))

    # Validate metric names
    for name, _ in parsed_metrics:
        if name not in COMPARE_METRICS:
            raise ValueError(
                f"Unknown metric '{name}'. Available: {list(COMPARE_METRICS.keys())}"
            )

    result = {}
    errors = {}
    for scenario_name, bundle_path in bundles.items():
        manifest = load_bundle(bundle_path)
        comp = load_bundle_dataframe(bundle_path, "compartments")
        # Requested transition columns (e.g. incidence) live in another file
        transition_cols = manifest.get("files", {}).get("transitions", {})
        flows = [
            v
            for v in variables or []
            if v not in comp.columns and v in transition_cols.get("columns", {})
        ]
        if flows:
            comp = comp.merge(
                load_bundle_dataframe(
                    bundle_path, "transitions", columns=["sim_id", "date"] + flows
                ),
                on=["sim_id", "date"],
            )
        infection_flows = None
        if any(name == "attack_rate" for name, _ in parsed_metrics):
            cols = infection_flow_columns_of(manifest)
            if cols:
                infection_flows = load_bundle_dataframe(
                    bundle_path, "transitions", columns=["sim_id", "date"] + cols
                )

        scenario_result = {}
        scenario_errors = {}
        for metric_name, metric_param in parsed_metrics:
            fn = COMPARE_METRICS[metric_name]
            kwargs = {}
            if variables:
                kwargs["variable"] = variables[0]
            if metric_param is not None:
                kwargs["threshold"] = metric_param
            if metric_name == "attack_rate":
                kwargs["flows"] = infection_flows
            value = fn(comp, manifest, **kwargs)
            se = _metric_se(value) if with_se else None
            if se is not None:
                scenario_errors[metric_name] = se
            value.pop("_samples", None)
            scenario_result[metric_name] = value

        result[scenario_name] = scenario_result
        errors[scenario_name] = scenario_errors

    return (result, errors) if with_se else result


def _metric_number(metric: Dict[str, Any]) -> Tuple[Optional[str], Any]:
    """The field of a metric that deltas are computed on, and its value."""
    for field in ("median", "value"):
        if field in metric:
            value = metric[field]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return field, value
            return field, None
    return None, None


def compare_deltas(
    result: Dict[str, Any],
    baseline: str,
    errors: Optional[Dict[str, Any]] = None,
):
    """Differences of every scenario from a baseline (scenario minus baseline).

    Deltas are taken on each metric's median (or value). With ``errors``
    (from ``compare_bundles(..., with_se=True)``), also returns the standard
    error of each delta, combining the two scenarios' errors in quadrature.

    Returns:
        ``{scenario: {metric: delta}}``, or ``(deltas, delta_errors)`` when
        ``errors`` is given.
    """
    if baseline not in result:
        raise KeyError(
            f"Baseline '{baseline}' not found. Available: {list(result.keys())}"
        )
    base_data = result[baseline]
    deltas = {}
    delta_errors = {}
    for scenario, metrics_data in result.items():
        if scenario == baseline or scenario.startswith("_"):
            continue
        scenario_delta = {}
        scenario_error = {}
        for metric_name, metric_val in metrics_data.items():
            base_val = base_data.get(metric_name, {})
            field, sv = _metric_number(metric_val)
            base_field, bv = _metric_number(base_val)
            if field is None or field != base_field or sv is None or bv is None:
                continue
            scenario_delta[metric_name] = sv - bv
            if errors is not None and field == "median":
                se_s = errors.get(scenario, {}).get(metric_name, {}).get("median")
                se_b = errors.get(baseline, {}).get(metric_name, {}).get("median")
                if se_s is not None and se_b is not None:
                    scenario_error[metric_name] = math.sqrt(se_s**2 + se_b**2)
        deltas[scenario] = scenario_delta
        delta_errors[scenario] = scenario_error
    return (deltas, delta_errors) if errors is not None else deltas


def round_comparison(result: Dict[str, Any], errors: Dict[str, Any]) -> Dict[str, Any]:
    """Round a comparison (with its deltas) to its bootstrap errors.

    Peak dates gain a ``reported`` field: the median date, or the 90%
    interval when it spans more than one day.
    """
    rounded = _precision.round_by_se(result, errors)
    for scenario, metrics_data in rounded.items():
        if scenario.startswith("_") or not isinstance(metrics_data, dict):
            continue
        peak_date = metrics_data.get("peak_date")
        if isinstance(peak_date, dict) and peak_date.get("ci90"):
            lo, hi = peak_date["ci90"]
            peak_date["reported"] = _precision.reported_date(
                lo, peak_date.get("median"), hi
            )
    return rounded
