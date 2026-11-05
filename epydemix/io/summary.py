"""``summary.json``: the headline numbers of a bundle, written next to its manifest.

``epydemix run``, ``calibrate`` and ``project`` write it at the end of every
run, so that scripts and agents can read the results without calling
``epydemix inspect``. It is a cache: every number in it is computed by the
inspection engine from the bundle's Parquet files, ``inspect`` keeps reading
those files, and no command requires ``summary.json`` to be present.
:func:`build_summary` recomputes the same content for any bundle.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .bundle import load_bundle, load_bundle_dataframe
from .inspect import (
    SUMMARY_QUANTILES,
    _cmd_posterior,
    _date_quantiles,
    _legacy_attack_rate_samples,
    _per_simulation_stats,
    _value_quantiles,
)
from .json_utils import NumpySafeEncoder

SUMMARY_SCHEMA_VERSION = 1
SUMMARY_FILENAME = "summary.json"


def _variable_summaries(comp) -> Dict[str, Any]:
    """Peak, peak date and final value of every ``_total`` compartment column."""
    variables = {}
    for var in [c for c in comp.columns if c.endswith("_total")]:
        pivoted = comp.pivot(index="date", columns="sim_id", values=var)
        stats = _per_simulation_stats(pivoted)
        variables[var] = {
            "peak": _value_quantiles(stats["peak"]),
            "peak_date": _date_quantiles(stats["peak_date"]),
            "final": _value_quantiles(stats["final"]),
        }
    return variables


def _attack_rate(comp) -> Optional[Dict[str, Any]]:
    attack = _legacy_attack_rate_samples(comp)
    if attack is None:
        return None
    result = _value_quantiles(attack * 100)
    result["units"] = "percent"
    return result


def build_summary(bundle: str, manifest: Optional[Dict] = None) -> Dict[str, Any]:
    """Compute the content of ``summary.json`` for a bundle, without writing it.

    Args:
        bundle: Path to the ``.epx`` bundle directory.
        manifest: The bundle's manifest, if already loaded.

    Returns:
        A JSON-serializable dict: per-variable quantiles (5%, 50%, 95%) of the
        peak value, peak date and final value of every ``_total`` compartment
        column, the attack rate, and, for calibration bundles, posterior
        statistics.
    """
    bundle = Path(bundle)
    if manifest is None:
        manifest = load_bundle(str(bundle))
    files = manifest.get("files", {})

    summary: Dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "bundle_id": manifest.get("id"),
        "type": manifest.get("type"),
        "quantiles": [str(q) for q in SUMMARY_QUANTILES],
    }

    simulation = manifest.get("simulation") or {}
    calibration = manifest.get("calibration") or {}
    if "n_simulations" in simulation:
        summary["n_simulations"] = simulation["n_simulations"]
    elif "n_particles" in calibration:
        summary["n_particles"] = calibration["n_particles"]

    if "compartments" in files and (bundle / files["compartments"]["path"]).exists():
        comp = load_bundle_dataframe(str(bundle), "compartments")
        summary["variables"] = _variable_summaries(comp)
        summary["attack_rate"] = _attack_rate(comp)
        summary["attack_rate_method"] = "legacy"

    if (bundle / "posterior.parquet").exists():
        summary["posterior"] = _cmd_posterior(str(bundle))

    return summary


def write_summary(bundle: str, manifest: Optional[Dict] = None) -> Dict[str, Any]:
    """Write ``summary.json`` into a bundle and register it in the manifest.

    Args:
        bundle: Path to the ``.epx`` bundle directory.
        manifest: The bundle's manifest, if already loaded.

    Returns:
        The updated manifest (also written back to ``manifest.json``).
    """
    bundle = Path(bundle)
    if manifest is None:
        manifest = load_bundle(str(bundle))
    summary = build_summary(str(bundle), manifest)
    with open(bundle / SUMMARY_FILENAME, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpySafeEncoder)

    manifest.setdefault("files", {})["summary"] = {
        "path": SUMMARY_FILENAME,
        "description": "Headline results (peaks, final values, attack rate, "
        "posterior statistics). A cache of what `epydemix inspect` computes "
        "from the Parquet files.",
        "schema_version": SUMMARY_SCHEMA_VERSION,
    }
    with open(bundle / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, cls=NumpySafeEncoder)
    return manifest


def try_write_summary(bundle: str, manifest: Dict) -> Dict[str, Any]:
    """:func:`write_summary`, but a failure only warns: the bundle stays valid."""
    try:
        return write_summary(bundle, manifest)
    except Exception as e:  # pragma: no cover - defensive
        print(f"Warning: could not write {SUMMARY_FILENAME}: {e}", file=sys.stderr)
        return manifest


def load_summary(bundle: str) -> Dict[str, Any]:
    """Read a bundle's ``summary.json``, or compute it if the bundle has none."""
    path = Path(bundle) / SUMMARY_FILENAME
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return build_summary(bundle)
