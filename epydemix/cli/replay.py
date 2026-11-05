"""``epydemix replay``: re-run a bundle and compare its results with the original."""

import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from ..io.bundle import load_bundle
from ..io.summary import load_summary

_DATE_LENGTH = len("2026-01-01")


def _is_date(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != _DATE_LENGTH:
        return False
    try:
        pd.Timestamp(value)
    except (ValueError, TypeError):
        return False
    return True


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def compare_summaries(
    original: Dict[str, Any], replayed: Dict[str, Any], n_days: Optional[int] = None
) -> Dict[str, Any]:
    """Largest relative difference between the numbers of two summaries.

    Compares the unrounded values (``raw``) when both summaries have them.
    Numbers differ by ``|a - b| / max(|a|, |b|)``; dates by their distance in
    days over the length of the simulation (``n_days``), or 1 if unknown.
    A number present in one summary only counts as a difference of 1.

    Returns:
        ``{"max_rel_diff", "worst", "n_compared"}``.
    """
    if "raw" in original and "raw" in replayed:
        a, b = original["raw"], replayed["raw"]
    else:
        skip = {"bundle_id", "raw", "mc_se", "precision"}
        a = {k: v for k, v in original.items() if k not in skip}
        b = {k: v for k, v in replayed.items() if k not in skip}

    state = {"max": 0.0, "worst": None, "n": 0}

    def record(diff: float, path: str) -> None:
        state["n"] += 1
        if diff > state["max"]:
            state["max"], state["worst"] = diff, path

    def walk(x: Any, y: Any, path: str) -> None:
        if isinstance(x, dict) and isinstance(y, dict):
            for key in sorted(set(x) | set(y), key=str):
                sub = f"{path}.{key}" if path else str(key)
                if key not in x or key not in y:
                    present = x.get(key, y.get(key))
                    if _numeric(present) or _is_date(present):
                        record(1.0, sub)
                    continue
                walk(x[key], y[key], sub)
        elif isinstance(x, list) and isinstance(y, list):
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f"{path}[{i}]")
            if len(x) != len(y):
                record(1.0, f"{path}[length]")
        elif _numeric(x) and _numeric(y):
            scale = max(abs(x), abs(y))
            diff = 0.0 if scale == 0 else abs(x - y) / scale
            if math.isnan(diff):
                diff = 0.0 if (math.isnan(x) and math.isnan(y)) else 1.0
            record(diff, path)
        elif _is_date(x) and _is_date(y):
            days = abs((pd.Timestamp(x) - pd.Timestamp(y)).days)
            record(days / n_days if n_days else float(days > 0), path)

    walk(a, b, "")
    return {
        "max_rel_diff": state["max"],
        "worst": state["worst"],
        "n_compared": state["n"],
    }


def _load_stored_config(bundle: Path) -> Dict[str, Any]:
    import json

    import yaml

    if (bundle / "config.yaml").exists():
        with open(bundle / "config.yaml") as f:
            return yaml.safe_load(f) or {}
    if (bundle / "config.json").exists():
        with open(bundle / "config.json") as f:
            return json.load(f)
    raise FileNotFoundError(f"No config.yaml or config.json in {bundle}")


def _config_dir(prov: Dict[str, Any], bundle: Path) -> Path:
    """Where relative paths in the stored config (data files) resolve."""
    path = prov.get("config_path")
    if path and Path(path).parent.exists():
        return Path(path).parent
    return bundle


def _simulation_days(manifest: Dict[str, Any]) -> Optional[int]:
    sim = manifest.get("simulation") or {}
    if sim.get("start_date") and sim.get("end_date"):
        return (pd.Timestamp(sim["end_date"]) - pd.Timestamp(sim["start_date"])).days
    return None


def rerun(bundle, output) -> Tuple[str, Dict[str, Any]]:
    """Re-execute the command that wrote ``bundle`` into ``output``."""
    from .execute import calibrate_to_bundle, project_to_bundle, run_to_bundle

    bundle = Path(bundle)
    manifest = load_bundle(str(bundle))
    prov = manifest.get("provenance") or {}
    command = prov.get("command")
    config = _load_stored_config(bundle)
    config_dir = _config_dir(prov, bundle)
    extra = {"replay_of": str(bundle.resolve())}
    for key in ("sweep", "point_key", "parent_seed"):
        if key in prov:
            extra[key] = prov[key]

    if command in ("run", "sweep"):
        new = run_to_bundle(
            config, output, config_dir, command=command, extra_provenance=extra
        )
    elif command == "calibrate":
        new = calibrate_to_bundle(
            config, output, config_dir, command=command, extra_provenance=extra
        )
    elif command == "project":
        parent = prov.get("parent_bundle")
        if not parent or not Path(parent).exists():
            raise FileNotFoundError(
                f"The calibration bundle this projection sampled from ({parent}) "
                "is not available, so it cannot be replayed"
            )
        new = project_to_bundle(
            config, parent, output, config_dir, command=command, extra_provenance=extra
        )
    else:
        raise ValueError(
            f"Cannot replay a bundle written by {command!r}; replay supports "
            "bundles written by run, calibrate, project and sweep points"
        )
    return command, new


def replay_bundle(bundle, output=None, tol: float = 0.05) -> Dict[str, Any]:
    """Re-run a bundle with its stored config and seed; compare the summaries.

    Args:
        bundle: Path to the bundle.
        output: Where to keep the re-run bundle; a temporary directory
            (removed afterwards) by default.
        tol: Largest relative difference that counts as reproduced.
    """
    bundle = Path(bundle)
    manifest = load_bundle(str(bundle))
    tmpdir = None
    if output is None:
        tmpdir = tempfile.mkdtemp(prefix="epx-replay-")
        target = Path(tmpdir) / "replay.epx"
    else:
        target = Path(output)
    try:
        command, _ = rerun(bundle, target)
        diff = compare_summaries(
            load_summary(str(bundle)),
            load_summary(str(target)),
            n_days=_simulation_days(manifest),
        )
    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)
    return {
        "bundle": str(bundle.resolve()),
        "command": command,
        "seed": (manifest.get("provenance") or {}).get("seed"),
        "replayed_to": None if tmpdir is not None else str(target.resolve()),
        **diff,
        "tol": tol,
        "reproduced": diff["max_rel_diff"] <= tol,
    }
