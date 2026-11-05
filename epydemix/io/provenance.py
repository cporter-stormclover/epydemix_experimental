"""Content-addressed provenance for .epx bundles.

A bundle records enough to detect any change in what produced it: the hash
of every file it contains, the population data it was built from (repository,
version tag and per-file hashes), the code (epydemix version, git commit,
uncommitted changes), the environment (Python and every installed package),
the seed, the version of the agent contract, and, when set, the agent that
drove the run (``EPX_AGENT_RUNTIME``, ``EPX_AGENT_MODEL``).

:func:`verify_bundle` recomputes the hashes and reports any drift;
``epydemix replay`` re-runs a bundle and compares the results.
"""

import copy
import functools
import hashlib
import json
import os
import platform
import subprocess
import sys
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from .json_utils import NumpySafeEncoder

AGENT_ENV_VARS = ("EPX_AGENT_RUNTIME", "EPX_AGENT_MODEL")
POPULATION_DATA_REPO = "epistorm/epydemix-data"
DEFAULT_LAYERS = ["school", "work", "home", "community"]

# Files a bundle may gain after it is written, which do not change its content.
_UNSEALED = ("manifest.json",)
_UNSEALED_DIRS = ("figures",)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_sha256(config: Dict[str, Any]) -> str:
    """Hash of a config's content (key order and formatting do not matter)."""
    blob = json.dumps(config, sort_keys=True, cls=NumpySafeEncoder)
    return hashlib.sha256(blob.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Bundle contents
# ---------------------------------------------------------------------------


def bundle_file_hashes(bundle) -> Dict[str, str]:
    """sha256 of every file in a bundle, except manifests and figures/."""
    bundle = Path(bundle)
    hashes = {}
    for path in sorted(bundle.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(bundle).as_posix()
        parts = rel.split("/")
        # Manifests and figures may change after a bundle is written; a
        # sweep's points are bundles with seals of their own.
        if parts[-1] in _UNSEALED or any(d in _UNSEALED_DIRS for d in parts[:-1]):
            continue
        hashes[rel] = sha256_file(path)
    return hashes


def manifest_sha256(manifest: Dict[str, Any]) -> str:
    """Hash of a manifest's content, leaving out what may change afterwards.

    ``figures`` is excluded (``add_figure_to_manifest`` updates it), and so
    is the seal itself.
    """
    content = copy.deepcopy(manifest)
    content.pop("figures", None)
    prov = content.get("provenance") or {}
    prov.pop("bundle_files", None)
    prov.pop("manifest_sha256", None)
    return config_sha256(content)


def seal_bundle(bundle, manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Record the hash of every bundle file and of the manifest itself.

    Call once the bundle is complete (data, config, summary.json). Writes the
    updated manifest back and returns it.
    """
    bundle = Path(bundle)
    prov = manifest.setdefault("provenance", {})
    prov["bundle_files"] = bundle_file_hashes(bundle)
    prov["manifest_sha256"] = manifest_sha256(manifest)
    with open(bundle / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, cls=NumpySafeEncoder)
    return manifest


# ---------------------------------------------------------------------------
# Population data
# ---------------------------------------------------------------------------


def _fetch(url: str) -> bytes:
    """Download a file (honours the usual proxy environment variables)."""
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def population_data_block(
    population_name: str,
    data_version: str,
    layers: Optional[List[str]] = None,
    contacts_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Repository, version and per-file hashes of a named population's data.

    The files are the ones ``load_epydemix_population`` reads for this
    population. If they cannot be fetched (e.g. offline), the block records
    the error instead of the hashes.
    """
    import pandas as pd

    from ..population.population import (
        EPYDEMIX_DATA_BASE_URL,
        _get_contact_matrix_path,
        _get_demographic_path,
        _get_locations_path,
    )

    base = f"{EPYDEMIX_DATA_BASE_URL}/{data_version}/"
    block: Dict[str, Any] = {
        "repo": POPULATION_DATA_REPO,
        "version": data_version,
        "population": population_name,
        "layers": list(layers or DEFAULT_LAYERS),
    }
    try:
        locations_url = _get_locations_path(base, "age", is_remote=True)
        locations = _fetch(locations_url)
        if contacts_source is None:
            table = pd.read_csv(BytesIO(locations))
            match = table.loc[
                table["location"] == population_name, "primary_contact_source"
            ]
            contacts_source = str(match.iloc[0]) if len(match) else None
        block["contacts_source"] = contacts_source
        urls = [
            locations_url,
            _get_demographic_path(base, "age", population_name, is_remote=True),
        ] + [
            _get_contact_matrix_path(
                base, "age", population_name, contacts_source, layer, is_remote=True
            )
            for layer in block["layers"]
        ]
        files = {}
        for url in urls:
            data = locations if url == locations_url else _fetch(url)
            files[url[len(base) :]] = sha256_bytes(data)
        block["files"] = files
    except Exception as e:
        block["files"] = None
        block["error"] = f"could not hash the population data: {e}"
    return block


# ---------------------------------------------------------------------------
# Code and environment
# ---------------------------------------------------------------------------


def _git(args: List[str], cwd: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def code_block() -> Dict[str, Any]:
    """epydemix version, and the git commit it was imported from, if any."""
    from .manifest import _get_version

    block: Dict[str, Any] = {"epydemix": _get_version(), "commit": None, "dirty": None}
    package_root = Path(__file__).resolve().parents[2]
    toplevel = _git(["rev-parse", "--show-toplevel"], package_root)
    if toplevel and Path(toplevel).resolve() == package_root:
        block["commit"] = _git(["rev-parse", "HEAD"], package_root)
        status = _git(["status", "--porcelain", "--untracked-files=no"], package_root)
        block["dirty"] = bool(status) if status is not None else None
        return block
    # An installed wheel from a VCS URL records its commit (PEP 610).
    try:
        from importlib.metadata import distribution

        direct = distribution("epydemix").read_text("direct_url.json")
        if direct:
            vcs = json.loads(direct).get("vcs_info") or {}
            block["commit"] = vcs.get("commit_id")
    except Exception:
        pass
    return block


def environment_block() -> Dict[str, Any]:
    """Python version, platform and every installed distribution."""
    return copy.deepcopy(_environment())


@functools.lru_cache(maxsize=1)
def _environment() -> Dict[str, Any]:
    # Cached per process: a sweep seals many bundles in one invocation.
    from importlib.metadata import distributions

    packages = {}
    for dist in distributions():
        name = dist.metadata.get("Name")
        if name:
            packages[name.lower()] = dist.version
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": sys.platform,
        "packages": dict(sorted(packages.items())),
    }


def read_contract_version() -> Optional[str]:
    """Version (hash) of the AGENT.md contract this epydemix ships with."""
    try:
        from .._contract import CONTRACT_VERSION

        return CONTRACT_VERSION
    except ImportError:
        pass
    agent_md = Path(__file__).resolve().parents[2] / "AGENT.md"
    if agent_md.exists():
        return "sha256:" + sha256_file(agent_md)
    return None


def agent_block() -> Dict[str, str]:
    return {k: os.environ[k] for k in AGENT_ENV_VARS if os.environ.get(k)}


def provenance_block(
    command: str,
    config_path: Optional[str],
    used_config: Optional[Dict[str, Any]],
    population: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The provenance a CLI command records in a bundle's manifest.

    Args:
        command: ``run``, ``calibrate``, ``project`` or ``sweep``.
        config_path: The config file given on the command line.
        used_config: The resolved config the bundle stores (seed included).
        population: :func:`population_data_block` of a named population.
    """
    prov: Dict[str, Any] = {"command": command}
    if config_path is not None:
        prov["config_path"] = str(Path(config_path).resolve())
        try:
            prov["config_file_sha256"] = sha256_file(config_path)
        except OSError:
            pass
    if used_config:
        prov["config_sha256"] = config_sha256(used_config)
        seed = (used_config.get("simulation") or {}).get("seed")
        if seed is not None:
            prov["seed"] = seed
    if population is not None:
        prov["population_data"] = population
    prov["code"] = code_block()
    prov["environment"] = environment_block()
    prov["contract_version"] = read_contract_version()
    agent = agent_block()
    if agent:
        prov["agent"] = agent
    return prov


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_bundle(bundle, check_population: bool = True) -> Dict[str, Any]:
    """Recompute a bundle's hashes and report what drifted.

    Drift (``verified: false``) is a change to the bundle's sealed files, its
    manifest, its stored config, or the population data it was built from.
    Differences between the environment that produced the bundle and the one
    verifying it are reported, but are not drift.
    """
    from .bundle import load_bundle

    bundle = Path(bundle)
    manifest = load_bundle(str(bundle))
    prov = manifest.get("provenance") or {}
    report: Dict[str, Any] = {"bundle": str(bundle.resolve()), "drift": []}
    drift = report["drift"]

    recorded = prov.get("bundle_files")
    if recorded is None:
        report["verified"] = False
        report["reason"] = (
            "the bundle records no content hashes (written before epydemix "
            "sealed bundles, or not by the CLI); nothing to verify against"
        )
        return report

    current = bundle_file_hashes(bundle)
    files = {
        "modified": sorted(
            f for f in recorded if f in current and current[f] != recorded[f]
        ),
        "missing": sorted(f for f in recorded if f not in current),
        "added": sorted(f for f in current if f not in recorded),
    }
    files["unchanged"] = len(recorded) - len(files["modified"]) - len(files["missing"])
    report["files"] = files
    for kind in ("modified", "missing", "added"):
        for f in files[kind]:
            drift.append(f"{f}: {kind}")

    if prov.get("manifest_sha256") != manifest_sha256(manifest):
        drift.append("manifest.json: modified")
        report["manifest"] = "modified"
    else:
        report["manifest"] = "unchanged"

    config_file = bundle / "config.yaml"
    if config_file.exists() and "config_sha256" in prov:
        try:
            import yaml

            with open(config_file) as f:
                stored = yaml.safe_load(f) or {}
            same = config_sha256(stored) == prov["config_sha256"]
        except Exception:
            same = False
        report["config"] = "unchanged" if same else "modified"
        if not same:
            drift.append("config.yaml: content differs from config_sha256")

    population = prov.get("population_data")
    if population and population.get("files") and check_population:
        fresh = population_data_block(
            population["population"],
            population["version"],
            population.get("layers"),
            population.get("contacts_source"),
        )
        if fresh.get("files") is None:
            report["population_data"] = {
                "status": "unchecked",
                "error": fresh.get("error"),
            }
        else:
            changed = sorted(
                f for f, h in population["files"].items() if fresh["files"].get(f) != h
            )
            report["population_data"] = {
                "status": "drift" if changed else "unchanged",
                "modified": changed,
            }
            drift.extend(f"population data {f}: modified" for f in changed)
    elif population:
        report["population_data"] = {
            "status": "unchecked",
            "error": population.get("error") or "not checked (--offline)",
        }

    report["environment_differences"] = _environment_differences(prov)
    report["verified"] = not drift
    return report


def _environment_differences(prov: Dict[str, Any]) -> Dict[str, Any]:
    differences: Dict[str, Any] = {}
    code = prov.get("code") or {}
    now = code_block()
    for key in ("epydemix", "commit"):
        if code.get(key) is not None and code.get(key) != now.get(key):
            differences[key] = {"recorded": code.get(key), "now": now.get(key)}
    if code.get("dirty"):
        differences["dirty_at_write"] = True
    env = prov.get("environment") or {}
    current = environment_block()
    if env.get("python") and env["python"] != current["python"]:
        differences["python"] = {"recorded": env["python"], "now": current["python"]}
    recorded_pkgs = env.get("packages") or {}
    changed = {
        name: {"recorded": version, "now": current["packages"].get(name)}
        for name, version in recorded_pkgs.items()
        if current["packages"].get(name) != version
    }
    if changed:
        differences["packages"] = changed
    return differences
