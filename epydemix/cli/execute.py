"""Turn a resolved config into a finished bundle.

Shared by ``run``, ``calibrate``, ``project``, ``replay`` and ``sweep``: build
the provenance, save the results, write ``summary.json``, then seal the
bundle with the hashes of its files. Every function is a pure function of its
arguments and the files on disk.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from ..io.bundle import save_bundle
from ..io.provenance import population_data_block, provenance_block, seal_bundle
from ..io.summary import try_write_summary
from .config import DEFAULT_DATA_VERSION


def population_provenance(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Population data block for a config with a named population."""
    pop = config.get("population") or {}
    name = pop.get("name")
    if not name or name == "default":
        return None
    return population_data_block(
        name,
        pop.get("data_version", DEFAULT_DATA_VERSION),
        pop.get("contact_layers"),
    )


def finish_bundle(
    results,
    output,
    used_config: Dict[str, Any],
    command: str,
    config_path: Optional[str] = None,
    extra_provenance: Optional[Dict[str, Any]] = None,
    config_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Save results as a bundle with its summary and content hashes."""
    prov = provenance_block(
        command, config_path, used_config, population_provenance(used_config)
    )
    if extra_provenance:
        prov.update(extra_provenance)
    manifest = save_bundle(results, str(output), config=used_config, provenance=prov)
    if used_config.get("derived"):
        from .config import resolve_derived

        manifest["derived"] = resolve_derived(used_config, config_dir)
    if (used_config.get("population") or {}).get("age_groups"):
        from .config import age_group_mapping

        mapping = age_group_mapping(used_config)
        if mapping and isinstance(manifest.get("population"), dict):
            manifest["population"]["age_group_mapping"] = mapping
    manifest = try_write_summary(str(output), manifest)
    return seal_bundle(output, manifest)


def run_to_bundle(
    config: Dict[str, Any],
    output,
    config_dir: Optional[Path] = None,
    config_path: Optional[str] = None,
    command: str = "run",
    extra_provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from .config import run_from_config

    results, used = run_from_config(config, config_dir=config_dir)
    return finish_bundle(
        results, output, used, command, config_path, extra_provenance, config_dir
    )


def calibrate_to_bundle(
    config: Dict[str, Any],
    output,
    config_dir: Optional[Path] = None,
    config_path: Optional[str] = None,
    command: str = "calibrate",
    extra_provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from .config import calibrate_from_config

    results, used = calibrate_from_config(config, config_dir=config_dir)
    return finish_bundle(
        results, output, used, command, config_path, extra_provenance, config_dir
    )


def project_to_bundle(
    config: Dict[str, Any],
    calibration_bundle,
    output,
    config_dir: Optional[Path] = None,
    config_path: Optional[str] = None,
    command: str = "project",
    extra_provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from .config import project_from_config

    results, used = project_from_config(
        config, str(calibration_bundle), config_dir=config_dir
    )
    extra = {"parent_bundle": str(Path(calibration_bundle).resolve())}
    extra.update(extra_provenance or {})
    return finish_bundle(results, output, used, command, config_path, extra, config_dir)
