"""Click CLI for epydemix — the primary interface for LLM agents.

Usage::

    epydemix run config.yaml --output results.epx
    epydemix inspect results.epx quantiles --variables I_total --quantiles 0.05,0.5,0.95
    epydemix schema SEIR
    epydemix validate config.yaml
    epydemix models
    epydemix defaults
"""

import json
import sys
from pathlib import Path

import click

from ..io.json_utils import NumpySafeEncoder


def _print_json(data, precision=None):
    """Print a dict as JSON to stdout."""
    if precision is not None:
        from ..io.json_utils import _round_recursive

        data = _round_recursive(data, precision)
    click.echo(json.dumps(data, indent=2, cls=NumpySafeEncoder))


def _error_json(code, message, details=None):
    """Print a structured error to stderr and exit."""
    err = {"error": True, "code": code, "message": message}
    if details:
        err["details"] = details
    click.echo(json.dumps(err, indent=2), err=True)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(package_name="epydemix")
def cli():
    """Epydemix: epidemic modeling, simulation, and calibration.

    Agent-friendly CLI for driving epydemix from configs and inspecting results.
    Every command prints structured JSON to stdout and diagnostics/warnings to
    stderr, so stdout is always safe to parse or pipe.

    Run `epydemix COMMAND --help` for a given command's options and examples.
    """
    pass


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--output", "-o", default="results.epx", help="Path for the output .epx bundle."
)
def run(config_path, output):
    """Run a simulation from a YAML/JSON config file.

    Writes an .epx bundle and prints the manifest to stdout.
    """
    from .config import load_config, validate_config
    from .execute import run_to_bundle

    try:
        config = load_config(config_path)
    except Exception as e:
        _error_json("CONFIG_LOAD_ERROR", str(e))
    if "sweep" in config:
        _error_json(
            "INVALID_CONFIG",
            "The config has a 'sweep' block; run it with `epydemix sweep`.",
        )

    validation = validate_config(config)
    if not validation["valid"]:
        _error_json(
            "INVALID_CONFIG", "Config validation failed.", details=validation["errors"]
        )

    # Print warnings to stderr
    for w in validation.get("warnings", []):
        click.echo(f"Warning: {w}", err=True)

    try:
        config_dir = Path(config_path).parent
        manifest = run_to_bundle(
            config, output, config_dir=config_dir, config_path=config_path
        )
        click.echo(f"Bundle saved to: {output}", err=True)
        _print_json(manifest)
    except Exception as e:
        _error_json("RUNTIME_ERROR", str(e))


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--output", "-o", default="calibration.epx", help="Path for the output .epx bundle."
)
def calibrate(config_path, output):
    """Run ABC calibration from a YAML/JSON config file.

    The config must include a 'calibration' section with priors,
    observed data, and strategy settings.  See AGENT.md for full
    config reference.

    Writes an .epx calibration bundle and prints the manifest to stdout.
    """
    from .config import load_config, validate_calibration_config
    from .execute import calibrate_to_bundle

    try:
        config = load_config(config_path)
    except Exception as e:
        _error_json("CONFIG_LOAD_ERROR", str(e))

    validation = validate_calibration_config(config)
    if not validation["valid"]:
        _error_json(
            "INVALID_CONFIG",
            "Calibration config validation failed.",
            details=validation["errors"],
        )

    for w in validation.get("warnings", []):
        click.echo(f"Warning: {w}", err=True)

    try:
        config_dir = Path(config_path).parent
        manifest = calibrate_to_bundle(
            config, output, config_dir=config_dir, config_path=config_path
        )
        click.echo(f"Calibration bundle saved to: {output}", err=True)
        _print_json(manifest)
    except Exception as e:
        _error_json("RUNTIME_ERROR", str(e))


# ---------------------------------------------------------------------------
# project
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("calibration_bundle", type=click.Path(exists=True))
@click.option(
    "--config",
    "-c",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Projection overlay config (inherits from calibration bundle's config).",
)
@click.option(
    "--output", "-o", default="projection.epx", help="Path for the output .epx bundle."
)
def project(calibration_bundle, config_path, output):
    """Run forward projections from a calibration posterior.

    Samples parameter sets from the calibration posterior (weighted),
    runs forward simulations with the (possibly overridden) config,
    and saves results as a standard simulation bundle.

    \b
    The calibration bundle's stored config is always the implicit base.
    A projection overlay only needs to specify what changes — no ``base:``
    key is required:

    \b
      simulation:
        end_date: "2025-06-30"    # extend beyond calibration period
      overrides:
        - parameter: transmission_rate
          start_date: "2025-04-01"
          end_date: "2025-06-30"
          value: 0.15
      projection:
        n_simulations: 200

    If --config is omitted, the calibration bundle's own config is used
    (i.e. replay with posterior samples over the same period).

    \b
    Examples:
      epydemix project calibration.epx -o projection.epx
      epydemix project calibration.epx -c projection.yaml -o projection.epx
    """
    from .config import load_config, validate_projection_config
    from .execute import project_to_bundle

    # Build the effective config
    try:
        import yaml

        from .config import _deep_merge

        bundle_p = Path(calibration_bundle)
        cfg_yaml = bundle_p / "config.yaml"
        cfg_json = bundle_p / "config.json"
        if cfg_yaml.exists():
            with open(cfg_yaml) as f:
                bundle_config = yaml.safe_load(f) or {}
        elif cfg_json.exists():
            with open(cfg_json) as f:
                bundle_config = json.load(f)
        else:
            _error_json(
                "CONFIG_LOAD_ERROR",
                "No config.yaml or config.json found in calibration bundle.",
            )

        # Strip the calibration section — not applicable to projections.
        bundle_config.pop("calibration", None)

        if config_path is not None:
            # The overlay only needs to specify what changes (end_date,
            # overrides, projection settings, …).  The calibration bundle's
            # config is used as the implicit base, so ``base:`` in the overlay
            # is not required.  If the overlay does carry a ``base:`` key it is
            # still resolved normally — the bundle config is simply merged on
            # top of whatever that chain produces.
            overlay = load_config(config_path)
            config = _deep_merge(bundle_config, overlay)
        else:
            # No overlay — replay calibration period with posterior samples.
            config = bundle_config
    except Exception as e:
        _error_json("CONFIG_LOAD_ERROR", str(e))

    validation = validate_projection_config(config, calibration_bundle)
    if not validation["valid"]:
        _error_json(
            "INVALID_CONFIG",
            "Projection config validation failed.",
            details=validation["errors"],
        )

    for w in validation.get("warnings", []):
        click.echo(f"Warning: {w}", err=True)

    try:
        proj_config_dir = Path(config_path).parent if config_path is not None else None
        manifest = project_to_bundle(
            config,
            calibration_bundle,
            output,
            config_dir=proj_config_dir,
            config_path=config_path,
        )
        click.echo(f"Projection bundle saved to: {output}", err=True)
        _print_json(manifest)
    except Exception as e:
        _error_json("RUNTIME_ERROR", str(e))


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--prompt",
    "prompt_path",
    default=None,
    type=click.Path(exists=True),
    help="File with the prompt the config answers. Parameters set to a number "
    "written in it count as read (prompt_literal) and need no assumptions entry.",
)
@click.option(
    "--upgrade",
    is_flag=True,
    default=False,
    help="Rewrite the file's schema_version to the current one, in place.",
)
def validate(config_path, prompt_path, upgrade):
    """Validate a config file without running it.

    Automatically detects calibration configs (those with a 'calibration'
    section) and validates the calibration-specific fields as well.

    Warns about every parameter set to a non-default value that the
    `assumptions` block does not declare.

    Prints validation result as JSON to stdout.
    """
    from . import schema_version
    from .assumptions import prompt_literals
    from .config import load_config, validate_calibration_config, validate_config

    upgraded = None
    if upgrade:
        try:
            upgraded = schema_version.upgrade_file(config_path)
        except Exception as e:
            _error_json("CONFIG_LOAD_ERROR", f"Could not upgrade the config: {e}")

    try:
        config = load_config(config_path)
    except Exception as e:
        _error_json("CONFIG_LOAD_ERROR", str(e))

    literals = None
    if prompt_path is not None:
        with open(prompt_path) as f:
            literals = prompt_literals(f.read())

    if "sweep" in config:
        result = _validate_sweep(config)
    elif "calibration" in config:
        result = validate_calibration_config(config, prompt_literals=literals)
    else:
        result = validate_config(config, prompt_literals=literals)
    if upgraded is not None:
        result["upgraded"] = upgraded
    _print_json(result)

    if not result["valid"]:
        sys.exit(1)


def _validate_sweep(config):
    """Validate every point of a sweep config (nothing runs)."""
    from .config import resolve_seed
    from .sweep import SweepError, plan

    config = dict(config)
    seed = resolve_seed(config.get("simulation") or {})
    try:
        points = plan(config, seed)
    except SweepError as e:
        result = {"valid": False, "errors": e.errors, "warnings": []}
        if e.point:
            result["point"] = e.point
        return result
    warnings = []
    for point in points:
        for w in point["warnings"]:
            if w not in warnings:
                warnings.append(w)
    return {"valid": True, "errors": [], "warnings": warnings, "n_points": len(points)}


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--output", "-o", default="sweep.epx", help="Path for the output sweep bundle."
)
@click.option(
    "--workers",
    type=int,
    default=1,
    show_default=True,
    help="Points run in parallel (separate processes).",
)
def sweep(config_path, output, workers):
    """Run every point of a config's `sweep` block.

    Validates all points before running any, gives each its own seed (from
    the sweep seed and the point key), writes each as a complete bundle under
    <output>/points/<key>.epx/, and aggregates their summary.json files into
    <output>/summary.json (one row per point, with deltas against
    sweep.reference if set). Prints that summary.

    \b
      sweep:
        name: closure_timing
        metrics: [attack_rate, Infected_total.peak]
        over:
          - path: interventions.0.start_date
            name: start_date
            range: {start: "2026-12-01", stop: "2027-04-01", step: 7d}
        derive:
          interventions.0.end_date: "{{ start_date + 21d }}"

    \b
    Examples:
      epydemix sweep closure.yaml -o closure.epx --workers 4
      epydemix compare closure.epx -m attack_rate
    """
    from .config import load_config
    from .sweep import SweepError, run_sweep

    try:
        config = load_config(config_path)
    except Exception as e:
        _error_json("CONFIG_LOAD_ERROR", str(e))
    if "sweep" not in config:
        _error_json("INVALID_CONFIG", "The config has no 'sweep' block.")
    if workers < 1:
        _error_json("INVALID_CONFIG", "--workers must be at least 1.")

    try:
        summary = run_sweep(
            config,
            output,
            config_dir=Path(config_path).parent,
            config_path=config_path,
            workers=workers,
        )
    except SweepError as e:
        details = list(e.errors)
        message = str(e) if e.point is None else f"Sweep point {e.point} is invalid."
        err = {"error": True, "code": "INVALID_CONFIG", "message": message}
        if e.point:
            err["point"] = e.point
        err["details"] = details
        click.echo(json.dumps(err, indent=2), err=True)
        sys.exit(1)
    except Exception as e:
        _error_json("RUNTIME_ERROR", str(e))
    click.echo(f"Sweep bundle saved to: {output}", err=True)
    _print_json({"bundle": str(Path(output).resolve()), **summary})


# ---------------------------------------------------------------------------
# derive
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--raw",
    is_flag=True,
    default=False,
    help="Print full-precision values (no rounding).",
)
def derive(config_path, raw):
    """Resolve a config's derived parameters and print them.

    \b
      derived:
        transmission_rate: {from: R0, value: 2.5}
    sets transmission_rate = R0 * gamma / rho(C): rho(C) is the spectral
    radius of the population's contact matrix over the config's layers and
    gamma the model's effective removal rate, from its next-generation
    matrix (the recovery rate for SEIR). Values are rounded to the precision
    of their inputs; --raw prints them in full.

    \b
    Examples:
      epydemix derive config.yaml
      epydemix derive config.yaml --raw
    """
    from ..io.precision import format_parameter, round_significant
    from .config import (
        R0_PRECISION,
        _build_predefined,
        load_config,
        resolve_derived,
        validate_calibration_config,
        validate_config,
    )

    try:
        config = load_config(config_path)
    except Exception as e:
        _error_json("CONFIG_LOAD_ERROR", str(e))
    if not config.get("derived"):
        _error_json("INVALID_CONFIG", "The config has no 'derived' block.")
    validation = (
        validate_calibration_config(config)
        if "calibration" in config
        else validate_config(config)
    )
    if not validation["valid"]:
        _error_json(
            "INVALID_CONFIG", "Config validation failed.", details=validation["errors"]
        )
    try:
        resolved = resolve_derived(config, config_dir=Path(config_path).parent)
    except Exception as e:
        _error_json("RUNTIME_ERROR", str(e))

    if not raw:
        model_cfg = config.get("model") or {}
        model = _build_predefined(model_cfg.get("type", "custom"), model_cfg)
        registry = model.parameter_registry if model is not None else None
        precision = {"R0": R0_PRECISION}
        for name, details in resolved.items():
            gamma_inputs = [i for i in details["inputs"] if i != "R0"]
            details["value"] = format_parameter(
                name, details["value"], registry, details["inputs"], precision
            )
            if details["gamma"] is not None:
                details["gamma"] = format_parameter(
                    "gamma", details["gamma"], registry, gamma_inputs or None
                )
            details["rho"] = round_significant(details["rho"], 4)
    _print_json({"derived": resolved, "warnings": validation["warnings"]})


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


@cli.command(name="inspect")
@click.argument("bundle_path", type=click.Path(exists=True))
@click.argument("command")
@click.option(
    "--variables", "-v", default=None, help="Comma-separated list of variables."
)
@click.option(
    "--quantiles",
    "-q",
    default=None,
    help="Comma-separated quantile levels (e.g. 0.05,0.5,0.95).",
)
@click.option("--start", default=None, help="Start date for time slice (ISO format).")
@click.option("--end", default=None, help="End date for time slice (ISO format).")
@click.option(
    "--resample", default=None, help="Temporal resampling frequency (e.g. W, M)."
)
@click.option(
    "--round",
    "precision",
    type=int,
    default=None,
    help="Round every number to this many decimals instead of to its "
    "Monte Carlo error.",
)
@click.option(
    "--raw",
    is_flag=True,
    default=False,
    help="Print full-precision floats (no rounding).",
)
@click.option(
    "--generation",
    type=int,
    default=None,
    help="Calibration generation (for posterior/fit commands).",
)
@click.option(
    "--format",
    "output_format",
    default="json",
    type=click.Choice(["json", "csv", "tsv"]),
    help="Output format.",
)
def inspect_cmd(
    bundle_path,
    command,
    variables,
    quantiles,
    start,
    end,
    resample,
    precision,
    raw,
    generation,
    output_format,
):
    """Inspect an .epx result bundle.

    Numbers are rounded to the first significant digit of their bootstrap
    standard error, and a peak date whose 90% interval spans more than one
    day is reported as that interval (`reported`). Copy them as printed.
    `--raw` prints full precision; `--round N` rounds to N decimals instead.

    \b
    Commands:
      manifest    Return the manifest JSON
      quantiles   Compute quantiles across simulations
      summary     Summary statistics (peak, final state)
      peak        Peak timing and magnitude
      posterior   Calibration posterior summary
      fit         Calibration fit vs. observed data

    \b
    Examples:
      epydemix inspect results.epx manifest
      epydemix inspect results.epx quantiles -v I_total -q 0.05,0.5,0.95
      epydemix inspect results.epx summary -v I_total --start 2020-03-01
      epydemix inspect results.epx peak -v I_total
      epydemix inspect calibration.epx posterior
    """
    from ..io.inspect import inspect_bundle

    # Parse comma-separated options
    vars_list = variables.split(",") if variables else None
    quant_list = [float(q) for q in quantiles.split(",")] if quantiles else None

    kwargs = {
        "variables": vars_list,
        "start": start,
        "end": end,
        "resample": resample,
    }
    if quant_list is not None:
        kwargs["quantiles"] = quant_list
    if generation is not None:
        kwargs["generation"] = generation

    report = "raw" if (raw or precision is not None) else "rounded"
    try:
        result = inspect_bundle(bundle_path, command, report=report, **kwargs)
    except Exception as e:
        _error_json("INSPECT_ERROR", str(e))

    if output_format == "json":
        # manifest returns raw JSON — skip rounding to preserve exact parameter values
        effective_precision = None if command == "manifest" else precision
        _print_json(result, precision=effective_precision)
    elif output_format in ("csv", "tsv"):
        if precision is not None and command != "manifest":
            from ..io.json_utils import _round_recursive

            result = _round_recursive(result, precision)
        _print_tabular(result, output_format)


def _print_tabular(data, fmt):
    """Best-effort conversion of inspect result to CSV/TSV."""
    import pandas as pd

    sep = "," if fmt == "csv" else "\t"

    if "dates" in data:
        # Time-series result — build a DataFrame
        rows = {"date": data["dates"]}
        for key, val in data.items():
            if key == "dates":
                continue
            if isinstance(val, dict):
                for q, values in val.items():
                    rows[f"{key}_{q}"] = values
            elif isinstance(val, list):
                rows[key] = val
        df = pd.DataFrame(rows)
        click.echo(df.to_csv(index=False, sep=sep))
    else:
        # Key-value result — print as flat table
        click.echo(json.dumps(data, indent=2, cls=NumpySafeEncoder))


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("model_name")
@click.option(
    "--format",
    "fmt",
    default="json-schema",
    type=click.Choice(["json-schema", "describe"]),
    help="Output format.",
)
@click.option(
    "--waning-immunity",
    "waning_immunity",
    is_flag=True,
    default=False,
    help="Include the waning-immunity module (adds waning_rate).",
)
@click.option(
    "--vaccination",
    is_flag=True,
    default=False,
    help="Include the vaccination module (adds vaccination_rate, vaccine_efficacy).",
)
@click.option(
    "--outcome",
    default=None,
    type=click.Choice(["deaths", "hospitalization"]),
    help="Include an outcome module "
    "(deaths → mortality_rate; "
    "hospitalization → hospitalization_rate, hospitalization_recovery_rate).",
)
def schema(model_name, fmt, waning_immunity, vaccination, outcome):
    """Show the parameter schema of a predefined model, or of a custom model config.

    For a config file with a `model.custom` block, prints its inline parameter
    specs as JSON Schema and its compartment graph (compartments with roles,
    transitions, and a Mermaid diagram).

    \b
    Examples:
      epydemix schema SEIR
      epydemix schema SEIAR
      epydemix schema SEIR --waning-immunity --outcome deaths
      epydemix schema SIR --format describe
      epydemix schema seirhd.yaml
    """
    from ..model.predefined_models import SUPPORTED_MODELS, load_predefined_model

    if Path(model_name).is_file():
        from ..model import declarative
        from .config import _declarative_registry, load_config

        try:
            config = load_config(model_name)
        except Exception as e:
            _error_json("CONFIG_LOAD_ERROR", str(e))
        model_cfg = config.get("model") or {}
        if not isinstance(model_cfg.get("custom"), dict):
            _error_json(
                "UNKNOWN_MODEL",
                f"{model_name} has no model.custom block; for a predefined model "
                "pass its name (e.g. SEIR)",
            )
        registry = _declarative_registry(model_cfg["custom"])
        if fmt == "describe":
            click.echo(registry.describe())
            click.echo(declarative.graph(model_cfg["custom"])["mermaid"])
            return
        result = registry.to_json_schema(title=f"{Path(model_name).name} parameters")
        result["x-compartment-graph"] = declarative.graph(model_cfg["custom"])
        _print_json(result)
        return

    model_name = model_name.upper()
    if model_name not in SUPPORTED_MODELS:
        _error_json(
            "UNKNOWN_MODEL",
            f"Unknown model '{model_name}'. Available: {SUPPORTED_MODELS}",
        )

    try:
        model = load_predefined_model(
            model_name,
            waning_immunity=waning_immunity,
            vaccination=vaccination,
            outcome=outcome,
        )
    except ValueError as e:
        _error_json("INVALID_MODULE", str(e))

    registry = model.parameter_registry

    if fmt == "json-schema":
        _print_json(registry.to_json_schema(title=f"{model_name} Model Parameters"))
    else:
        click.echo(registry.describe())


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


@cli.command()
def models():
    """List available predefined models."""
    from ..model.predefined_models import SUPPORTED_MODELS

    _print_json({"models": SUPPORTED_MODELS})


# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("disease", required=False)
def defaults(disease):
    """List available disease defaults, or show one in detail.

    \b
    Examples:
      epydemix defaults              # list all
      epydemix defaults covid19      # show covid19 details
    """
    from ..parameters.defaults_loader import get_available_defaults, load_defaults

    if disease is None:
        available = get_available_defaults()
        _print_json({"defaults": available})
    else:
        try:
            d = load_defaults(disease)
            _print_json(d.to_dict())
        except FileNotFoundError as e:
            _error_json("UNKNOWN_DEFAULTS", str(e))


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("bundles", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "--metrics",
    "-m",
    default=None,
    help="Comma-separated metrics (e.g. attack_rate,peak,total_deaths,days_over:500).",
)
@click.option(
    "--variables",
    "-v",
    default=None,
    help="Comma-separated variable names for variable-specific metrics.",
)
@click.option(
    "--names",
    "-n",
    default=None,
    help="Comma-separated scenario names (same order as bundles). "
    "Defaults to bundle directory names.",
)
@click.option(
    "--round",
    "precision",
    type=int,
    default=None,
    help="Round every number to this many decimals instead of to its "
    "Monte Carlo error.",
)
@click.option(
    "--raw",
    is_flag=True,
    default=False,
    help="Print full-precision floats (no rounding).",
)
@click.option(
    "--baseline",
    "-b",
    default=None,
    help="Scenario name to use as baseline for delta computation.",
)
def compare(bundles, metrics, variables, names, precision, raw, baseline):
    """Compare multiple .epx bundles on standard metrics.

    Medians, intervals and deltas are rounded to their bootstrap standard
    error; `--raw` prints full precision, `--round N` rounds to N decimals.

    \b
    Built-in metrics:
      attack_rate    % of population ever infected
      peak           Peak value of a variable (default: first I-like _total)
      peak_date      Date of peak value
      total_deaths   Final value of death compartment
      days_over:N    Days the median of a variable exceeds threshold N
      final_value    Final value of a variable

    \b
    Examples:
      epydemix compare baseline.epx early.epx late.epx
      epydemix compare *.epx -m attack_rate,peak,total_deaths,days_over:500
      epydemix compare baseline.epx early.epx -n Baseline,Early -b Baseline
    """
    from ..io.bundle import load_bundle
    from ..io.inspect import compare_bundles, compare_deltas, round_comparison
    from .assumptions import compare as compare_assumptions

    # A sweep bundle stands for its points.
    expanded = []
    for b in bundles:
        try:
            manifest = load_bundle(b)
        except Exception as e:
            _error_json("COMPARE_ERROR", str(e))
        if manifest.get("type") == "SweepResults":
            expanded += [str(Path(b) / point["bundle"]) for point in manifest["points"]]
        else:
            expanded.append(b)
    # Point bundles are named after their key.
    sweep_names = [Path(b).stem for b in expanded]
    bundles = expanded

    # Parse names
    if names:
        name_list = names.split(",")
        if len(name_list) != len(bundles):
            _error_json(
                "COMPARE_ERROR",
                f"Got {len(name_list)} names but {len(bundles)} bundles.",
            )
    else:
        # Default: use directory stem
        name_list = sweep_names

    if len(set(name_list)) != len(name_list):
        _error_json(
            "COMPARE_ERROR",
            "Two bundles share a name; pass distinct names with --names.",
        )
    bundle_map = dict(zip(name_list, bundles))

    # Parse options
    metric_list = metrics.split(",") if metrics else None
    var_list = variables.split(",") if variables else None
    rounded = not raw and precision is None

    try:
        result, errors = compare_bundles(
            bundle_map,
            metrics=metric_list,
            variables=var_list,
            with_se=True,
        )
    except Exception as e:
        _error_json("COMPARE_ERROR", str(e))

    # Compute deltas if baseline specified.
    # Rounding is handled at the output layer.
    if baseline:
        if baseline not in result:
            _error_json(
                "COMPARE_ERROR",
                f"Baseline '{baseline}' not found. Available: {list(result.keys())}",
            )
        deltas, delta_errors = compare_deltas(result, baseline, errors)
        result["_deltas_vs_" + baseline] = deltas
        errors["_deltas_vs_" + baseline] = delta_errors

    # Assumption differences come first: they explain the metric deltas.
    differences = compare_assumptions(
        {name: load_bundle(path) for name, path in bundle_map.items()}
    )
    result = {"_assumptions": differences, **result}
    errors = {"_assumptions": {}, **errors}

    if rounded:
        _print_json(round_comparison(result, errors))
    else:
        _print_json(result, precision=precision)


# ---------------------------------------------------------------------------
# populations
# ---------------------------------------------------------------------------


@cli.command()
def populations():
    """List available population datasets."""
    try:
        from ..population.population import get_available_locations

        locations = get_available_locations()["location"].tolist()
        _print_json({"populations": locations})
    except Exception as e:
        _error_json("POPULATION_ERROR", str(e))


# ---------------------------------------------------------------------------
# verify / replay
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("bundle_path", type=click.Path(exists=True))
@click.option(
    "--offline",
    is_flag=True,
    default=False,
    help="Do not download the population data to check its hashes.",
)
def verify(bundle_path, offline):
    """Check a bundle against the hashes recorded when it was written.

    Recomputes the hash of every bundle file, of the manifest and of the
    stored config, and of the population data the run used (fetched again at
    the recorded version unless --offline). Prints a drift report and exits
    with status 1 if anything changed. Differences between the environment
    that wrote the bundle and this one are reported, not counted as drift.

    \b
    Examples:
      epydemix verify results.epx
      epydemix verify results.epx --offline
    """
    from ..io.provenance import verify_bundle

    try:
        report = verify_bundle(bundle_path, check_population=not offline)
    except Exception as e:
        _error_json("VERIFY_ERROR", str(e))
    _print_json(report)
    if not report["verified"]:
        sys.exit(1)


@cli.command()
@click.argument("bundle_path", type=click.Path(exists=True))
@click.option(
    "--tol",
    type=float,
    default=0.05,
    show_default=True,
    help="Largest relative difference that still counts as reproduced.",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Keep the re-run bundle at this path (default: a temporary directory).",
)
def replay(bundle_path, tol, output):
    """Re-run a bundle from its stored config and seed, and compare results.

    Re-executes the command that wrote the bundle (run, calibrate or project,
    the last from the recorded parent bundle) with the bundle's config.yaml,
    then compares every number in the two summary.json files. Prints the
    largest relative difference and exits with status 1 if it exceeds --tol.

    \b
    Examples:
      epydemix replay results.epx
      epydemix replay results.epx --tol 0 -o replayed.epx
    """
    from .replay import replay_bundle

    try:
        report = replay_bundle(bundle_path, output=output, tol=tol)
    except Exception as e:
        _error_json("REPLAY_ERROR", str(e))
    _print_json(report)
    if not report["reproduced"]:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
