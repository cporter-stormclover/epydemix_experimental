"""summary.json: written by run, calibrate and project; a cache of inspect."""

import json
import os

import pytest
import yaml
from click.testing import CliRunner

from epydemix.cli.main import cli
from epydemix.io import build_summary, inspect_bundle, load_summary, write_summary

from .case_study import write_case_study
from .test_cli import MINIMAL_CONFIG, _make_bundle, _run_calibration, _write_yaml


def _invoke(*args):
    result = CliRunner().invoke(cli, [str(a) for a in args])
    assert result.exit_code == 0, result.output
    return result


def _read(bundle, name):
    with open(os.path.join(bundle, name)) as f:
        return json.load(f)


def _assert_matches_inspect(bundle):
    """Every number in summary.json equals what inspect computes from Parquet.

    The rounded values match inspect's default (rounded) report and the
    values under ``raw`` match ``inspect --raw``.
    """
    summary = _read(bundle, "summary.json")
    variables = list(summary["variables"])
    assert variables, "a simulation bundle summarizes its _total compartments"
    for report, tree, approx in (
        ("rounded", summary["variables"], lambda v: v),
        ("raw", summary["raw"]["variables"], pytest.approx),
    ):
        peak = inspect_bundle(bundle, "peak", variables=variables, report=report)
        stats = inspect_bundle(bundle, "summary", variables=variables, report=report)
        for var in variables:
            got = tree[var]
            for q, legacy in (("0.05", "0.05"), ("0.5", "0.50"), ("0.95", "0.95")):
                assert got["peak"][q] == approx(peak[var]["peak_value"][legacy])
                assert got["peak_date"][q] == peak[var]["peak_date"][legacy]
                assert got["final"][q] == approx(stats[var]["final_value"][legacy])
    rounded_peak = inspect_bundle(bundle, "peak", variables=variables, report="rounded")
    for var in variables:
        assert (
            summary["variables"][var]["peak_date"]["reported"]
            == rounded_peak[var]["peak_date"]["reported"]
        )


class TestWrittenByEveryRunCommand:
    def test_run(self, tmp_path):
        out = tmp_path / "run.epx"
        _invoke("run", _write_yaml(tmp_path, MINIMAL_CONFIG), "-o", out)
        manifest = _read(out, "manifest.json")
        assert manifest["files"]["summary"]["path"] == "summary.json"
        assert manifest["files"]["summary"]["schema_version"] == 3
        summary = _read(out, "summary.json")
        assert summary["schema_version"] == 3
        assert summary["bundle_id"] == manifest["id"]
        assert summary["n_simulations"] == MINIMAL_CONFIG["simulation"]["n_simulations"]
        assert set(summary["variables"]) == {
            "Susceptible_total",
            "Infected_total",
            "Recovered_total",
        }
        assert summary["attack_rate_method"] == "transition_flows"
        assert summary["attack_rate_flows"] == ["Susceptible_to_Infected_total"]
        assert summary["attack_rate"]["units"] == "percent"
        _assert_matches_inspect(str(out))

    def test_stdout_manifest_lists_the_summary(self, tmp_path):
        out = tmp_path / "run.epx"
        result = CliRunner().invoke(
            cli, ["run", _write_yaml(tmp_path, MINIMAL_CONFIG), "-o", str(out)]
        )
        printed = json.loads(result.stdout)
        assert "summary" in printed["files"]

    def test_calibrate_has_posterior_statistics(self, tmp_path):
        bundle = _run_calibration(tmp_path)
        summary = _read(bundle, "summary.json")
        for report, tree in (
            ("rounded", summary["posterior"]),
            ("raw", summary["raw"]["posterior"]),
        ):
            posterior = inspect_bundle(bundle, "posterior", report=report)
            assert set(tree) == set(posterior)
            for name, stats in posterior.items():
                for key in ("mean", "median", "std"):
                    assert tree[name][key] == pytest.approx(stats[key])
                assert tree[name]["ci95"] == pytest.approx(stats["ci95"])
        assert "variables" not in summary  # no compartments.parquet

    def test_project(self, tmp_path):
        bundle = _run_calibration(tmp_path)
        out = tmp_path / "proj.epx"
        _invoke("project", bundle, "-o", out)
        assert "summary" in _read(out, "manifest.json")["files"]
        _assert_matches_inspect(str(out))


class TestIsOnlyACache:
    def test_inspect_does_not_need_it(self, tmp_path):
        out = tmp_path / "run.epx"
        _invoke("run", _write_yaml(tmp_path, MINIMAL_CONFIG), "-o", out)
        os.remove(out / "summary.json")
        result = _invoke("inspect", out, "peak", "-v", "Infected_total")
        assert "Infected_total" in json.loads(result.output)

    def test_build_summary_recomputes_it(self, tmp_path):
        out = tmp_path / "run.epx"
        _invoke("run", _write_yaml(tmp_path, MINIMAL_CONFIG), "-o", out)
        assert build_summary(str(out)) == _read(out, "summary.json")
        os.remove(out / "summary.json")
        assert load_summary(str(out))["variables"]

    def test_python_api_bundles_can_be_summarized(self, tmp_path):
        bundle = _make_bundle(tmp_path)
        manifest = write_summary(bundle)
        assert manifest["files"]["summary"]["path"] == "summary.json"
        assert _read(bundle, "manifest.json")["files"]["summary"]
        _assert_matches_inspect(bundle)


def test_case_study_writes_four_valid_summaries(tmp_path, offline_italy):
    """The AGENT.md case study: four scenario bundles, each with a summary."""
    configs = write_case_study(tmp_path, n_simulations=5)
    for name, config in configs.items():
        out = tmp_path / f"{name}.epx"
        _invoke("run", config, "-o", out)
        _assert_matches_inspect(str(out))
        assert "Vaccinated_total" in _read(out, "summary.json")["variables"]
    stored = yaml.safe_load(open(tmp_path / "high.epx" / "config.yaml"))
    assert stored["parameters"]["vaccination_rate"] == 0.0115
