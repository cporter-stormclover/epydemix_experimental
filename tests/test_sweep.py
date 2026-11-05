"""The sweep primitive: grammar, seeds, validation before running, aggregation."""

import copy
import json
import os

import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from epydemix.cli.main import cli
from epydemix.cli.sweep import (
    SweepError,
    expand_grid,
    expand_range,
    metric_path,
    parse_duration,
    point_seed,
    render_template,
    set_path,
)

from .test_cli import _write_yaml

# The school-closure task: a three-week school closure starting on each of
# 18 weekly dates from 1 Dec 2026 to 1 Apr 2027 (flat population, for speed).
CLOSURE = {
    "model": {"type": "SEIR"},
    "derived": {"transmission_rate": {"from": "R0", "value": 1.6}},
    "parameters": {"incubation_rate": 0.5, "recovery_rate": 0.33},
    "population": {"size": 100000},
    "simulation": {
        "start_date": "2026-10-01",
        "end_date": "2027-06-30",
        "n_simulations": 3,
        "seed": 11,
    },
    "initial_conditions": {
        "Susceptible": 0.9999,
        "Exposed": 0.00005,
        "Infected": 0.00005,
        "Recovered": 0.0,
    },
    "interventions": [
        {
            "layer": "all",
            "start_date": "2026-12-01",
            "end_date": "2026-12-22",
            "reduction": 0.3,
        }
    ],
    "assumptions": [
        {
            "parameter": "incubation_rate",
            "value": 0.5,
            "basis": "derived",
            "rationale": "2 days",
        },
        {
            "parameter": "recovery_rate",
            "value": 0.33,
            "basis": "derived",
            "rationale": "3 days",
        },
    ],
    "sweep": {
        "name": "closure_timing",
        "metrics": ["attack_rate", "Infected_total.peak", "Infected_total.peak_date"],
        "over": [
            {
                "path": "interventions.0.start_date",
                "name": "start_date",
                "range": {"start": "2026-12-01", "stop": "2027-04-01", "step": "7d"},
            }
        ],
        "derive": {"interventions.0.end_date": "{{ start_date + 21d }}"},
    },
}


def _invoke(*args, expect=0):
    result = CliRunner().invoke(cli, [str(a) for a in args])
    assert result.exit_code == expect, result.output
    return result


def _with_sweep(**changes):
    cfg = copy.deepcopy(CLOSURE)
    cfg["sweep"].update(changes)
    return cfg


class TestGrammar:
    def test_eighteen_weekly_dates(self):
        dates = expand_range(
            {"start": "2026-12-01", "stop": "2027-04-01", "step": "7d"}
        )
        assert len(dates) == 18
        assert dates[0] == "2026-12-01" and dates[-1] == "2027-03-30"

    def test_numeric_range_includes_stop(self):
        assert expand_range({"start": 0.1, "stop": 0.3, "step": 0.1}) == [0.1, 0.2, 0.3]
        assert expand_range({"start": 1, "stop": 7, "step": 3}) == [1, 4, 7]

    @pytest.mark.parametrize(
        "spec",
        [
            {"start": "2026-12-01", "stop": "2027-04-01", "step": 7},
            {"start": 0.1, "stop": 0.3, "step": -0.1},
            {"start": 0.1, "stop": 0.3},
            {"start": "2026-12-01", "stop": 3, "step": "7d"},
        ],
    )
    def test_bad_ranges(self, spec):
        with pytest.raises(SweepError):
            expand_range(spec)

    def test_durations(self):
        assert parse_duration("7d") == 7
        assert parse_duration("2w") == 14
        assert parse_duration("7") is None

    @pytest.mark.parametrize(
        "template, expected",
        [
            ("{{ start_date + 21d }}", "2026-12-22"),
            ("{{ start_date - 1w }}", "2026-11-24"),
            ("{{start_date+2w-1d}}", "2026-12-14"),
            ("{{ start_date }}", "2026-12-01"),
        ],
    )
    def test_templates(self, template, expected):
        assert render_template(template, {"start_date": "2026-12-01"}) == expected

    @pytest.mark.parametrize(
        "template",
        [
            "{{ start_date * 2 }}",
            "{{ start_date + 21 }}",
            "{{ __import__('os').system('true') }}",
            "{{ start_date.year }}",
            "{{ other + 1d }}",
            "start_date + 21d",
            "{{ start_date + 21d }} and more",
        ],
    )
    def test_templates_reject_anything_else(self, template):
        with pytest.raises(SweepError):
            render_template(template, {"start_date": "2026-12-01"})

    def test_durations_need_a_date(self):
        with pytest.raises(SweepError, match="not a date"):
            render_template("{{ rate + 1d }}", {"rate": 0.3})

    def test_set_path(self):
        cfg = {"interventions": [{"start_date": "x"}], "parameters": {}}
        set_path(cfg, "interventions.0.start_date", "2026-12-01")
        set_path(cfg, "interventions[0].reduction", 0.5)
        set_path(cfg, "parameters.beta", 0.3)
        set_path(cfg, "simulation.seed", 4)
        assert cfg["interventions"][0] == {"start_date": "2026-12-01", "reduction": 0.5}
        assert cfg["parameters"]["beta"] == 0.3 and cfg["simulation"]["seed"] == 4
        with pytest.raises(SweepError, match="no item 3"):
            set_path(cfg, "interventions.3.start_date", "2026-12-01")

    def test_cartesian_product_and_keys(self):
        points = expand_grid(
            {
                "over": [
                    {"path": "parameters.a", "values": [1, 2]},
                    {"path": "parameters.b", "values": [0.1, 0.2, 0.3]},
                ]
            }
        )
        assert len(points) == 6
        assert points[0]["key"] == "a=1__b=0.1"
        assert points[-1]["paths"] == {"parameters.a": 2, "parameters.b": 0.3}

    def test_seeds_are_deterministic_and_32_bit(self):
        assert point_seed(11, "a=1") == point_seed(11, "a=1")
        assert point_seed(11, "a=1") != point_seed(11, "a=2")
        assert point_seed(12, "a=1") != point_seed(11, "a=1")
        assert 0 <= point_seed(11, "a=1") < 2**32

    def test_metric_shortcuts(self):
        assert metric_path("attack_rate") == ["attack_rate", "0.5"]
        assert metric_path("Infected_total.peak") == [
            "variables",
            "Infected_total",
            "peak",
            "0.5",
        ]
        assert metric_path("I_total.final.0.95") == [
            "variables",
            "I_total",
            "final",
            "0.95",
        ]


class TestSweepCommand:
    @pytest.fixture(scope="class")
    def sweep(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("sweep")
        out = tmp / "closure.epx"
        result = _invoke(
            "sweep",
            _write_yaml(tmp, CLOSURE, "closure.yaml"),
            "-o",
            out,
            "--workers",
            "2",
        )
        return tmp, out, json.loads(result.stdout)

    def test_one_call_runs_every_point(self, sweep):
        _, out, printed = sweep
        assert printed["n_points"] == 18
        assert len(os.listdir(out / "points")) == 18
        with open(out / "summary.json") as f:
            summary = json.load(f)
        assert [p["variables"]["start_date"] for p in summary["points"]][:2] == [
            "2026-12-01",
            "2026-12-08",
        ]
        assert set(summary["points"][0]["metrics"]) == set(CLOSURE["sweep"]["metrics"])
        with open(out / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["type"] == "SweepResults"
        assert manifest["sweep"]["parent_seed"] == 11

    def test_each_point_is_a_complete_bundle(self, sweep):
        _, out, printed = sweep
        row = printed["points"][5]
        point = out / row["bundle"]
        with open(point / "manifest.json") as f:
            prov = json.load(f)["provenance"]
        assert prov["command"] == "sweep"
        assert prov["point_key"] == row["key"]
        assert prov["parent_seed"] == 11
        assert prov["seed"] == row["seed"] == point_seed(11, row["key"])
        stored = yaml.safe_load(open(point / "config.yaml"))
        assert "sweep" not in stored
        assert (
            stored["interventions"][0]["start_date"] == row["variables"]["start_date"]
        )
        assert stored["interventions"][0]["end_date"] == render_template(
            "{{ start_date + 21d }}", row["variables"]
        )

    def test_rerunning_a_point_config_reproduces_it(self, sweep, tmp_path):
        _, out, printed = sweep
        point = out / printed["points"][7]["bundle"]
        again = tmp_path / "again.epx"
        _invoke("run", point / "config.yaml", "-o", again)
        a = pd.read_parquet(point / "compartments.parquet")
        b = pd.read_parquet(again / "compartments.parquet")
        assert a.equals(b)

    def test_points_differ_and_seeds_differ(self, sweep):
        _, _, printed = sweep
        assert len({p["seed"] for p in printed["points"]}) == 18

    def test_parent_bundle_is_verifiable(self, sweep):
        _, out, _ = sweep
        report = json.loads(_invoke("verify", out, "--offline").stdout)
        assert report["verified"]

    def test_compare_accepts_a_sweep_bundle(self, sweep):
        _, out, printed = sweep
        result = json.loads(_invoke("compare", out, "-m", "attack_rate").stdout)
        scenarios = [k for k in result if not k.startswith("_")]
        assert scenarios == [p["key"] for p in printed["points"]]

    def test_reference_deltas(self, sweep, tmp_path):
        tmp, out, printed = sweep
        cfg = _with_sweep(reference=str(out / printed["points"][0]["bundle"]))
        cfg["sweep"]["over"][0]["range"]["stop"] = "2026-12-08"
        ref_out = tmp_path / "ref.epx"
        result = _invoke("sweep", _write_yaml(tmp_path, cfg), "-o", ref_out)
        rows = json.loads(result.stdout)["points"]
        assert rows[0]["deltas"]["attack_rate"] == 0
        assert rows[0]["deltas"]["Infected_total.peak_date"] == 0
        assert isinstance(rows[1]["deltas"]["Infected_total.peak_date"], int)


class TestValidationBeforeRunning:
    def test_out_of_range_value_fails_before_any_simulation(
        self, tmp_path, monkeypatch
    ):
        import epydemix.cli.config as config_module

        def no_runs(*args, **kwargs):
            raise AssertionError("a simulation started")

        monkeypatch.setattr(config_module, "run_from_config", no_runs)
        cfg = copy.deepcopy(CLOSURE)
        cfg["sweep"] = {
            "over": [{"path": "parameters.recovery_rate", "values": [0.2, 0.3, 25.0]}]
        }
        out = tmp_path / "bad.epx"
        result = CliRunner().invoke(
            cli, ["sweep", _write_yaml(tmp_path, cfg), "-o", str(out)]
        )
        assert result.exit_code == 1
        error = json.loads(result.output[result.output.index("{") :])
        assert error["point"] == "recovery_rate=25.0"
        assert any("above the maximum" in d for d in error["details"])
        assert not out.exists()

    def test_validate_checks_every_point(self, tmp_path):
        report = json.loads(_invoke("validate", _write_yaml(tmp_path, CLOSURE)).stdout)
        assert report == {"valid": True, "errors": [], "warnings": [], "n_points": 18}
        cfg = copy.deepcopy(CLOSURE)
        cfg["sweep"]["over"][0]["path"] = "interventions.4.start_date"
        report = json.loads(
            _invoke("validate", _write_yaml(tmp_path, cfg), expect=1).stdout
        )
        assert report["point"].startswith("start_date=")

    def test_malformed_block(self, tmp_path):
        cfg = _with_sweep(over=[{"path": "parameters.recovery_rate"}])
        result = _invoke(
            "sweep", _write_yaml(tmp_path, cfg), "-o", tmp_path / "x.epx", expect=1
        )
        assert "exactly one of 'values' or 'range'" in result.output

    def test_run_refuses_a_sweep_config(self, tmp_path):
        result = _invoke(
            "run", _write_yaml(tmp_path, CLOSURE), "-o", tmp_path / "x.epx", expect=1
        )
        assert "epydemix sweep" in result.output

    def test_swept_parameters_are_declared(self, tmp_path):
        cfg = copy.deepcopy(CLOSURE)
        cfg["sweep"] = {
            "over": [{"path": "parameters.recovery_rate", "values": [0.2, 0.25]}]
        }
        cfg["assumptions"] = cfg["assumptions"][:1]
        report = json.loads(_invoke("validate", _write_yaml(tmp_path, cfg)).stdout)
        assert report["warnings"] == []
