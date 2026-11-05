"""Precision-aware reporting: rounding to Monte Carlo error and to inputs."""

import json
import os

import numpy as np
import pytest
from click.testing import CliRunner

from epydemix import load_predefined_model
from epydemix.cli.main import cli
from epydemix.io import inspect_bundle
from epydemix.io.precision import (
    bootstrap_se,
    format_parameter,
    reported_date,
    round_by_se,
    round_to_se,
)
from epydemix.parameters import ParameterSpec

from .test_cli import MINIMAL_CONFIG, _make_bundle, _write_yaml


def _json(*args):
    result = CliRunner().invoke(cli, [str(a) for a in args])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


class TestRoundingRule:
    @pytest.mark.parametrize(
        "value, se, expected",
        [
            (1532.7, 12.3, 1530),  # SE in the tens: round to tens
            (1532.7, 1.2, 1533),
            (0.0359742, 0.004, 0.036),
            (0.0359742, 0.0004, 0.036),
            (0.0359742, 0.00004, 0.03597),
            (82.29780957, 0.004, 82.298),
            (120.0, 500.0, 100),  # an error larger than the value
        ],
    )
    def test_first_significant_digit_of_the_error(self, value, se, expected):
        assert round_to_se(value, se) == expected

    def test_coarse_results_are_integers(self):
        assert isinstance(round_to_se(1532.7, 12.3), int)
        assert isinstance(round_to_se(0.0359742, 0.004), float)

    def test_no_error_keeps_six_significant_figures(self):
        assert round_to_se(0.035974202777, 0.0) == 0.0359742
        assert round_to_se(99869.0, None) == 99869.0

    def test_non_numbers_pass_through(self):
        assert round_to_se("2026-01-01", 1.0) == "2026-01-01"
        assert round_to_se(True, 1.0) is True

    def test_round_by_se_follows_the_error_tree(self):
        tree = {"a": {"median": 10.123, "ci90": [9.87, 10.55]}, "b": 3.14159}
        se = {"a": {"median": 0.1, "ci90": [0.02, 0.3]}}
        assert round_by_se(tree, se) == {
            "a": {"median": 10.1, "ci90": [9.87, 10.6]},
            "b": 3.14159,
        }


class TestBootstrap:
    def test_deterministic(self):
        x = np.random.default_rng(1).normal(size=100)
        assert np.array_equal(bootstrap_se(x, [0.5]), bootstrap_se(x, [0.5]))

    def test_matches_the_theoretical_error_of_the_median(self):
        # One bootstrap estimate scatters by ~20%; their average is unbiased.
        n = 400
        estimates = [
            bootstrap_se(np.random.default_rng(seed).normal(size=n), [0.5])[0]
            for seed in range(20)
        ]
        theory = 1.2533 / np.sqrt(n)  # sqrt(pi/2) * sigma / sqrt(n)
        assert np.mean(estimates) == pytest.approx(theory, rel=0.1)

    def test_single_sample_has_no_error(self):
        assert bootstrap_se([3.0], [0.05, 0.5]).tolist() == [0.0, 0.0]


class TestParameterPrecision:
    def test_derived_parameter_takes_the_minimum_input_precision(self):
        registry = load_predefined_model("SEIR").parameter_registry
        value = 2.5 * 0.2 / 13.89884865809907  # 0.0359742...
        rounded = format_parameter(
            "transmission_rate",
            value,
            registry,
            inputs=["R0", "recovery_rate"],
            input_precision={"R0": 2},
        )
        assert rounded == 0.036
        assert str(rounded) == "0.036"

    def test_predefined_specs_declare_precision(self):
        registry = load_predefined_model("SEIR", vaccination=True).parameter_registry
        assert all(spec.precision == 2 for spec in registry)
        schema = registry.to_json_schema()
        assert schema["properties"]["recovery_rate"]["x-precision"] == 2

    def test_precision_round_trips(self):
        spec = ParameterSpec("beta", "transmission", "rate", precision=3)
        assert ParameterSpec.from_dict(spec.to_dict()).precision == 3

    @pytest.mark.parametrize("bad", [0, -1, 2.5, True])
    def test_precision_must_be_a_positive_integer(self, bad):
        with pytest.raises(ValueError, match="precision"):
            ParameterSpec("beta", "transmission", "rate", precision=bad)


class TestReportedDates:
    def test_single_day(self):
        assert reported_date("2026-01-04", "2026-01-04", "2026-01-05") == "2026-01-04"

    def test_interval_when_spread_exceeds_a_day(self):
        assert (
            reported_date("2026-01-03", "2026-01-04", "2026-01-06")
            == "2026-01-03/2026-01-06"
        )


class TestCLIReporting:
    @pytest.fixture
    def bundle(self, tmp_path):
        out = tmp_path / "run.epx"
        CliRunner().invoke(
            cli, ["run", _write_yaml(tmp_path, MINIMAL_CONFIG), "-o", str(out)]
        )
        return str(out)

    def test_default_rounds_to_monte_carlo_error(self, bundle):
        rounded = _json("inspect", bundle, "peak", "-v", "Infected_total")
        raw = _json("inspect", bundle, "peak", "-v", "Infected_total", "--raw")
        peak, raw_peak = rounded["Infected_total"], raw["Infected_total"]
        assert "reported" in peak["peak_date"]
        assert "reported" not in raw_peak["peak_date"]
        for q in ("0.05", "0.50", "0.95"):
            assert peak["peak_value"][q] == pytest.approx(
                raw_peak["peak_value"][q], rel=0.05
            )

    def test_explicit_round_keeps_decimal_rounding(self, bundle):
        raw = _json("inspect", bundle, "quantiles", "-v", "Infected_total", "--raw")
        two = _json(
            "inspect", bundle, "quantiles", "-v", "Infected_total", "--round", "2"
        )
        assert two["Infected_total"]["0.5"] == [
            round(v, 2) for v in raw["Infected_total"]["0.5"]
        ]

    def test_quantiles_are_rounded_per_date(self, bundle):
        rounded = _json("inspect", bundle, "quantiles", "-v", "Infected_total")
        raw = inspect_bundle(bundle, "quantiles", variables=["Infected_total"])
        assert len(rounded["Infected_total"]["0.5"]) == len(
            raw["Infected_total"]["0.5"]
        )
        assert rounded["Infected_total"]["0.5"] != raw["Infected_total"]["0.5"]

    def test_manifest_is_never_rounded(self, bundle):
        manifest = _json("inspect", bundle, "manifest")
        with open(os.path.join(bundle, "manifest.json")) as f:
            assert manifest == json.load(f)

    def test_summary_json_matches_rounded_inspect(self, bundle):
        with open(os.path.join(bundle, "summary.json")) as f:
            summary = json.load(f)
        peak = _json("inspect", bundle, "peak", "-v", "Infected_total")[
            "Infected_total"
        ]
        assert (
            summary["variables"]["Infected_total"]["peak"]["0.5"]
            == peak["peak_value"]["0.50"]
        )
        assert (
            summary["variables"]["Infected_total"]["peak_date"]["reported"]
            == peak["peak_date"]["reported"]
        )
        assert summary["precision"]["n_boot"] == 200

    def test_inspect_reuses_the_cached_error(self, bundle):
        path = os.path.join(bundle, "summary.json")
        with open(path) as f:
            summary = json.load(f)
        # A cached error of 10,000 must round the peak to tens of thousands.
        summary["mc_se"]["variables"]["Infected_total"]["peak"] = {
            "0.05": 1e4,
            "0.5": 1e4,
            "0.95": 1e4,
        }
        with open(path, "w") as f:
            json.dump(summary, f)
        peak = _json("inspect", bundle, "peak", "-v", "Infected_total")
        assert all(
            v % 10000 == 0 for v in peak["Infected_total"]["peak_value"].values()
        )
        # A time slice cannot use the whole-window cache.
        sliced = _json(
            "inspect", bundle, "peak", "-v", "Infected_total", "--start", "2023-01-02"
        )
        assert (
            sliced["Infected_total"]["peak_value"]
            != peak["Infected_total"]["peak_value"]
        )

    def test_compare_rounds_medians_and_deltas(self, tmp_path):
        a = _make_bundle(tmp_path, "a.epx")
        b = _make_bundle(tmp_path, "b.epx")
        rounded = _json("compare", a, b, "-m", "peak,attack_rate", "-b", "a")
        raw = _json("compare", a, b, "-m", "peak,attack_rate", "-b", "a", "--raw")
        assert rounded["_deltas_vs_a"]["b"]["peak"] == pytest.approx(
            raw["_deltas_vs_a"]["b"]["peak"],
            abs=abs(raw["_deltas_vs_a"]["b"]["peak"]) + 50,
        )
        assert rounded["a"]["peak"]["median"] == pytest.approx(
            raw["a"]["peak"]["median"], rel=0.05
        )

    def test_python_api_stays_raw(self, bundle):
        default = inspect_bundle(bundle, "peak", variables=["Infected_total"])
        assert "reported" not in default["Infected_total"]["peak_date"]
        with pytest.raises(ValueError, match="report"):
            inspect_bundle(bundle, "peak", report="pretty")
