"""The assumptions block, qualitative bands and config schema versions."""

import copy
import json
import os

import pytest
import yaml
from click.testing import CliRunner

from epydemix import load_predefined_model
from epydemix.cli.assumptions import compare, merge, prompt_literals
from epydemix.cli.config import load_config, validate_config
from epydemix.cli.main import cli
from epydemix.cli.schema_version import CONFIG_SCHEMA_VERSION
from epydemix.parameters import ParameterSpec

from .case_study import SCENARIOS, write_case_study
from .test_cli import MINIMAL_CONFIG, _write_yaml


def _invoke(*args, expect=0):
    result = CliRunner().invoke(cli, [str(a) for a in args])
    assert result.exit_code == expect, result.output
    return result


def _undeclared(warnings, name):
    return [w for w in warnings if f"assumptions entry for '{name}'" in w]


def _entry(parameter, value, basis="agent_choice", rationale="chosen"):
    return {
        "parameter": parameter,
        "value": value,
        "basis": basis,
        "rationale": rationale,
    }


class TestCaseStudy:
    def test_each_overlay_warns_about_its_undeclared_vaccination_rate(self, tmp_path):
        for name, path in write_case_study(tmp_path).items():
            report = json.loads(_invoke("validate", path).output)
            assert report["valid"]
            found = _undeclared(report["warnings"], "vaccination_rate")
            assert len(found) == 1, (name, report["warnings"])
            assert repr(SCENARIOS[name]) in found[0]

    def test_declared_assumptions_silence_every_warning(self, tmp_path):
        for name, path in write_case_study(tmp_path, declare=True).items():
            report = json.loads(_invoke("validate", path).output)
            assert report == {"valid": True, "errors": [], "warnings": []}, name

    def test_band_is_named_in_the_warning(self, tmp_path):
        path = write_case_study(tmp_path)["low"]
        warnings = json.loads(_invoke("validate", path).output)["warnings"]
        assert "inside the 'low' band" in _undeclared(warnings, "vaccination_rate")[0]

    def test_assumptions_reach_the_manifest_and_compare(self, tmp_path, offline_italy):
        paths = write_case_study(tmp_path, n_simulations=3, declare=True)
        bundles = []
        for name in ("low", "high"):
            out = tmp_path / f"{name}.epx"
            _invoke("run", paths[name], "-o", out)
            bundles.append(str(out))
            with open(out / "manifest.json") as f:
                declared = {a["parameter"]: a for a in json.load(f)["assumptions"]}
            # Base and overlay declarations are merged, not replaced.
            assert set(declared) == {
                "transmission_rate",
                "incubation_rate",
                "recovery_rate",
                "vaccine_efficacy",
                "vaccination_rate",
            }
            assert declared["vaccination_rate"]["value"] == SCENARIOS[name]

        output = json.loads(_invoke("compare", *bundles, "-m", "peak").output)
        assert list(output)[0] == "_assumptions"  # printed before the metrics
        differing = output["_assumptions"]["differing"]
        assert [d["parameter"] for d in differing] == ["vaccination_rate"]
        assert differing[0]["values"]["low"]["value"] == 0.0015
        assert differing[0]["values"]["high"]["basis"] == "agent_choice"
        assert "transmission_rate" in output["_assumptions"]["same"]


class TestValidation:
    def test_default_values_need_no_declaration(self):
        assert not _undeclared(
            validate_config(
                {
                    **MINIMAL_CONFIG,
                    "parameters": {"transmission_rate": 0.3, "recovery_rate": 0.1},
                }
            )["warnings"],
            "transmission_rate",
        )

    def test_non_default_values_warn(self):
        cfg = copy.deepcopy(MINIMAL_CONFIG)
        cfg["parameters"]["transmission_rate"] = 0.45
        assert _undeclared(validate_config(cfg)["warnings"], "transmission_rate")

    def test_prompt_literals_count_as_read(self):
        cfg = copy.deepcopy(MINIMAL_CONFIG)
        cfg["parameters"]["transmission_rate"] = 0.45
        literals = prompt_literals("A pathogen with transmission rate 0.45 per day.")
        warnings = validate_config(cfg, prompt_literals=literals)["warnings"]
        assert not _undeclared(warnings, "transmission_rate")

    def test_validate_reads_the_prompt_file(self, tmp_path):
        cfg = copy.deepcopy(MINIMAL_CONFIG)
        cfg["parameters"]["recovery_rate"] = 0.2
        (tmp_path / "prompt.md").write_text("Recovery happens at 20% per day.")
        path = _write_yaml(tmp_path, cfg)
        without = json.loads(_invoke("validate", path).output)["warnings"]
        with_prompt = json.loads(
            _invoke("validate", path, "--prompt", tmp_path / "prompt.md").output
        )["warnings"]
        assert _undeclared(without, "recovery_rate")
        assert not _undeclared(with_prompt, "recovery_rate")

    def test_override_values_need_a_basis_too(self):
        cfg = copy.deepcopy(MINIMAL_CONFIG)
        cfg["overrides"] = [
            {
                "parameter": "transmission_rate",
                "start_date": "2023-01-05",
                "end_date": "2023-01-10",
                "value": 0.1,
            }
        ]
        warnings = validate_config(cfg)["warnings"]
        assert any(w.startswith("overrides[0].value = 0.1") for w in warnings)

    def test_calibrated_parameters_are_not_chosen(self):
        cfg = copy.deepcopy(MINIMAL_CONFIG)
        cfg["calibration"] = {"priors": {"transmission_rate": {"low": 0.1, "high": 1}}}
        assert not _undeclared(validate_config(cfg)["warnings"], "transmission_rate")

    @pytest.mark.parametrize(
        "block, message",
        [
            ({"parameter": "x"}, "missing value, basis, rationale"),
            ([_entry("recovery_rate", 0.1, basis="guess")], "basis 'guess'"),
            ([_entry("recovery_rate", 0.1, rationale="  ")], "non-empty"),
            (["recovery_rate"], "must be a mapping"),
        ],
    )
    def test_malformed_entries_are_errors(self, block, message):
        cfg = copy.deepcopy(MINIMAL_CONFIG)
        cfg["assumptions"] = block if isinstance(block, list) else [block]
        if isinstance(block, dict):
            cfg["assumptions"] = [block]
        report = validate_config(cfg)
        assert not report["valid"]
        assert any(message in e for e in report["errors"]), report["errors"]

    def test_declared_value_must_match_the_config(self):
        cfg = copy.deepcopy(MINIMAL_CONFIG)
        cfg["parameters"]["transmission_rate"] = 0.45
        cfg["assumptions"] = [_entry("transmission_rate", 0.5)]
        warnings = validate_config(cfg)["warnings"]
        assert any("declared as 0.5 but the config sets 0.45" in w for w in warnings)

    def test_declaring_an_unused_parameter_warns(self):
        cfg = copy.deepcopy(MINIMAL_CONFIG)
        cfg["assumptions"] = [_entry("waning_rate", 0.01)]
        warnings = validate_config(cfg)["warnings"]
        assert any(
            "'waning_rate' is not a parameter this config sets" in w for w in warnings
        )

    def test_custom_models_declare_every_value(self):
        cfg = {
            "model": {
                "type": "custom",
                "compartments": ["S", "I", "R"],
                "transitions": [
                    {
                        "source": "S",
                        "target": "I",
                        "kind": "mediated",
                        "params": ["beta", "I"],
                    },
                    {
                        "source": "I",
                        "target": "R",
                        "kind": "spontaneous",
                        "params": "gamma",
                    },
                ],
            },
            "parameters": {"beta": 0.3, "gamma": 0.1},
            "simulation": MINIMAL_CONFIG["simulation"],
            "initial_conditions": {"S": 0.99, "I": 0.01, "R": 0.0},
        }
        assert len(validate_config(cfg)["warnings"]) == 2
        cfg["assumptions"] = [_entry("beta", 0.3), _entry("gamma", 0.1, "derived")]
        assert validate_config(cfg)["warnings"] == []


class TestInheritance:
    def test_assumptions_merge_by_parameter(self, tmp_path):
        base = copy.deepcopy(MINIMAL_CONFIG)
        base["assumptions"] = [
            _entry("transmission_rate", 0.3),
            _entry("recovery_rate", 0.1),
        ]
        _write_yaml(tmp_path, base, "base.yaml")
        overlay = {"base": "base.yaml", "assumptions": [_entry("recovery_rate", 0.2)]}
        merged = load_config(_write_yaml(tmp_path, overlay, "child.yaml"))
        assert [(a["parameter"], a["value"]) for a in merged["assumptions"]] == [
            ("transmission_rate", 0.3),
            ("recovery_rate", 0.2),
        ]

    def test_merge_keeps_order_and_overlay_wins(self):
        assert merge(
            [{"parameter": "a", "value": 1}], [{"parameter": "a", "value": 2}]
        ) == [{"parameter": "a", "value": 2}]


class TestCompare:
    def test_undeclared_parameter_differences_are_listed(self):
        manifests = {
            "a": {"parameters_used": {"beta": 0.3, "gamma": 0.1}},
            "b": {"parameters_used": {"beta": 0.4, "gamma": 0.1}},
        }
        result = compare(manifests)
        assert result["differing"] == []
        assert result["undeclared_differences"] == [
            {"parameter": "beta", "values": {"a": 0.3, "b": 0.4}}
        ]

    def test_a_missing_declaration_is_a_difference(self):
        manifests = {
            "a": {"assumptions": [_entry("beta", 0.3)]},
            "b": {},
        }
        differing = compare(manifests)["differing"]
        assert differing[0]["values"]["b"] is None


class TestQualitativeBands:
    def test_vaccination_rate_bands_cite_a_source(self):
        spec = load_predefined_model("SIR", vaccination=True).parameter_registry.get(
            "vaccination_rate"
        )
        assert set(spec.qualitative_bands) == {"low", "moderate", "high"}
        assert all(
            "Health Affairs" in b["source"] for b in spec.qualitative_bands.values()
        )
        assert spec.band_of(0.0015) == "low"
        assert spec.band_of(0.0045) == "moderate"
        assert spec.band_of(0.0115) == "high"
        assert spec.band_of(0.007) is None

    def test_bands_are_in_the_schema(self):
        schema = json.loads(_invoke("schema", "SEIR", "--vaccination").output)
        bands = schema["properties"]["vaccination_rate"]["x-qualitative-bands"]
        assert bands["moderate"]["min"] == 0.003

    def test_a_band_without_a_source_is_rejected(self):
        with pytest.raises(ValueError, match="source"):
            ParameterSpec(
                "beta",
                "transmission",
                "rate",
                qualitative_bands={"high": {"min": 0.5, "max": 1}},
            )

    def test_bands_round_trip(self):
        bands = {"low": {"min": 0.1, "max": 0.2, "source": "a paper"}}
        spec = ParameterSpec("beta", "transmission", "rate", qualitative_bands=bands)
        assert ParameterSpec.from_dict(spec.to_dict()).qualitative_bands == bands


class TestSchemaVersion:
    def test_unversioned_configs_are_current(self):
        assert validate_config(MINIMAL_CONFIG)["valid"]

    def test_previous_version_is_accepted_with_a_warning_for_newer_keys(self):
        cfg = copy.deepcopy(MINIMAL_CONFIG)
        cfg["schema_version"] = 1
        cfg["assumptions"] = [_entry("transmission_rate", 0.3)]
        report = validate_config(cfg)
        assert report["valid"]
        assert any("introduced in schema_version 2" in w for w in report["warnings"])

    @pytest.mark.parametrize("bad", [0, "2", 1.5])
    def test_malformed_version_is_rejected(self, bad):
        cfg = {**MINIMAL_CONFIG, "schema_version": bad}
        assert not validate_config(cfg)["valid"]

    def test_future_version_is_rejected(self):
        cfg = {**MINIMAL_CONFIG, "schema_version": CONFIG_SCHEMA_VERSION + 1}
        report = validate_config(cfg)
        assert any("newer than this epydemix" in e for e in report["errors"])

    def test_upgrade_in_place_keeps_comments(self, tmp_path):
        path = tmp_path / "old.yaml"
        text = yaml.safe_dump({**MINIMAL_CONFIG, "schema_version": 1})
        path.write_text("# my notes\n" + text)
        report = json.loads(_invoke("validate", path, "--upgrade").output)
        assert report["upgraded"]["from"] == 1
        assert report["upgraded"]["to"] == CONFIG_SCHEMA_VERSION
        upgraded = path.read_text()
        assert upgraded.startswith("# my notes\n")
        assert yaml.safe_load(upgraded)["schema_version"] == CONFIG_SCHEMA_VERSION

    def test_upgrade_adds_a_version_to_unversioned_files(self, tmp_path):
        path = _write_yaml(tmp_path, MINIMAL_CONFIG)
        report = json.loads(_invoke("validate", path, "--upgrade").output)
        assert report["upgraded"]["from"] is None
        assert load_config(path)["schema_version"] == CONFIG_SCHEMA_VERSION

    def test_upgrade_json(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text(json.dumps(MINIMAL_CONFIG))
        _invoke("validate", path, "--upgrade")
        assert json.loads(path.read_text())["schema_version"] == CONFIG_SCHEMA_VERSION
        assert os.path.exists(path)
