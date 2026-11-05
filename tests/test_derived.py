"""Derived parameters, semantic checks and the model-aware attack rate."""

import copy
import json
import os

import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from epydemix import load_predefined_model
from epydemix.cli.config import load_config, resolve_derived, validate_config
from epydemix.cli.main import cli
from epydemix.io import compare_bundles
from epydemix.model.reproduction import (
    basic_reproduction_number,
    disease_ngm,
    infection_flow_columns,
    solve_rate_for_r0,
    split_transition,
)
from epydemix.parameters import SemanticCheck

from .case_study import BASE, write_case_study
from .conftest import italy_population
from .test_cli import MINIMAL_CONFIG, _make_bundle, _write_yaml


def _invoke(*args, expect=0):
    result = CliRunner().invoke(cli, [str(a) for a in args])
    assert result.exit_code == expect, result.output
    return result


def _json(*args, expect=0):
    return json.loads(_invoke(*args, expect=expect).stdout)


def _italy_seir(**params):
    model = load_predefined_model("SEIR", vaccination=True, **params)
    model.set_population(italy_population())
    return model


class TestNextGenerationMatrix:
    def test_seir_gamma_is_the_recovery_rate(self):
        model = _italy_seir(recovery_rate=0.2, incubation_rate=0.25)
        solution = solve_rate_for_r0(model, 2.5)
        assert solution["gamma"] == pytest.approx(0.2)
        assert solution["rho"] == pytest.approx(13.8988, rel=1e-4)
        assert solution["value"] == pytest.approx(0.036, rel=0.01)
        model.parameters["transmission_rate"] = solution["value"]
        assert basic_reproduction_number(model) == pytest.approx(2.5)

    def test_seiar_closed_form(self):
        model = load_predefined_model(
            "SEIAR",
            transmission_rate=1.0,
            recovery_rate=0.2,
            asymptomatic_fraction=0.4,
            asymptomatic_recovery_rate=0.25,
            asymptomatic_relative_infectivity=0.5,
        )
        expected = 0.6 / 0.2 + 0.4 * 0.5 / 0.25
        assert disease_ngm(model)["R0_per_contact"] == pytest.approx(expected)

    def test_outcome_modules_keep_the_removal_rate(self):
        for outcome, rate in (
            ("deaths", "mortality_rate"),
            ("hospitalization", "hospitalization_rate"),
        ):
            model = load_predefined_model(
                "SEIR",
                outcome=outcome,
                transmission_rate=1.0,
                recovery_rate=0.2,
                **{rate: 0.1},
            )
            assert disease_ngm(model)["R0_per_contact"] == pytest.approx(5.0)

    def test_breakthrough_infections_do_not_count(self):
        model = _italy_seir(recovery_rate=0.2, vaccine_efficacy=0.0)
        assert "vaccine_efficacy" not in solve_rate_for_r0(model, 2.5)["inputs"]

    def test_rate_must_be_proportional(self):
        model = load_predefined_model("SIR", recovery_rate=0.1)
        with pytest.raises(ValueError, match="not proportional"):
            solve_rate_for_r0(model, 2.0, parameter="recovery_rate")

    def test_split_transition_handles_underscores(self):
        comps = ["S", "I_mild", "I_severe", "R"]
        assert split_transition("I_mild_to_R_total", comps) == ("I_mild", "R")
        assert split_transition("S_to_I_severe_total", comps) == ("S", "I_severe")
        cols = ["S_to_I_mild_total", "I_mild_to_I_severe_total", "I_mild_to_R_total"]
        assert infection_flow_columns(cols, comps, ["I_mild", "I_severe"]) == [
            "S_to_I_mild_total"
        ]


class TestDerive:
    def test_case_study_derives_0_036(self, tmp_path, offline_italy):
        config = write_case_study(tmp_path)["high"]
        rounded = _json("derive", config)["derived"]["transmission_rate"]
        raw = _json("derive", config, "--raw")["derived"]["transmission_rate"]
        assert rounded["value"] == 0.036
        assert raw["value"] == pytest.approx(0.036, rel=0.01)
        assert raw["value"] != rounded["value"]
        assert raw["formula"] == "transmission_rate = R0 * gamma / rho(C)"
        assert raw["gamma"] == pytest.approx(0.2)
        assert rounded["rho"] == 13.9
        assert raw["contact_layers"] == ["home", "work", "school", "community"]

    def test_run_records_the_derivation(self, tmp_path, offline_italy):
        out = tmp_path / "high.epx"
        _invoke("run", write_case_study(tmp_path, n_simulations=2)["high"], "-o", out)
        with open(out / "manifest.json") as f:
            manifest = json.load(f)
        derived = manifest["derived"]["transmission_rate"]
        assert manifest["parameters_used"]["transmission_rate"] == derived["value"]
        assert {"rho", "gamma", "formula", "R0"} <= set(derived)
        # The stored config keeps the derivation, not a literal, so that a
        # re-run derives it again (and stays valid).
        stored = load_config(str(out / "config.yaml"))
        assert "transmission_rate" not in stored["parameters"]
        assert stored["derived"] == {"transmission_rate": {"from": "R0", "value": 2.5}}
        assert validate_config(stored)["valid"]
        assert _json("replay", out, "--tol", "0")["max_rel_diff"] == 0.0

    def test_no_derived_block(self, tmp_path):
        result = _invoke("derive", _write_yaml(tmp_path, MINIMAL_CONFIG), expect=1)
        assert "no 'derived' block" in result.output


def _seir_config(**sections):
    cfg = copy.deepcopy(BASE)
    cfg["population"] = {"size": 100000}
    cfg.update(sections)
    return cfg


class TestDerivedValidation:
    def test_literal_and_derived_is_rejected(self):
        cfg = _seir_config()
        cfg["parameters"]["transmission_rate"] = 0.036
        report = validate_config(cfg)
        assert any("both in parameters" in e for e in report["errors"])

    @pytest.mark.parametrize(
        "spec, message",
        [
            ({"from": "R_eff", "value": 2}, "not a known derivation"),
            ({"from": "R0", "value": -1}, "positive number"),
            ({"from": "R0"}, "positive number"),
            ("2.5", "mapping"),
        ],
    )
    def test_malformed(self, spec, message):
        report = validate_config(_seir_config(derived={"transmission_rate": spec}))
        assert any(message in e for e in report["errors"]), report["errors"]

    def test_unknown_parameter(self):
        report = validate_config(
            _seir_config(derived={"beta": {"from": "R0", "value": 2}})
        )
        assert any("'beta' is not a parameter" in e for e in report["errors"])

    def test_implausible_r0_warns(self):
        cfg = _seir_config(derived={"transmission_rate": {"from": "R0", "value": 40}})
        report = validate_config(cfg)
        assert report["valid"]
        assert any("implausible" in w for w in report["warnings"])

    def test_flat_population_has_rho_one(self):
        resolved = resolve_derived(_seir_config())
        assert resolved["transmission_rate"]["rho"] == pytest.approx(1.0)
        assert resolved["transmission_rate"]["value"] == pytest.approx(2.5 * 0.2)

    def test_calibrated_inputs_are_rejected(self):
        cfg = _seir_config()
        del cfg["parameters"]["recovery_rate"]
        cfg["calibration"] = {"priors": {"recovery_rate": {"low": 0.1, "high": 0.3}}}
        report = validate_config(cfg)
        assert any("the calibration samples" in e for e in report["errors"])

    def test_custom_model_needs_infected_roles(self):
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
            "derived": {"beta": {"from": "R0", "value": 2.0}},
            "parameters": {"gamma": 0.1},
            "simulation": MINIMAL_CONFIG["simulation"],
            "initial_conditions": {"S": 0.99, "I": 0.01, "R": 0.0},
        }
        report = validate_config(cfg)
        assert any("role: infected" in e for e in report["errors"]), report["errors"]


class TestSemanticChecks:
    def test_vaccination_rate_requires_the_module(self):
        cfg = copy.deepcopy(MINIMAL_CONFIG)
        cfg["model"] = {"type": "SIR", "vaccination": False}
        cfg["parameters"]["vaccination_rate"] = 0.01
        report = validate_config(cfg)
        assert not report["valid"]
        mentions = [e for e in report["errors"] if "vaccination_rate" in e]
        assert len(mentions) == 1  # explained once, not twice
        assert "model.vaccination: true" in mentions[0]

    def test_zero_rate_does_not_trigger_the_rule(self):
        cfg = copy.deepcopy(MINIMAL_CONFIG)
        cfg["model"] = {"type": "SIR", "waning_immunity": True}
        cfg["parameters"]["waning_rate"] = 0.01
        assert validate_config(cfg)["valid"]

    @pytest.mark.parametrize("section", ["interventions", "overrides"])
    def test_windows_must_be_ordered(self, section):
        cfg = copy.deepcopy(MINIMAL_CONFIG)
        item = {"start_date": "2023-01-20", "end_date": "2023-01-10"}
        item.update(
            {"layer": "all", "reduction": 0.5}
            if section == "interventions"
            else {"parameter": "transmission_rate", "value": 0.1}
        )
        cfg[section] = [item]
        report = validate_config(cfg)
        assert any(f"{section}[0]: start_date" in e for e in report["errors"])

    def test_checks_are_in_the_schema(self):
        schema = _json("schema", "SEIR")
        names = {c["name"] for c in schema["x-semantic-checks"]}
        assert "vaccination_rate_requires_module" in names
        assert "derived_r0_plausible" in names

    def test_round_trip(self):
        check = SemanticCheck(
            name="x", rule="range", quantity="derived.*.value", max=3, message="m"
        )
        assert SemanticCheck.from_dict(check.to_dict()) == check
        with pytest.raises(ValueError):
            SemanticCheck(name="x", rule="nope", message="m")


class TestAttackRate:
    def test_leaky_vaccine_equals_the_hand_recipe(self, tmp_path, offline_italy):
        out = tmp_path / "high.epx"
        _invoke("run", write_case_study(tmp_path, n_simulations=5)["high"], "-o", out)
        with open(out / "summary.json") as f:
            summary = json.load(f)
        assert summary["attack_rate_method"] == "transition_flows"
        assert summary["attack_rate_flows"] == [
            "Susceptible_to_Exposed_total",
            "Vaccinated_to_Exposed_total",
        ]
        flows = pd.read_parquet(out / "transitions.parquet")
        comp = pd.read_parquet(out / "compartments.parquet")
        totals = [c for c in comp.columns if c.endswith("_total")]
        N = (
            comp[comp["date"] == comp["date"].min()]
            .groupby("sim_id")[totals]
            .sum()
            .sum(axis=1)
        )
        by_hand = (
            flows.groupby("sim_id")[
                ["Susceptible_to_Exposed_total", "Vaccinated_to_Exposed_total"]
            ]
            .sum()
            .sum(axis=1)
            / N
            * 100
        )
        assert summary["raw"]["attack_rate"]["0.5"] == pytest.approx(by_hand.median())
        compared = compare_bundles({"high": str(out)}, metrics=["attack_rate"])
        assert compared["high"]["attack_rate"]["method"] == "transition_flows"
        assert compared["high"]["attack_rate"]["median"] == pytest.approx(
            by_hand.median()
        )
        # Susceptible depletion would count every vaccinated person.
        legacy = (
            1
            - comp.groupby("sim_id")["Susceptible_total"].last()
            / comp.groupby("sim_id")["Susceptible_total"].first()
        )
        assert legacy.median() * 100 > by_hand.median() + 10

    def test_predefined_bundles_without_a_config(self, tmp_path):
        bundle = _make_bundle(tmp_path)  # Python API, no config
        result = compare_bundles({"r": bundle}, metrics=["attack_rate"])
        assert result["r"]["attack_rate"]["method"] == "transition_flows"
        flows = pd.read_parquet(os.path.join(bundle, "transitions.parquet"))
        cum = flows.groupby("sim_id")["Susceptible_to_Infected_total"].sum()
        assert result["r"]["attack_rate"]["median"] == pytest.approx(
            np.median(cum / 10000 * 100)
        )

    def test_custom_models_without_roles_fall_back(self, tmp_path):
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
        out = tmp_path / "custom.epx"
        _invoke("run", _write_yaml(tmp_path, cfg), "-o", out)
        with open(out / "summary.json") as f:
            assert json.load(f)["attack_rate_method"] == "legacy"
        stored = yaml.safe_load(open(out / "config.yaml"))
        assert stored["model"]["type"] == "custom"
