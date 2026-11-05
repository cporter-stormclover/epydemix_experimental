"""Declarative custom models (model.custom), the expression grammar, age groups."""

import copy
import json
import os
import random

import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from epydemix.cli.config import (
    build_model_from_config,
    resolve_derived,
    validate_config,
)
from epydemix.cli.main import cli
from epydemix.model.expressions import ExpressionError, compile_expression
from epydemix.population.population import (
    age_group_mapping_from_cuts,
    load_epydemix_population,
)

from .test_cli import _write_yaml

# The paper's T1 task: SEIRHD with nosocomial transmission, as YAML.
SEIRHD = {
    "model": {
        "type": "custom",
        "custom": {
            "compartments": [
                "S",
                {"name": "E", "role": "infected"},
                {"name": "I", "role": "infected"},
                {"name": "H", "role": "infected"},
                "R",
                "D",
            ],
            "parameters": {
                "beta": {
                    "kind": "rate",
                    "range": [0, 2],
                    "unit": "1/days",
                    "precision": 2,
                },
                "beta_hospital": {"kind": "rate", "range": [0, 2], "unit": "1/days"},
                "sigma": {"kind": "rate", "range": [0, 10], "unit": "1/days"},
                "gamma": {"kind": "rate", "range": [0, 10], "unit": "1/days"},
                "p_hosp": {"kind": "probability", "range": [0, 1]},
                "delta": {"kind": "rate", "range": [0, 10], "unit": "1/days"},
                "p_death": {"kind": "probability", "range": [0, 1]},
            },
            "transitions": [
                {
                    "from": "S",
                    "to": "E",
                    "kind": "contact",
                    "rate": "beta",
                    "agents": ["I"],
                },
                {
                    "from": "S",
                    "to": "E",
                    "kind": "contact",
                    "rate": "beta_hospital",
                    "agents": ["H"],
                },
                {"from": "E", "to": "I", "rate": "sigma"},
                {"from": "I", "to": "H", "expression": "gamma * p_hosp"},
                {"from": "I", "to": "R", "expression": "gamma * (1 - p_hosp)"},
                {"from": "H", "to": "D", "expression": "delta * p_death"},
                {"from": "H", "to": "R", "expression": "delta * (1 - p_death)"},
            ],
        },
    },
    "parameters": {
        "beta": 0.3,
        "beta_hospital": 0.05,
        "sigma": 1 / 3,
        "gamma": 0.2,
        "p_hosp": 0.1,
        "delta": 0.125,
        "p_death": 0.2,
    },
    "population": {"size": 1000000},
    "simulation": {
        "start_date": "2026-01-01",
        "end_date": "2026-03-31",
        "n_simulations": 5,
        "seed": 2026,
    },
    "initial_conditions": {"S": 999900, "E": 0, "I": 100, "H": 0, "R": 0, "D": 0},
}

# The same model in the original grammar (type: custom, source/target/params).
SEIRHD_ORIGINAL = {
    **{k: v for k, v in SEIRHD.items() if k != "model"},
    "model": {
        "type": "custom",
        "compartments": ["S", "E", "I", "H", "R", "D"],
        "transitions": [
            {"source": "S", "target": "E", "kind": "mediated", "params": ["beta", "I"]},
            {
                "source": "S",
                "target": "E",
                "kind": "mediated",
                "params": ["beta_hospital", "H"],
            },
            {"source": "E", "target": "I", "kind": "spontaneous", "params": "sigma"},
            {
                "source": "I",
                "target": "H",
                "kind": "spontaneous",
                "params": "gamma * p_hosp",
            },
            {
                "source": "I",
                "target": "R",
                "kind": "spontaneous",
                "params": "gamma * (1 - p_hosp)",
            },
            {
                "source": "H",
                "target": "D",
                "kind": "spontaneous",
                "params": "delta * p_death",
            },
            {
                "source": "H",
                "target": "R",
                "kind": "spontaneous",
                "params": "delta * (1 - p_death)",
            },
        ],
    },
}


def _invoke(*args, expect=0):
    result = CliRunner().invoke(cli, [str(a) for a in args])
    assert result.exit_code == expect, result.output
    return result


def _errors(cfg):
    return validate_config(cfg)["errors"]


def _custom(**changes):
    cfg = copy.deepcopy(SEIRHD)
    cfg["model"]["custom"].update(changes)
    return cfg


class TestExpressionGrammar:
    NAMES = {"beta", "gamma", "H", "I", "pop", "capacity"}

    @pytest.mark.parametrize(
        "source",
        [
            "beta",
            "beta * (1 - gamma)",
            "-beta + 2.5e-3 / gamma",
            "min(beta, gamma)",
            "max(0, min(1, H / (capacity * pop)), gamma)",
        ],
    )
    def test_accepted(self, source):
        compile_expression(source, self.NAMES)

    @pytest.mark.parametrize(
        "source",
        [
            "__import__('os').system('true')",
            "beta.real",
            "().__class__.__bases__",
            "pop[0]",
            "[x for x in pop]",
            "{beta: 1}",
            "lambda: beta",
            "beta if gamma else 0",
            "beta ** 2",
            "beta % 2",
            "beta < gamma",
            "beta and gamma",
            "not beta",
            "+beta",
            "abs(beta)",
            "eval('1')",
            "min(beta)",
            "min(beta, key=gamma)",
            "min(*pop)",
            "'text'",
            "True",
            "None",
            "f'{beta}'",
            "beta; gamma",
            "beta = 1",
            "unknown * beta",
            "min",
            "",
            "beta + " * 40 + "beta",
        ],
    )
    def test_rejected(self, source):
        with pytest.raises(ExpressionError):
            compile_expression(source, self.NAMES)

    def test_fuzz_never_crashes_otherwise(self):
        """Random strings are either valid expressions or ExpressionErrors."""
        rng = random.Random(8)
        alphabet = [
            "beta",
            "gamma",
            "H",
            "pop",
            "min",
            "max",
            "abs",
            "__import__",
            "os",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            ",",
            ".",
            ":",
            "+",
            "-",
            "*",
            "/",
            "**",
            "%",
            "<",
            "==",
            "and",
            "lambda",
            "if",
            "else",
            "for",
            "in",
            "1",
            "0.5",
            "1e309",
            "'s'",
            "x",
            "__class__",
            "=",
            ";",
            "\\n",
            " ",
        ]
        accepted = 0
        for _ in range(3000):
            source = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 12)))
            try:
                expr = compile_expression(source, self.NAMES)
            except ExpressionError:
                continue
            accepted += 1
            env = {
                "beta": 0.3,
                "gamma": 0.0,
                "H": np.array([0.0, 5.0]),
                "pop": np.array([10.0, 20.0]),
            }
            np.asarray(expr.evaluate(env), dtype=float)  # must not raise
        assert accepted > 20  # the fuzzer exercises valid expressions too

    def test_evaluation_is_elementwise(self):
        expr = compile_expression("min(1, H / (capacity * pop))", self.NAMES)
        out = expr.evaluate(
            {
                "H": np.array([5.0, 50.0]),
                "capacity": 0.1,
                "pop": np.array([100.0, 100.0]),
            }
        )
        assert out.tolist() == [0.5, 1.0]


class TestValidation:
    def test_seirhd_is_valid(self):
        report = validate_config(SEIRHD)
        assert report["valid"], report["errors"]

    def test_unreachable_compartment(self):
        cfg = copy.deepcopy(SEIRHD)
        cfg["model"]["custom"]["compartments"].append("Q")
        cfg["initial_conditions"]["Q"] = 0
        assert any("'Q' is unreachable" in e for e in _errors(cfg))

    def test_undeclared_rate_parameter(self):
        cfg = copy.deepcopy(SEIRHD)
        cfg["model"]["custom"]["transitions"][2]["rate"] = "sigma2"
        assert any("'sigma2' is not a declared parameter" in e for e in _errors(cfg))

    def test_undeclared_name_in_an_expression(self):
        cfg = copy.deepcopy(SEIRHD)
        cfg["model"]["custom"]["transitions"][3]["expression"] = "gamma * p_hospital"
        assert any("unknown symbol 'p_hospital'" in e for e in _errors(cfg))

    def test_values_are_checked_against_inline_specs(self):
        cfg = copy.deepcopy(SEIRHD)
        cfg["parameters"]["p_hosp"] = 1.5
        assert any("above the maximum of 1" in e for e in _errors(cfg))
        cfg = copy.deepcopy(SEIRHD)
        cfg["parameters"]["betaa"] = 0.3
        assert any("did you mean 'beta'" in e for e in _errors(cfg))

    @pytest.mark.parametrize(
        "change, message",
        [
            (
                {"from": "X", "to": "E", "rate": "beta"},
                "'X' is not a declared compartment",
            ),
            (
                {"from": "S", "to": "E", "kind": "contact", "rate": "beta"},
                "needs agents",
            ),
            (
                {"from": "S", "to": "E", "rate": "beta", "expression": "beta"},
                "exactly one of",
            ),
            ({"from": "S", "to": "S", "rate": "beta"}, "same compartment"),
            (
                {"from": "S", "to": "E", "kind": "mediated", "rate": "beta"},
                "must be one of",
            ),
        ],
    )
    def test_bad_transitions(self, change, message):
        cfg = copy.deepcopy(SEIRHD)
        cfg["model"]["custom"]["transitions"].append(change)
        assert any(message in e for e in _errors(cfg)), _errors(cfg)

    def test_bad_roles_and_specs(self):
        cfg = _custom(
            compartments=["S", {"name": "E", "role": "exposed"}, "I", "H", "R", "D"]
        )
        assert any("role 'exposed'" in e for e in _errors(cfg))
        cfg = copy.deepcopy(SEIRHD)
        cfg["model"]["custom"]["parameters"]["beta"]["kind"] = "speed"
        assert any("kind must be one of" in e for e in _errors(cfg))
        cfg = copy.deepcopy(SEIRHD)
        del cfg["parameters"]["delta"]
        assert any("declared but has no value" in e for e in _errors(cfg))


class TestRunning:
    def test_same_results_as_the_original_grammar(self, tmp_path):
        new, old = tmp_path / "new.epx", tmp_path / "old.epx"
        _invoke("run", _write_yaml(tmp_path, SEIRHD, "new.yaml"), "-o", new)
        _invoke("run", _write_yaml(tmp_path, SEIRHD_ORIGINAL, "old.yaml"), "-o", old)
        a = pd.read_parquet(new / "compartments.parquet")
        b = pd.read_parquet(old / "compartments.parquet")
        pd.testing.assert_frame_equal(a[sorted(a.columns)], b[sorted(b.columns)])

    def test_roles_give_the_model_aware_attack_rate(self, tmp_path):
        out = tmp_path / "r.epx"
        _invoke("run", _write_yaml(tmp_path, SEIRHD), "-o", out)
        with open(out / "summary.json") as f:
            summary = json.load(f)
        assert summary["attack_rate_method"] == "transition_flows"
        with open(out / "manifest.json") as f:
            assert json.load(f)["model"]["infected_compartments"] == ["E", "H", "I"]

    def test_derive_r0_for_a_custom_model(self):
        cfg = copy.deepcopy(SEIRHD)
        del cfg["parameters"]["beta"]
        cfg["derived"] = {"beta": {"from": "R0", "value": 1.54}}
        resolved = resolve_derived(cfg)["beta"]
        # R0 = beta / gamma + p_hosp * beta_hospital / delta, so beta = 0.3
        assert resolved["value"] == pytest.approx(0.3, rel=1e-6)
        assert validate_config(cfg)["valid"]

    def test_state_dependent_rate(self, tmp_path):
        cfg = copy.deepcopy(SEIRHD)
        custom = cfg["model"]["custom"]
        custom["parameters"]["capacity"] = {"kind": "proportion", "range": [0, 1]}
        cfg["parameters"]["capacity"] = 0.0001
        # Deaths rise once hospitals pass their capacity (100 beds per million).
        custom["transitions"][5]["expression"] = (
            "delta * p_death * max(1, H / (capacity * pop))"
        )
        assert validate_config(cfg)["valid"], validate_config(cfg)["errors"]
        out = tmp_path / "cap.epx"
        _invoke("run", _write_yaml(tmp_path, cfg), "-o", out)
        base = tmp_path / "base.epx"
        _invoke("run", _write_yaml(tmp_path, SEIRHD, "base.yaml"), "-o", base)

        def deaths(bundle):
            comp = pd.read_parquet(bundle / "compartments.parquet")
            return comp.groupby("sim_id")["D_total"].last().median()

        assert deaths(out) > deaths(base)
        with pytest.raises(ValueError, match="change during a run"):
            cfg["derived"] = {"beta": {"from": "R0", "value": 1.5}}
            del cfg["parameters"]["beta"]
            resolve_derived(cfg)

    def test_schema_prints_the_compartment_graph(self, tmp_path):
        path = _write_yaml(tmp_path, SEIRHD)
        schema = json.loads(_invoke("schema", path).stdout)
        graph = schema["x-compartment-graph"]
        assert {"name": "H", "role": "infected"} in graph["compartments"]
        assert {
            "from": "S",
            "to": "E",
            "kind": "contact",
            "rate": "beta_hospital",
            "agents": ["H"],
        } in graph["edges"]
        assert 'S -->|"beta_hospital via H"| E' in graph["mermaid"]
        assert schema["properties"]["beta"]["x-precision"] == 2
        _invoke("schema", _write_yaml(tmp_path, SEIRHD_ORIGINAL, "old.yaml"), expect=1)

    def test_scheduled_doses(self, tmp_path):
        cfg = copy.deepcopy(SEIRHD)
        custom = cfg["model"]["custom"]
        custom["compartments"].append("V")
        custom["transitions"].append(
            {"from": "S", "to": "V", "kind": "scheduled", "schedule": [1000] * 90}
        )
        cfg["initial_conditions"]["V"] = 0
        out = tmp_path / "v.epx"
        _invoke("run", _write_yaml(tmp_path, cfg), "-o", out)
        flows = pd.read_parquet(out / "transitions.parquet")
        assert flows.groupby("sim_id")["S_to_V_total"].sum().iloc[0] == pytest.approx(
            90000, rel=0.01
        )


def _synthetic_data(root, name="Testland"):
    """A local epydemix-data tree with single-year ages (like mistry_2021)."""
    ages = [str(a) for a in range(84)] + ["84+"]
    rng = np.random.default_rng(0)
    sizes = rng.integers(5000, 15000, size=len(ages))
    (root / "data" / name / "demographic").mkdir(parents=True)
    pd.DataFrame({"group_name": ages, "value": sizes}).to_csv(
        root / "data" / name / "demographic" / "age_distribution.csv", index=False
    )
    (root / "locations.csv").write_text(
        f"location,primary_contact_source\n{name},mistry_2021\n"
    )
    cdir = root / "data" / name / "contact_matrices" / "mistry_2021"
    cdir.mkdir(parents=True)
    for layer in ("home", "work", "school", "community"):
        pd.DataFrame(rng.uniform(0, 0.5, size=(len(ages), len(ages)))).to_csv(
            cdir / f"contacts_matrix_{layer}.csv", header=False, index=False
        )
    return sizes


class TestAgeGroups:
    def test_mapping_from_cuts(self):
        mapping = age_group_mapping_from_cuts([0, 20, 65], "mistry_2021")
        assert list(mapping) == ["0-19", "20-64", "65+"]
        assert mapping["65+"][-1] == "84+"
        prem = age_group_mapping_from_cuts([0, 20, 65], "prem_2021")
        assert prem["0-19"] == ["0-4", "5-9", "10-14", "15-19"]
        with pytest.raises(ValueError, match="bin boundary"):
            age_group_mapping_from_cuts([0, 22], "prem_2021")

    def test_regrouping_conserves_population_and_weights_contacts(self, tmp_path):
        sizes = _synthetic_data(tmp_path)
        fine = load_epydemix_population(
            "Testland",
            path_to_data=str(tmp_path) + "/",
            age_group_mapping={
                str(a) if a < 84 else "84+": [str(a) if a < 84 else "84+"]
                for a in range(85)
            },
        )
        coarse = load_epydemix_population(
            "Testland",
            path_to_data=str(tmp_path) + "/",
            age_group_mapping=age_group_mapping_from_cuts([0, 20, 65], "mistry_2021"),
        )
        assert list(coarse.Nk_names) == ["0-19", "20-64", "65+"]
        assert coarse.Nk.sum() == sizes.sum() == fine.Nk.sum()
        bounds = [(0, 20), (20, 65), (65, 85)]
        for layer in coarse.contact_matrices:
            fine_rows = fine.contact_matrices[layer].sum(axis=1)
            coarse_rows = coarse.contact_matrices[layer].sum(axis=1)
            for g, (lo, hi) in enumerate(bounds):
                weights = fine.Nk[lo:hi]
                expected = (fine_rows[lo:hi] * weights).sum() / weights.sum()
                assert coarse_rows[g] == pytest.approx(expected)

    def test_config_age_groups(self, tmp_path, monkeypatch):
        import epydemix.cli.config as config_module
        import epydemix.io.provenance as provenance
        import epydemix.model.epimodel as epimodel

        data = tmp_path / "data_repo"
        data.mkdir()
        _synthetic_data(data)
        config_module.clear_population_cache()
        monkeypatch.setattr(
            config_module, "contacts_source", lambda name, version: "mistry_2021"
        )
        monkeypatch.setattr(
            epimodel,
            "load_epydemix_population",
            lambda name, **kw: load_epydemix_population(
                name, path_to_data=str(data) + "/", **kw
            ),
        )
        monkeypatch.setattr(provenance, "_fetch", lambda url: b"offline")
        cfg = {
            "model": {"type": "SIR"},
            "population": {"name": "Testland", "age_groups": [0, 20, 65]},
            "simulation": {
                "start_date": "2026-01-01",
                "end_date": "2026-02-01",
                "n_simulations": 2,
            },
            "initial_conditions": {
                "Susceptible": {"default": 0.999, "65+": 0.99},
                "Infected": {"default": 0.001, "65+": 0.01},
                "Recovered": 0.0,
            },
        }
        out = tmp_path / "groups.epx"
        _invoke("run", _write_yaml(tmp_path, cfg), "-o", out)
        with open(out / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["population"]["demographic_groups"] == ["0-19", "20-64", "65+"]
        mapping = manifest["population"]["age_group_mapping"]
        assert (
            mapping["cuts"] == [0, 20, 65]
            and mapping["contacts_source"] == "mistry_2021"
        )
        assert mapping["groups"]["0-19"][0] == "0"
        config_module.clear_population_cache()

    @pytest.mark.parametrize("bad", [[20, 65], [0, 65, 20], [0, 1.5], "0,20"])
    def test_malformed_cuts(self, bad):
        cfg = {
            "model": {"type": "SIR"},
            "population": {"name": "Italy", "age_groups": bad},
            "simulation": {"start_date": "2026-01-01", "end_date": "2026-02-01"},
        }
        assert any("age_groups" in e for e in _errors(cfg))

    @pytest.mark.skipif(
        not os.environ.get("EPX_NETWORK_TESTS"), reason="downloads Italy"
    )
    def test_italy_three_bins(self):
        cfg = {
            "model": {"type": "SIR"},
            "population": {"name": "Italy", "age_groups": [0, 20, 65]},
        }
        model = build_model_from_config(cfg)
        default = load_epydemix_population("Italy")
        assert list(model.population.Nk_names) == ["0-19", "20-64", "65+"]
        assert model.population.Nk.sum() == default.Nk.sum()


def test_case_study_style_config_still_valid(tmp_path):
    """The original custom grammar keeps working next to model.custom."""
    assert validate_config(SEIRHD_ORIGINAL)["valid"]
    assert yaml.safe_load(yaml.safe_dump(SEIRHD))  # serializable as YAML
