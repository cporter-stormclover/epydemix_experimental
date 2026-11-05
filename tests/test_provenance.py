"""Content-addressed provenance: what a bundle records, verify and replay."""

import copy
import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from click.testing import CliRunner  # noqa: E402

from epydemix.cli.config import validate_config  # noqa: E402
from epydemix.cli.main import cli  # noqa: E402
from epydemix.cli.replay import compare_summaries  # noqa: E402
from epydemix.io import add_figure_to_manifest  # noqa: E402

from .case_study import write_case_study  # noqa: E402
from .test_cli import (  # noqa: E402
    MINIMAL_CONFIG,
    _make_bundle,
    _run_calibration,
    _write_yaml,
)


def _invoke(*args, expect=0):
    result = CliRunner().invoke(cli, [str(a) for a in args])
    assert result.exit_code == expect, result.output
    return result


def _json(*args, expect=0):
    return json.loads(_invoke(*args, expect=expect).stdout)


def _manifest(bundle):
    with open(os.path.join(bundle, "manifest.json")) as f:
        return json.load(f)


@pytest.fixture
def bundle(tmp_path):
    cfg = copy.deepcopy(MINIMAL_CONFIG)
    cfg["simulation"]["seed"] = 7
    out = tmp_path / "run.epx"
    _invoke("run", _write_yaml(tmp_path, cfg), "-o", out)
    return str(out)


class TestRecorded:
    def test_provenance_block(self, bundle, monkeypatch):
        prov = _manifest(bundle)["provenance"]
        assert prov["command"] == "run"
        assert prov["seed"] == 7
        assert len(prov["config_sha256"]) == 64
        assert len(prov["config_file_sha256"]) == 64
        assert prov["code"]["epydemix"]
        assert "commit" in prov["code"] and "dirty" in prov["code"]
        assert prov["environment"]["python"]
        assert "numpy" in prov["environment"]["packages"]
        assert prov["contract_version"]
        assert set(prov["bundle_files"]) == {
            "compartments.parquet",
            "transitions.parquet",
            "parameters.parquet",
            "config.yaml",
            "summary.json",
        }
        assert len(prov["manifest_sha256"]) == 64
        assert "agent" not in prov  # not driven by an agent runtime

    def test_agent_identity_from_the_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EPX_AGENT_RUNTIME", "claude-code")
        monkeypatch.setenv("EPX_AGENT_MODEL", "some-model")
        out = tmp_path / "agent.epx"
        _invoke("run", _write_yaml(tmp_path, MINIMAL_CONFIG), "-o", out)
        assert _manifest(out)["provenance"]["agent"] == {
            "EPX_AGENT_RUNTIME": "claude-code",
            "EPX_AGENT_MODEL": "some-model",
        }

    def test_population_data_is_hashed(self, tmp_path, offline_italy):
        paths = write_case_study(tmp_path, n_simulations=2)
        out = tmp_path / "low.epx"
        _invoke("run", paths["low"], "-o", out)
        data = _manifest(out)["provenance"]["population_data"]
        assert data["repo"] == "epistorm/epydemix-data"
        assert data["version"] == "v1.2.0"
        assert data["contacts_source"] == "mistry_2021"
        assert set(data["files"]) == {
            "locations.csv",
            "data/Italy/demographic/age_distribution.csv",
            *(
                f"data/Italy/contact_matrices/mistry_2021/contacts_matrix_{layer}.csv"
                for layer in ("home", "work", "school", "community")
            ),
        }

    def test_data_version_is_passed_to_the_loader(self, tmp_path, offline_italy):
        paths = write_case_study(tmp_path, n_simulations=2)
        with open(paths["low"], "a") as f:
            f.write("population: {data_version: v1.1.0}\n")
        out = tmp_path / "low.epx"
        _invoke("run", paths["low"], "-o", out)
        assert offline_italy.loads[-1]["data_version"] == "v1.1.0"
        assert _manifest(out)["provenance"]["population_data"]["version"] == "v1.1.0"


class TestDataVersionValidation:
    def _config(self, version):
        cfg = copy.deepcopy(MINIMAL_CONFIG)
        cfg["population"] = {"name": "Italy", "data_version": version}
        return cfg

    def test_release_tag(self):
        report = validate_config(self._config("v1.2.0"))
        assert report["valid"]
        assert not any("data_version" in w for w in report["warnings"])

    def test_branch_warns(self):
        report = validate_config(self._config("main"))
        assert report["valid"]
        assert any("not a release tag" in w for w in report["warnings"])

    @pytest.mark.parametrize("bad", ["", 1.2, None])
    def test_malformed(self, bad):
        assert not validate_config(self._config(bad))["valid"]


class TestVerify:
    def test_fresh_bundle_passes(self, bundle):
        report = _json("verify", bundle, "--offline")
        assert report["verified"] is True
        assert report["drift"] == []
        assert report["files"]["unchanged"] == 5

    def test_one_byte_of_config_yaml(self, bundle):
        with open(os.path.join(bundle, "config.yaml"), "a") as f:
            f.write(" ")
        report = _json("verify", bundle, "--offline", expect=1)
        assert report["verified"] is False
        assert report["drift"] == ["config.yaml: modified"]

    def test_edited_config_value(self, bundle):
        path = os.path.join(bundle, "config.yaml")
        with open(path) as f:
            text = f.read()
        with open(path, "w") as f:
            f.write(text.replace("recovery_rate: 0.1", "recovery_rate: 0.2"))
        report = _json("verify", bundle, "--offline", expect=1)
        assert report["config"] == "modified"

    def test_missing_and_added_files(self, bundle):
        os.remove(os.path.join(bundle, "transitions.parquet"))
        with open(os.path.join(bundle, "notes.txt"), "w") as f:
            f.write("hello")
        report = _json("verify", bundle, "--offline", expect=1)
        assert report["files"]["missing"] == ["transitions.parquet"]
        assert report["files"]["added"] == ["notes.txt"]

    def test_edited_manifest(self, bundle):
        manifest = _manifest(bundle)
        manifest["parameters_used"]["transmission_rate"] = 9.9
        with open(os.path.join(bundle, "manifest.json"), "w") as f:
            json.dump(manifest, f)
        report = _json("verify", bundle, "--offline", expect=1)
        assert report["manifest"] == "modified"

    def test_figures_are_not_drift(self, bundle):
        fig, ax = plt.subplots()
        ax.plot([0, 1])
        fig.savefig(os.path.join(bundle, "figures", "curve.png"))
        plt.close(fig)
        add_figure_to_manifest(bundle, "curve.png", "a curve")
        assert _json("verify", bundle, "--offline")["verified"] is True

    def test_unsealed_bundle_cannot_be_verified(self, tmp_path):
        bundle = _make_bundle(tmp_path)  # saved through the Python API
        report = _json("verify", bundle, expect=1)
        assert report["verified"] is False
        assert "no content hashes" in report["reason"]

    def test_population_data_drift(self, tmp_path, offline_italy):
        out = tmp_path / "low.epx"
        _invoke("run", write_case_study(tmp_path, n_simulations=2)["low"], "-o", out)
        assert _json("verify", out)["population_data"]["status"] == "unchanged"
        url = (
            "https://raw.githubusercontent.com/epistorm/epydemix-data/v1.2.0/"
            "data/Italy/contact_matrices/mistry_2021/contacts_matrix_school.csv"
        )
        offline_italy.files[url] = b"a moved tag"
        report = _json("verify", out, expect=1)
        assert report["population_data"]["modified"] == [
            "data/Italy/contact_matrices/mistry_2021/contacts_matrix_school.csv"
        ]
        offline = _json("verify", out, "--offline")
        assert offline["population_data"]["status"] == "unchecked"


class TestReplay:
    def test_fixed_seed_run_reproduces_exactly(self, bundle):
        report = _json("replay", bundle, "--tol", "0")
        assert report["reproduced"] is True
        assert report["max_rel_diff"] == 0.0
        assert report["n_compared"] > 0
        assert report["replayed_to"] is None  # temporary bundle removed

    def test_keeps_the_replay_when_asked(self, bundle, tmp_path):
        out = tmp_path / "again.epx"
        report = _json("replay", bundle, "-o", out)
        assert report["replayed_to"] == str(out.resolve())
        assert _manifest(out)["provenance"]["replay_of"] == os.path.abspath(bundle)

    def test_calibration_and_projection(self, tmp_path):
        cal = _run_calibration(tmp_path)
        assert _json("replay", cal, "--tol", "0")["max_rel_diff"] == 0.0
        proj = tmp_path / "proj.epx"
        _invoke("project", cal, "-o", proj)
        assert _json("replay", proj, "--tol", "0")["max_rel_diff"] == 0.0

    def test_detects_a_difference(self, bundle):
        path = os.path.join(bundle, "summary.json")
        with open(path) as f:
            summary = json.load(f)
        summary["raw"]["variables"]["Infected_total"]["peak"]["0.5"] *= 1.5
        with open(path, "w") as f:
            json.dump(summary, f)
        report = _json("replay", bundle, expect=1)
        assert report["reproduced"] is False
        assert report["worst"] == "variables.Infected_total.peak.0.5"
        assert report["max_rel_diff"] == pytest.approx(1 / 3)

    def test_projection_without_its_parent(self, tmp_path):
        cal = _run_calibration(tmp_path)
        proj = tmp_path / "proj.epx"
        _invoke("project", cal, "-o", proj)
        shutil.rmtree(cal)
        result = _invoke("replay", proj, expect=1)
        assert "REPLAY_ERROR" in result.output


class TestCompareSummaries:
    def test_dates_scale_by_simulation_length(self):
        a = {"raw": {"v": {"peak_date": {"0.5": "2026-01-10"}}}}
        b = {"raw": {"v": {"peak_date": {"0.5": "2026-01-20"}}}}
        assert compare_summaries(a, b, n_days=100)["max_rel_diff"] == pytest.approx(0.1)

    def test_missing_numbers_count(self):
        a = {"raw": {"x": 1.0, "y": 2.0}}
        b = {"raw": {"x": 1.0}}
        result = compare_summaries(a, b)
        assert result["max_rel_diff"] == 1.0
        assert result["worst"] == "y"
