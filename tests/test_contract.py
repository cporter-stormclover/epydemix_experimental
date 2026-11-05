"""The agent contract is generated from the code and matches what the CLI prints."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from epydemix import _contract
from epydemix.cli.main import cli
from epydemix.cli.responses import (
    CompareResult,
    DeriveResult,
    ErrorResult,
    ManifestResult,
    ReplayResult,
    SummaryFile,
    SweepResult,
    ValidateResult,
    VerifyResult,
    conforms,
)

from .test_cli import MINIMAL_CONFIG, _run_calibration, _write_yaml
from .test_sweep import CLOSURE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_contract  # noqa: E402


def _stdout(*args, expect=0):
    result = CliRunner().invoke(cli, [str(a) for a in args])
    assert result.exit_code == expect, result.output
    return json.loads(result.stdout)


class TestGenerated:
    def test_agent_md_is_up_to_date(self):
        assert build_contract.main(["--check"]) == 0

    def test_build_is_deterministic(self):
        first = build_contract.build()
        second = build_contract.build()
        assert first == second

    def test_under_the_token_ceiling(self):
        text, _, tokens = build_contract.build()
        assert tokens < build_contract.TOKEN_CEILING
        assert _contract.CONTRACT_TOKENS == tokens

    def test_every_command_is_documented(self):
        text = (ROOT / "AGENT.md").read_text()
        for name in cli.commands:
            assert f"**`epydemix {name}" in text, name

    def test_a_changed_flag_makes_the_check_fail(self, monkeypatch):
        import click

        option = next(p for p in cli.commands["run"].params if "--output" in p.opts)
        monkeypatch.setattr(option, "help", "A different help text.")
        assert isinstance(option, click.Option)
        assert build_contract.main(["--check"]) == 1

    def test_the_script_runs(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "build_contract.py"), "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


class TestContractVersion:
    def test_manifests_record_it(self, tmp_path):
        manifest = _stdout(
            "run", _write_yaml(tmp_path, MINIMAL_CONFIG), "-o", tmp_path / "r.epx"
        )
        assert manifest["provenance"]["contract_version"] == _contract.CONTRACT_VERSION

    def test_it_is_the_hash_of_agent_md(self):
        import hashlib

        digest = hashlib.sha256((ROOT / "AGENT.md").read_text().encode()).hexdigest()
        assert _contract.CONTRACT_VERSION == f"sha256:{digest}"


class TestOutputsConform:
    def test_validate(self, tmp_path):
        out = _stdout("validate", _write_yaml(tmp_path, MINIMAL_CONFIG))
        assert conforms(out, ValidateResult) == []

    def test_run_manifest_and_summary(self, tmp_path):
        bundle = tmp_path / "r.epx"
        manifest = _stdout("run", _write_yaml(tmp_path, MINIMAL_CONFIG), "-o", bundle)
        assert conforms(manifest, ManifestResult) == []
        with open(bundle / "summary.json") as f:
            assert conforms(json.load(f), SummaryFile) == []
        assert conforms(_stdout("verify", bundle, "--offline"), VerifyResult) == []
        assert conforms(_stdout("replay", bundle), ReplayResult) == []
        assert conforms(_stdout("compare", bundle), CompareResult) == []

    def test_calibration_manifest_and_summary(self, tmp_path):
        bundle = Path(_run_calibration(tmp_path))
        with open(bundle / "manifest.json") as f:
            assert conforms(json.load(f), ManifestResult) == []
        with open(bundle / "summary.json") as f:
            assert conforms(json.load(f), SummaryFile) == []

    def test_derive_and_sweep(self, tmp_path):
        assert (
            conforms(
                _stdout(
                    "derive",
                    _write_yaml(
                        tmp_path, {k: v for k, v in CLOSURE.items() if k != "sweep"}
                    ),
                ),
                DeriveResult,
            )
            == []
        )
        small = json.loads(json.dumps(CLOSURE))
        small["sweep"]["over"][0]["range"]["stop"] = "2026-12-08"
        out = _stdout(
            "sweep", _write_yaml(tmp_path, small, "s.yaml"), "-o", tmp_path / "s.epx"
        )
        assert conforms(out, SweepResult) == []
        assert (
            conforms(
                _stdout("validate", _write_yaml(tmp_path, small, "s.yaml")),
                ValidateResult,
            )
            == []
        )

    def test_errors(self, tmp_path):
        result = CliRunner().invoke(
            cli, ["derive", _write_yaml(tmp_path, MINIMAL_CONFIG)]
        )
        assert result.exit_code == 1
        error = json.loads(result.output[result.output.index("{") :])
        assert conforms(error, ErrorResult) == []

    @pytest.mark.parametrize("model", [ValidateResult, SweepResult, VerifyResult])
    def test_conforms_reports_problems(self, model):
        assert conforms({"unexpected_key": 1}, model)
