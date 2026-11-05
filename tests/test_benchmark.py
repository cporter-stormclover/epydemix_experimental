"""The correctness benchmark: graders, references, runner, scorecard.

The public self-test task runs always. The other tasks' references are
private (benchmark/references.enc); their tests run when EPX_BENCHMARK_KEY
is set, unpacking the references first. Named populations come from the
test snapshot (a sitecustomize put on the sessions' path), so nothing is
downloaded.
"""

import json
import os
import sys
import tarfile
from io import BytesIO
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))

from harness import references  # noqa: E402
from harness.adapters import ReferenceAdapter, ShellAdapter  # noqa: E402
from harness.checks import Context, _path_parts, run_checks  # noqa: E402
from harness.runner import (  # noqa: E402
    interleaved,
    reference_outputs,
    run_benchmark,
    run_task,
)
from harness.scorecard import render_markdown, scorecard  # noqa: E402
from harness.tasks import ARCHIVE, TASKS_DIR, load_task, load_tasks  # noqa: E402

from .test_contract_examples import ITALY, SITECUSTOMIZE  # noqa: E402

PRIVATE = [
    "t1_seirhd",
    "t2_school_closure",
    "t3_calibration",
    "t4_lockdown_timing",
    "t5_asymptomatic",
    "case_study_vaccination",
]


@pytest.fixture(scope="module")
def offline_env(tmp_path_factory):
    site = tmp_path_factory.mktemp("site")
    (site / "sitecustomize.py").write_text(SITECUSTOMIZE)
    path = os.environ.get("PYTHONPATH")
    return {
        "PYTHONPATH": os.pathsep.join([str(site)] + ([path] if path else [])),
        "EPX_TEST_ITALY": str(ITALY),
    }


def _set_parameter(name, value):
    """A mutation of a reference solution: a deliberately wrong parameter."""

    def mutate(workspace: Path):
        for path in (workspace / "reference_files").glob("*.yaml"):
            config = yaml.safe_load(path.read_text())
            if name in (config.get("parameters") or {}):
                config["parameters"][name] = value
                path.write_text(yaml.safe_dump(config, sort_keys=False))

    return mutate


def _session(task, tmp_path, offline_env, mutate=None):
    cache = tmp_path / "references"
    reference = reference_outputs(task, cache, "shared", offline_env)
    return run_task(
        task,
        ReferenceAdapter(task, mutate=mutate),
        "contract",
        0,
        tmp_path / "runs",
        env_mode="shared",
        reference=reference,
        replay=True,
        extra_env=offline_env,
    )


class TestTasks:
    def test_every_task_is_complete(self):
        tasks = load_tasks()
        assert {t.name for t in tasks} == set(PRIVATE) | {"selftest"}
        for task in tasks:
            assert task.prompt.strip()
            assert task.meta["id"] == task.name
            assert task.checks
            types = {c["type"] for c in task.checks}
            assert {"provenance_complete", "assumptions_declared"} <= types, task.name

    def test_paper_tasks_are_seeded(self):
        assert {load_task(n).meta["paper_task"] for n in PRIVATE} == {
            "T1",
            "T2",
            "T3",
            "T4",
            "T5",
            "case study",
        }

    def test_private_references_are_not_committed(self):
        ignored = (ROOT / ".gitignore").read_text()
        assert "benchmark/tasks/*/reference/" in ignored
        assert "!benchmark/tasks/selftest/reference/" in ignored
        assert ARCHIVE.exists()


class TestSelfTest:
    def test_reference_passes_every_check(self, tmp_path, offline_env):
        row = _session(load_task("selftest"), tmp_path, offline_env)
        assert row["pass_rate"] == 1.0, row["failed_checks"]
        assert row["reproduced"] == 1.0
        assert row["exit_code"] == 0

    def test_wrong_transmission_rate_fails_only_config_and_metric(
        self, tmp_path, offline_env
    ):
        task = load_task("selftest")
        row = _session(
            task, tmp_path, offline_env, _set_parameter("transmission_rate", 0.36)
        )
        failed = {c["type"] for c in row["checks"] if not c["passed"]}
        assert failed == {"config_equals", "metric_within"}
        assert row["failed_checks"] == ["transmission_rate", "peak"]


@pytest.mark.skipif(
    not os.environ.get(references.KEY_ENV),
    reason=f"the private references need {references.KEY_ENV}",
)
class TestPrivateReferences:
    @pytest.fixture(scope="class", autouse=True)
    def unpacked(self):
        references.unpack(ARCHIVE, TASKS_DIR, os.environ[references.KEY_ENV])

    def test_references_pass_every_check(self, tmp_path, offline_env):
        rows = run_benchmark(
            load_tasks(PRIVATE),
            ReferenceAdapter,
            ["contract"],
            1,
            tmp_path / "results.jsonl",
            tmp_path / "runs",
            env_mode="shared",
            extra_env=offline_env,
        )
        failures = {r["task"]: r["failed_checks"] for r in rows if not r["passed"]}
        assert failures == {}
        assert all(r["reproduced"] == 1.0 for r in rows)

    def test_wrong_beta_fails_only_config_and_metric(self, tmp_path, offline_env):
        row = _session(
            load_task("t1_seirhd"), tmp_path, offline_env, _set_parameter("beta", 0.36)
        )
        failed = {c["type"] for c in row["checks"] if not c["passed"]}
        assert failed == {"config_equals", "metric_within"}
        assert set(row["failed_checks"]) == {
            "community_transmission",
            "peak_hospital_occupancy",
            "total_deaths",
        }


class TestChecks:
    @pytest.fixture
    def workspace(self, tmp_path, offline_env):
        task = load_task("selftest")
        ref = reference_outputs(task, tmp_path / "ref", "shared", offline_env)
        return ref

    def _ctx(self, ws, message="The peak is in February."):
        return Context(ws, load_task("selftest").prompt, message, ws)

    def test_quantile_levels_stay_whole_in_paths(self):
        assert _path_parts("variables.H_total.peak.0.5") == [
            "variables",
            "H_total",
            "peak",
            "0.5",
        ]
        assert _path_parts("interventions[0].reduction") == [
            "interventions",
            "0",
            "reduction",
        ]

    def test_transition_rates_ignore_parameter_names(self, tmp_path):
        bundle = tmp_path / "outputs" / "x.epx"
        bundle.mkdir(parents=True)
        (bundle / "manifest.json").write_text(json.dumps({"type": "SimulationResults"}))
        (bundle / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "model": {
                        "type": "custom",
                        "transitions": [
                            {
                                "source": "S",
                                "target": "E",
                                "kind": "mediated",
                                "params": ["b", "I"],
                            },
                            {
                                "source": "I",
                                "target": "H",
                                "kind": "spontaneous",
                                "params": "g * p",
                            },
                        ],
                    },
                    "parameters": {"b": 0.3, "g": 0.2, "p": 0.1},
                }
            )
        )
        checks = [
            {"type": "config_equals", "path": "transition:S->E@I", "expect": 0.3},
            {
                "type": "config_equals",
                "path": "transition:I->H",
                "expect": 0.02,
                "rtol": 1e-9,
            },
            {"type": "config_equals", "path": "transition:S->E@H", "expect": 0.05},
        ]
        results = run_checks(Context(tmp_path, "", ""), checks)
        assert [r.passed for r in results] == [True, True, False]

    def test_report_and_ranking(self, workspace):
        checks = [
            {"type": "report_mentions", "pattern": "(?i)peak"},
            {"type": "report_mentions", "pattern": "(?i)deaths"},
            {
                "type": "ranking_preserved",
                "metric": "variables.Infected_total.peak.0.5",
                "items": ["outputs/sir.epx"],
                "order": "reference",
            },
            {"type": "nonsense"},
        ]
        results = run_checks(self._ctx(workspace), checks)
        assert [r.passed for r in results] == [True, False, True, False]
        assert "unknown check type" in results[3].detail

    def test_tampered_bundle_fails_provenance(self, workspace):
        config = workspace / "outputs" / "sir.epx" / "config.yaml"
        config.write_text(config.read_text() + "\n")
        result = run_checks(self._ctx(workspace), [{"type": "provenance_complete"}])[0]
        assert not result.passed and "config.yaml" in result.detail

    def test_no_outputs(self, tmp_path):
        results = run_checks(
            Context(tmp_path, "", ""),
            [
                {"type": "config_equals", "path": "parameters.x", "expect": 1},
                {"type": "provenance_complete"},
            ],
        )
        assert not any(r.passed for r in results)


class TestReferencesArchive:
    def test_round_trip(self, tmp_path):
        task = tmp_path / "tasks" / "demo" / "reference"
        task.mkdir(parents=True)
        (task / "solution.sh").write_text("echo hi\n")
        archive = tmp_path / "refs.enc"
        assert references.pack(tmp_path / "tasks", archive, "secret", ["demo"]) == [
            "demo"
        ]
        assert b"echo hi" not in archive.read_bytes()
        (task / "solution.sh").unlink()
        assert references.unpack(archive, tmp_path / "tasks", "secret") == ["demo"]
        assert (task / "solution.sh").read_text() == "echo hi\n"
        with pytest.raises(references.ReferenceError, match="wrong key"):
            references.unpack(archive, tmp_path / "tasks", "other")

    def test_refuses_paths_outside_reference_dirs(self, tmp_path):
        buffer = BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            data = b"x"
            info = tarfile.TarInfo("../../evil.sh")
            info.size = len(data)
            tar.addfile(info, BytesIO(data))
        archive = tmp_path / "evil.enc"
        archive.write_bytes(references.encrypt(buffer.getvalue(), "k"))
        with pytest.raises(references.ReferenceError, match="refusing"):
            references.unpack(archive, tmp_path / "tasks", "k")


class TestRunner:
    def test_conditions_interleave(self):
        assert list(interleaved(["a", "b"], 3)) == [
            (0, "a"),
            (0, "b"),
            (1, "b"),
            (1, "a"),
            (2, "a"),
            (2, "b"),
        ]

    def test_shell_adapter_reads_metrics(self, tmp_path):
        script = tmp_path / "agent.sh"
        script.write_text(
            "cat > /dev/null\necho 'The peak is high.'\n"
            'echo \'{"epx_metrics": {"cost_usd": 0.5, "turns": 3}}\'\n'
        )
        result = ShellAdapter(f"bash {script}").run(
            "prompt", tmp_path, dict(os.environ), [], 60
        )
        assert result.final_message == "The peak is high."
        assert (result.cost_usd, result.turns) == (0.5, 3)

    def test_session_row_and_scorecard(self, tmp_path, offline_env):
        script = tmp_path / "agent.sh"
        script.write_text("cat > prompt_received.md\necho 'No idea.'\n")
        rows = run_benchmark(
            [load_task("selftest")],
            lambda task: ShellAdapter(f"bash {script}"),
            ["contract", "no_contract"],
            1,
            tmp_path / "results.jsonl",
            tmp_path / "runs",
            env_mode="shared",
            extra_env=offline_env,
            keep=True,
        )
        assert [r["condition"] for r in rows] == ["contract", "no_contract"]
        for row in rows:
            assert row["passed"] is False and row["pass_rate"] < 0.5
            assert {
                "cost_usd",
                "turns",
                "input_tokens",
                "output_tokens",
                "wall_clock_s",
            } <= set(row)
        ws = Path(rows[0]["workspace"])
        assert (ws / "AGENT.md").exists()
        assert "./outputs/" in (ws / "prompt_received.md").read_text()
        assert not (Path(rows[1]["workspace"]) / "AGENT.md").exists()
        lines = (tmp_path / "results.jsonl").read_text().splitlines()
        assert len(lines) == 2
        table = scorecard([json.loads(line) for line in lines])
        assert {r["condition"] for r in table} == {"contract", "no_contract"}
        assert "| shell | contract | 1 |" in render_markdown(table)
