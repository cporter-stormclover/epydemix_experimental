"""Run agents on benchmark tasks and grade the sessions.

Every session gets a fresh workspace (the task's data files, plus the agent
contract under the ``contract`` condition) and an isolated virtualenv, runs
one agent with an allowed-tool list and a timeout, and yields one JSON row:
cost, turns, tokens, wall clock, the grader's checks and whether the output
bundles replay. Conditions are interleaved (ABAB...), so that drift in the
agent service affects them alike.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .adapters import ALLOWED_TOOLS, Adapter, AgentResult, ReferenceAdapter
from .checks import Context, find_bundles, run_checks
from .tasks import BENCHMARK_DIR, Task

REPO = BENCHMARK_DIR.parent

CONDITIONS = {
    "contract": {
        "files": ["AGENT.md", "AGENT_EXAMPLES.md"],
        "preamble": "Read AGENT.md, the guide to the epydemix CLI, before you start.",
    },
    "no_contract": {
        "files": [],
        "preamble": "The epydemix command-line tool is installed (`epydemix --help`).",
    },
}

FOOTER = (
    "\n\n---\nWork in the current directory. Write every result bundle under "
    "./outputs/ (for example ./outputs/baseline.epx). Finish with a short "
    "report that answers the question."
)


def assemble_prompt(task: Task, condition: str) -> str:
    return CONDITIONS[condition]["preamble"] + "\n\n" + task.prompt.strip() + FOOTER


def fresh_workspace(task: Task, condition: str, root: Path) -> Path:
    if condition not in CONDITIONS:
        raise ValueError(
            f"unknown condition {condition!r}; choose from {list(CONDITIONS)}"
        )
    root.mkdir(parents=True, exist_ok=True)
    ws = Path(tempfile.mkdtemp(prefix=f"{task.name}-{condition}-", dir=root))
    (ws / "outputs").mkdir()
    (ws / "prompt.md").write_text(task.prompt)
    if (task.path / "data").is_dir():
        shutil.copytree(task.path / "data", ws / "data")
    for name in CONDITIONS[condition]["files"]:
        shutil.copy(REPO / name, ws / name)
    return ws


def isolated_env(
    ws: Path, mode: str = "isolated", extra: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """Environment variables for a session.

    ``isolated``: a virtualenv in the workspace, seeing the installed
    dependencies (``--system-site-packages``) but with its own epydemix,
    installed from this repository without network access, so whatever the
    agent installs or breaks stays in the workspace. ``shared``: the current
    interpreter (fast; for tests and debugging).
    """
    env = dict(os.environ)
    if mode == "isolated":
        venv = ws / ".venv"
        subprocess.run(
            [sys.executable, "-m", "venv", "--system-site-packages", str(venv)],
            check=True,
            capture_output=True,
        )
        bindir = venv / ("Scripts" if os.name == "nt" else "bin")
        subprocess.run(
            [
                str(bindir / "python"),
                "-m",
                "pip",
                "install",
                "--quiet",
                "--no-deps",
                "--no-build-isolation",
                str(REPO),
            ],
            check=True,
            capture_output=True,
        )
        env["VIRTUAL_ENV"] = str(venv)
        env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "")
    elif mode == "shared":
        env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    else:
        raise ValueError(f"unknown venv mode {mode!r}")
    env.update(extra or {})
    return env


def replay_all(ws: Path, env: Dict[str, str], tol: float = 0.05) -> Dict[str, Any]:
    """Replay every output bundle (a sweep through its first point)."""
    results = {}
    for bundle in find_bundles(ws, "outputs/*.epx"):
        with open(bundle / "manifest.json") as f:
            manifest = json.load(f)
        target = bundle
        if manifest.get("type") == "SweepResults":
            if not manifest.get("points"):
                continue
            target = bundle / manifest["points"][0]["bundle"]
        proc = subprocess.run(
            ["epydemix", "replay", str(target), "--tol", str(tol)],
            cwd=ws,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            report = json.loads(proc.stdout)
            results[bundle.name] = bool(report.get("reproduced"))
        except json.JSONDecodeError:
            results[bundle.name] = False
    rate = sum(results.values()) / len(results) if results else None
    return {"bundles": results, "rate": rate}


def reference_outputs(
    task: Task, cache: Path, env_mode: str, extra_env=None
) -> Optional[Path]:
    """Run (once, cached) the task's reference solution; return its workspace."""
    if not task.has_reference:
        return None
    target = cache / task.name
    if (target / "outputs").exists() and (target / ".complete").exists():
        return target
    shutil.rmtree(target, ignore_errors=True)
    ws = fresh_workspace(task, "no_contract", cache)
    env = isolated_env(ws, env_mode, extra_env)
    result = ReferenceAdapter(task).run(
        assemble_prompt(task, "no_contract"), ws, env, ALLOWED_TOOLS, task.timeout
    )
    if result.exit_code != 0:
        raise RuntimeError(f"reference solution of {task.name} failed: {result.stderr}")
    ws.rename(target)
    (target / ".complete").write_text("")
    return target


def run_task(
    task: Task,
    adapter: Adapter,
    condition: str,
    iteration: int,
    root: Path,
    env_mode: str = "isolated",
    tools: Optional[List[str]] = None,
    reference: Optional[Path] = None,
    replay: bool = True,
    extra_env: Optional[Dict[str, str]] = None,
    keep: bool = False,
) -> Dict[str, Any]:
    """One session: workspace, agent, grading. Returns its JSON row."""
    tools = tools or ALLOWED_TOOLS
    ws = fresh_workspace(task, condition, root)
    env = isolated_env(
        ws,
        env_mode,
        {
            "EPX_AGENT_RUNTIME": adapter.runtime,
            **({"EPX_AGENT_MODEL": adapter.model} if adapter.model else {}),
            **(extra_env or {}),
        },
    )
    prompt = assemble_prompt(task, condition)
    start = time.monotonic()
    result: AgentResult = adapter.run(prompt, ws, env, tools, task.timeout)
    wall = time.monotonic() - start
    (ws / "final_message.md").write_text(result.final_message or "")
    with open(ws / "agent_log.jsonl", "w") as f:
        for event in result.log:
            f.write(json.dumps(event) + "\n")

    ctx = Context(ws, task.prompt, result.final_message, reference)
    checks = run_checks(ctx, task.checks)
    reproduced = replay_all(ws, env) if replay else {"bundles": {}, "rate": None}
    row = {
        "task": task.name,
        "agent": adapter.name,
        "model": adapter.model,
        "condition": condition,
        "iteration": iteration,
        "exit_code": result.exit_code,
        "cost_usd": result.cost_usd,
        "turns": result.turns,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "wall_clock_s": round(wall, 2),
        "tools": tools,
        "pass_rate": sum(c.passed for c in checks) / len(checks) if checks else None,
        "passed": all(c.passed for c in checks),
        "failed_checks": [c.name for c in checks if not c.passed],
        "checks": [c.to_dict() for c in checks],
        "reproduced": reproduced["rate"],
        "replay": reproduced["bundles"],
        "workspace": str(ws) if keep else None,
    }
    if not keep:
        shutil.rmtree(ws, ignore_errors=True)
    return row


def interleaved(conditions: List[str], iterations: int):
    """``(iteration, condition)`` pairs alternating the conditions' order."""
    for i in range(iterations):
        order = conditions if i % 2 == 0 else list(reversed(conditions))
        for condition in order:
            yield i, condition


def run_benchmark(
    tasks: List[Task],
    adapter_for: Callable[[Task], Adapter],
    conditions: List[str],
    iterations: int,
    out: Path,
    root: Path,
    env_mode: str = "isolated",
    tools: Optional[List[str]] = None,
    replay: bool = True,
    extra_env: Optional[Dict[str, str]] = None,
    keep: bool = False,
) -> List[Dict[str, Any]]:
    """Run every task, condition and iteration; append rows to ``out`` (JSONL)."""
    rows = []
    cache = root / "_references"
    for task in tasks:
        reference = reference_outputs(task, cache, env_mode, extra_env)
        for iteration, condition in interleaved(conditions, iterations):
            row = run_task(
                task,
                adapter_for(task),
                condition,
                iteration,
                root,
                env_mode=env_mode,
                tools=tools,
                reference=reference,
                replay=replay,
                extra_env=extra_env,
                keep=keep,
            )
            rows.append(row)
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
    return rows
