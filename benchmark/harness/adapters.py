"""Agent adapters: run a terminal agent on a prompt in a workspace.

Each adapter returns an :class:`AgentResult` with the agent's final message
and whatever it reports of cost, turns and tokens (``None`` when the agent
does not report it). The command lines of the ``claude-code``, ``codex`` and
``opencode`` adapters follow those tools' documented non-interactive modes;
they are not exercised by this repository's tests (which use ``shell`` and
``reference``), so check them against the installed versions before a run.
"""

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

ALLOWED_TOOLS = ["Bash", "Read", "Write", "Edit", "Glob", "Grep"]


@dataclass
class AgentResult:
    final_message: str = ""
    exit_code: int = 0
    cost_usd: Optional[float] = None
    turns: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    log: List[Dict[str, Any]] = field(default_factory=list)
    stderr: str = ""


def _run(cmd, cwd, env, timeout, stdin=None):
    try:
        return subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            shell=isinstance(cmd, str),
        )
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(
            cmd,
            124,
            (e.stdout or b"").decode()
            if isinstance(e.stdout, bytes)
            else (e.stdout or ""),
            "timeout",
        )


def _json_lines(text: str) -> List[Dict[str, Any]]:
    events = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("{"):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return events


class Adapter:
    name = "adapter"
    runtime = "unknown"
    model: Optional[str] = None

    def run(
        self,
        prompt: str,
        cwd: Path,
        env: Dict[str, str],
        tools: List[str],
        timeout: int,
    ) -> AgentResult:
        raise NotImplementedError


class ShellAdapter(Adapter):
    """Any command that reads the prompt on stdin and prints its answer.

    If the last line of its output is a JSON object with an ``epx_metrics``
    key, its ``cost_usd``, ``turns``, ``input_tokens`` and ``output_tokens``
    are recorded and the line is dropped from the final message.
    """

    name = "shell"
    runtime = "shell"

    def __init__(self, command: str):
        self.command = command

    def run(self, prompt, cwd, env, tools, timeout):
        proc = _run(self.command, cwd, env, timeout, stdin=prompt)
        out = proc.stdout or ""
        result = AgentResult(
            exit_code=proc.returncode, stderr=(proc.stderr or "")[-5000:]
        )
        lines = out.rstrip("\n").split("\n")
        if lines and lines[-1].startswith("{"):
            try:
                metrics = json.loads(lines[-1]).get("epx_metrics")
            except json.JSONDecodeError:
                metrics = None
            if isinstance(metrics, dict):
                for key in ("cost_usd", "turns", "input_tokens", "output_tokens"):
                    setattr(result, key, metrics.get(key))
                lines = lines[:-1]
        result.final_message = "\n".join(lines)[-20000:]
        return result


class ReferenceAdapter(Adapter):
    """Runs a task's reference solution: the benchmark's self-check."""

    name = "reference"
    runtime = "reference"

    def __init__(self, task, mutate=None):
        self.task = task
        self.mutate = mutate  # callable(workspace) applied before solution.sh

    def run(self, prompt, cwd, env, tools, timeout):
        import shutil

        ref = self.task.reference
        if not (ref / "solution.sh").exists():
            return AgentResult(
                exit_code=2, stderr=f"{self.task.name} has no reference solution"
            )
        for path in ref.iterdir():
            if path.name not in ("report.md",):
                target = Path(cwd) / "reference_files" / path.name
                target.parent.mkdir(exist_ok=True)
                (shutil.copytree if path.is_dir() else shutil.copy)(path, target)
        if self.mutate:
            self.mutate(Path(cwd))
        proc = _run(
            ["bash", "-e", str(Path(cwd) / "reference_files" / "solution.sh")],
            cwd,
            env,
            timeout,
        )
        report = (ref / "report.md").read_text() if (ref / "report.md").exists() else ""
        return AgentResult(
            final_message=report,
            exit_code=proc.returncode,
            stderr=(proc.stderr or "")[-5000:],
            cost_usd=0.0,
            turns=0,
        )


class ClaudeCodeAdapter(Adapter):
    """``claude -p`` with stream-json output (cost, turns and usage reported)."""

    name = "claude-code"
    runtime = "claude-code"

    def __init__(self, model=None, binary="claude"):
        self.model, self.binary = model, binary

    def run(self, prompt, cwd, env, tools, timeout):
        cmd = [self.binary, "-p", prompt, "--output-format", "stream-json", "--verbose"]
        cmd += ["--allowedTools", ",".join(tools)]
        if self.model:
            cmd += ["--model", self.model]
        proc = _run(cmd, cwd, env, timeout)
        events = _json_lines(proc.stdout or "")
        result = AgentResult(
            exit_code=proc.returncode, log=events, stderr=(proc.stderr or "")[-5000:]
        )
        for event in events:
            if event.get("type") == "result":
                result.final_message = event.get("result") or ""
                result.cost_usd = event.get("total_cost_usd")
                result.turns = event.get("num_turns")
                usage = event.get("usage") or {}
                result.input_tokens = sum(
                    usage.get(k) or 0
                    for k in (
                        "input_tokens",
                        "cache_read_input_tokens",
                        "cache_creation_input_tokens",
                    )
                )
                result.output_tokens = usage.get("output_tokens")
        return result


class CodexAdapter(Adapter):
    """``codex exec --json`` (JSONL events; usage per turn, no cost)."""

    name = "codex"
    runtime = "codex"

    def __init__(self, model=None, binary="codex"):
        self.model, self.binary = model, binary

    def run(self, prompt, cwd, env, tools, timeout):
        cmd = [self.binary, "exec", "--json", "--skip-git-repo-check", "--full-auto"]
        if self.model:
            cmd += ["--model", self.model]
        cmd.append(prompt)
        proc = _run(cmd, cwd, env, timeout)
        events = _json_lines(proc.stdout or "")
        result = AgentResult(
            exit_code=proc.returncode, log=events, stderr=(proc.stderr or "")[-5000:]
        )
        turns, tokens_in, tokens_out = 0, 0, 0
        for event in events:
            if event.get("type") == "turn.completed":
                turns += 1
                usage = event.get("usage") or {}
                tokens_in += usage.get("input_tokens") or 0
                tokens_out += usage.get("output_tokens") or 0
            item = event.get("item") or {}
            if (
                event.get("type") == "item.completed"
                and item.get("type") == "agent_message"
            ):
                result.final_message = item.get("text") or result.final_message
        result.turns, result.input_tokens, result.output_tokens = (
            turns,
            tokens_in,
            tokens_out,
        )
        return result


class OpenCodeAdapter(Adapter):
    """``opencode run --format json`` (assistant text and token counts if present)."""

    name = "opencode"
    runtime = "opencode"

    def __init__(self, model=None, binary="opencode"):
        self.model, self.binary = model, binary

    def run(self, prompt, cwd, env, tools, timeout):
        cmd = [self.binary, "run", "--format", "json"]
        if self.model:
            cmd += ["--model", self.model]
        cmd.append(prompt)
        proc = _run(cmd, cwd, env, timeout)
        events = _json_lines(proc.stdout or "")
        result = AgentResult(
            exit_code=proc.returncode, log=events, stderr=(proc.stderr or "")[-5000:]
        )
        texts, tokens_in, tokens_out, cost = [], 0, 0, None
        for event in events:
            part = event.get("part") or event
            if part.get("type") == "text" and part.get("text"):
                texts.append(part["text"])
            tokens = part.get("tokens") or {}
            tokens_in += tokens.get("input") or 0
            tokens_out += tokens.get("output") or 0
            if part.get("cost") is not None:
                cost = (cost or 0.0) + float(part["cost"])
        result.final_message = texts[-1] if texts else (proc.stdout or "")[-20000:]
        result.input_tokens, result.output_tokens, result.cost_usd = (
            tokens_in,
            tokens_out,
            cost,
        )
        return result


def make_adapter(name: str, task=None, command=None, model=None) -> Adapter:
    if name == "shell":
        if not command:
            raise ValueError("the shell adapter needs --command")
        return ShellAdapter(command)
    if name == "reference":
        return ReferenceAdapter(task)
    adapters = {
        "claude-code": ClaudeCodeAdapter,
        "codex": CodexAdapter,
        "opencode": OpenCodeAdapter,
    }
    if name not in adapters:
        raise ValueError(
            f"unknown agent {name!r}; choose from shell, reference, {', '.join(adapters)}"
        )
    adapter = adapters[name](model=model)
    return adapter
