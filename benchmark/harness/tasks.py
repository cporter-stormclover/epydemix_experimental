"""Benchmark tasks: ``tasks/<name>/`` with prompt.md, grader.yaml and meta.yaml.

``data/`` (optional) is copied into the agent's workspace. ``reference/``
holds the author's solution: ``solution.sh`` (run in a workspace, it writes
``outputs/*.epx``), ``report.md`` (the answer, as an agent would write it) and
any configs it uses. References are private unless ``meta.yaml`` sets
``public_reference: true`` (see :mod:`harness.references`).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
TASKS_DIR = BENCHMARK_DIR / "tasks"
ARCHIVE = BENCHMARK_DIR / "references.enc"


@dataclass
class Task:
    name: str
    path: Path
    prompt: str
    meta: Dict[str, Any]
    checks: List[Dict[str, Any]]

    @property
    def reference(self) -> Path:
        return self.path / "reference"

    @property
    def has_reference(self) -> bool:
        return (self.reference / "solution.sh").exists()

    @property
    def public_reference(self) -> bool:
        return bool(self.meta.get("public_reference"))

    @property
    def timeout(self) -> int:
        return int(self.meta.get("timeout_s", 3600))


def load_task(name: str, tasks_dir: Optional[Path] = None) -> Task:
    path = Path(tasks_dir or TASKS_DIR) / name
    if not (path / "prompt.md").exists():
        raise FileNotFoundError(f"no task {name!r} in {path.parent}")
    with open(path / "meta.yaml") as f:
        meta = yaml.safe_load(f) or {}
    with open(path / "grader.yaml") as f:
        checks = (yaml.safe_load(f) or {}).get("checks") or []
    return Task(name, path, (path / "prompt.md").read_text(), meta, checks)


def task_names(tasks_dir: Optional[Path] = None) -> List[str]:
    root = Path(tasks_dir or TASKS_DIR)
    return sorted(p.name for p in root.iterdir() if (p / "prompt.md").exists())


def load_tasks(names=None, tasks_dir: Optional[Path] = None) -> List[Task]:
    if names in (None, "all", ["all"]):
        names = task_names(tasks_dir)
    return [load_task(n, tasks_dir) for n in names]


def private_tasks(tasks_dir: Optional[Path] = None) -> List[str]:
    return [t.name for t in load_tasks(None, tasks_dir) if not t.public_reference]
