"""Run the epydemix agent benchmark.

    python benchmark/run.py list
    python benchmark/run.py run --agent reference --tasks all --out results.jsonl
    python benchmark/run.py run --agent claude-code --model <model> \\
        --conditions contract,no_contract --iterations 3 --out results.jsonl
    python benchmark/run.py run --agent shell --command "my-agent --stdin" --tasks t1_seirhd
    python benchmark/run.py score results.jsonl
    python benchmark/run.py references unpack     # needs EPX_BENCHMARK_KEY
    python benchmark/run.py references pack       # maintainers, after editing references
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import references  # noqa: E402
from harness.adapters import ALLOWED_TOOLS, make_adapter  # noqa: E402
from harness.runner import CONDITIONS, run_benchmark  # noqa: E402
from harness.scorecard import load_rows, render_markdown, scorecard  # noqa: E402
from harness.tasks import ARCHIVE, TASKS_DIR, load_tasks, private_tasks  # noqa: E402


def cmd_list(args):
    for task in load_tasks():
        status = "reference available" if task.has_reference else "reference locked"
        print(f"{task.name:28} {task.meta.get('title', '')}  [{status}]")


def cmd_run(args):
    tasks = load_tasks(args.tasks.split(",") if args.tasks != "all" else "all")
    conditions = args.conditions.split(",")
    for c in conditions:
        if c not in CONDITIONS:
            raise SystemExit(f"unknown condition {c!r}; choose from {list(CONDITIONS)}")
    missing = [t.name for t in tasks if not t.has_reference]
    if missing:
        raise SystemExit(
            f"no reference solution for {', '.join(missing)}: run "
            f"`python benchmark/run.py references unpack` with {references.KEY_ENV} set"
        )
    out = Path(args.out)
    rows = run_benchmark(
        tasks,
        lambda task: make_adapter(
            args.agent, task=task, command=args.command, model=args.model
        ),
        conditions,
        args.iterations,
        out,
        Path(args.workdir),
        env_mode=args.venv,
        tools=args.tools.split(",") if args.tools else ALLOWED_TOOLS,
        replay=not args.no_replay,
        keep=args.keep,
    )
    print(render_markdown(scorecard(rows)))


def cmd_score(args):
    table = scorecard(load_rows(Path(args.results)))
    print(
        json.dumps(table, indent=2) if args.format == "json" else render_markdown(table)
    )


def cmd_references(args):
    secret = references.secret_from_env()
    if args.action == "pack":
        packed = references.pack(TASKS_DIR, Path(args.archive), secret, private_tasks())
        print(f"Encrypted the references of {', '.join(packed)} into {args.archive}")
    else:
        tasks = references.unpack(Path(args.archive), TASKS_DIR, secret)
        print(f"Unpacked the references of {', '.join(tasks)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="The epydemix agent benchmark.")
    sub = parser.add_subparsers(dest="command_name", required=True)
    sub.add_parser("list", help="List the tasks.").set_defaults(func=cmd_list)

    run = sub.add_parser("run", help="Run an agent on tasks and grade it.")
    run.add_argument(
        "--agent",
        required=True,
        help="claude-code, codex, opencode, shell or reference",
    )
    run.add_argument(
        "--command", help="For --agent shell: a command reading the prompt on stdin."
    )
    run.add_argument(
        "--model", help="Model to ask the agent for (recorded in the results)."
    )
    run.add_argument(
        "--tasks", default="all", help="Comma-separated task names, or all."
    )
    run.add_argument(
        "--conditions",
        default="contract",
        help="contract, no_contract (comma-separated).",
    )
    run.add_argument("--iterations", type=int, default=1)
    run.add_argument(
        "--tools", help=f"Allowed tools (default {','.join(ALLOWED_TOOLS)})."
    )
    run.add_argument("--venv", default="isolated", choices=["isolated", "shared"])
    run.add_argument(
        "--workdir", default="benchmark-runs", help="Where workspaces are created."
    )
    run.add_argument(
        "--out", default="benchmark-results.jsonl", help="JSON lines, appended."
    )
    run.add_argument("--no-replay", action="store_true", help="Skip the replay check.")
    run.add_argument("--keep", action="store_true", help="Keep the workspaces.")
    run.set_defaults(func=cmd_run)

    score = sub.add_parser("score", help="Scorecard from a results file.")
    score.add_argument("results")
    score.add_argument("--format", default="markdown", choices=["markdown", "json"])
    score.set_defaults(func=cmd_score)

    refs = sub.add_parser("references", help="Pack or unpack the encrypted references.")
    refs.add_argument("action", choices=["pack", "unpack"])
    refs.add_argument("--archive", default=str(ARCHIVE))
    refs.set_defaults(func=cmd_references)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
