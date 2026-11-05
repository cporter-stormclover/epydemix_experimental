# Epydemix agent benchmark

A benchmark that scores whether an agent driving epydemix gets the answer
**right**, not only how much it cost. Each task is a question an epidemiologist
might ask; an agent works in a fresh workspace with the `epydemix` CLI and
leaves result bundles in `./outputs/` and a final report. A declarative grader
then checks the configs it wrote, the numbers it obtained, its provenance and
declared assumptions, and its report.

| Task | Paper | What it exercises |
|---|---|---|
| `t1_seirhd` | T1 | custom model: SEIRHD with nosocomial transmission |
| `t2_school_closure` | T2 | a sweep of 18 weekly school-closure start dates in Italy |
| `t3_calibration` | T3 | ABC calibration to incidence data, then a projection |
| `t4_lockdown_timing` | T4 | early vs late contact reduction in Italy (R0 = 2.5) |
| `t5_asymptomatic` | T5 | SEIAR from R0 and durations |
| `case_study_vaccination` | case study | qualitative vaccination rates, ranking and robustness |
| `selftest` | — | a public toy task that tests the harness itself (CI) |

The five paper tasks were reconstructed from the descriptions available when
the benchmark was written (arXiv:2609.28692 could not be consulted); their
prompts, not the paper, define them.

## Running it

```bash
python benchmark/run.py list
EPX_BENCHMARK_KEY=... python benchmark/run.py references unpack
python benchmark/run.py run --agent reference --tasks all            # self-check: 100%
python benchmark/run.py run --agent claude-code --model <model> \
    --conditions contract,no_contract --iterations 3 --out results.jsonl
python benchmark/run.py run --agent shell --command "my-agent --stdin" --tasks t1_seirhd
python benchmark/run.py score results.jsonl
```

Agents: `claude-code` (`claude -p`, stream-json), `codex` (`codex exec --json`),
`opencode` (`opencode run --format json`), `shell` (any command that reads the
prompt on stdin and prints its answer; a last line
`{"epx_metrics": {"cost_usd": ..., "turns": ...}}` reports metrics) and
`reference` (the author's solution). The first three follow those tools'
documented non-interactive modes but are not exercised by this repository's
tests: check their flags against the versions you run.

## How a session runs

1. **Fresh workspace**: the prompt, the task's `data/`, and under the
   `contract` condition `AGENT.md` and `AGENT_EXAMPLES.md` (`no_contract`
   leaves the agent with `epydemix --help`).
2. **Isolated environment** (`--venv isolated`, the default): a virtualenv in
   the workspace that sees the installed dependencies but has its own copy of
   epydemix, so nothing an agent installs or breaks leaks out.
   `EPX_AGENT_RUNTIME` / `EPX_AGENT_MODEL` are set, so every bundle's
   provenance names the agent.
3. **Allowed tools** (`--tools`, default `Bash,Read,Write,Edit,Glob,Grep`) and
   the task's timeout.
4. **Interleaved conditions**: iterations alternate the conditions' order
   (A B, B A, ...), so drift in an agent service affects both alike.
5. **Grading**: `grader.yaml`'s checks (see `harness/checks.py`):
   `config_equals`, `metric_within`, `provenance_complete`,
   `assumptions_declared`, `ranking_preserved`, `report_mentions`. Reference
   values come from running the reference solution once per invocation; it is
   seeded, so its bundles are the same every time.
6. **Reproducibility**: every output bundle is replayed (`epydemix replay`,
   relative tolerance 0.05).

Each session appends one JSON line: task, agent, model, condition, iteration,
cost, turns, tokens, wall clock, every check's result, pass rate, failed
checks and the replay outcome. `score` turns a results file into a scorecard
(pass rate, sessions fully passed, median cost, turns and wall clock,
reproducibility) per agent and condition.

## References are private

A benchmark whose answers are public stops measuring anything once they reach
training data. The reference solutions are committed only as
`benchmark/references.enc`, encrypted with a key kept outside the repository
(`EPX_BENCHMARK_KEY`); `tasks/*/reference/` is git-ignored except for the
self-test. Without the key the harness, the graders and the self-test still
run; scoring the other tasks needs the references. If the references ever
reach a public branch, rotate the tasks (see
[CONTRIBUTING-benchmark.md](../CONTRIBUTING-benchmark.md)).
