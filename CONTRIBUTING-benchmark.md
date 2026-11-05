# Contributing a benchmark task

A task is a directory under `benchmark/tasks/`:

```
benchmark/tasks/<task>/
  prompt.md        # the question, as a user would ask it
  meta.yaml        # id, title, paper_task, description, needs_population, timeout_s
  grader.yaml      # the checks
  data/            # optional: files copied into the agent's workspace
  reference/       # private: solution.sh, report.md and the configs it uses
```

## The prompt

Write it as a person would, with every number the answer depends on and
nothing that gives away the solution. Fix the names of the output bundles the
grader reads (`Save it as outputs/baseline.epx`), and the names of
compartments if checks read them. The harness adds a footer asking the agent
to write bundles under `./outputs/` and to finish with a report.

## meta.yaml

```yaml
id: t6_my_task              # the directory name
title: One line
paper_task: null            # or T1..T5 / "case study"
description: >
  What the task exercises.
needs_population: true      # uses a named population (downloaded at run time)
timeout_s: 3600
public_reference: false     # true only for the self-test
```

## grader.yaml

```yaml
checks:
  - {name: r0, type: config_equals, path: r0, expect: 2.5, rtol: 0.02}
  - {name: rate, type: config_equals, path: "transition:S->E@I", expect: 0.3, rtol: 0.01}
  - {name: peak, type: metric_within, bundle: "outputs/baseline.epx",
     metric: variables.Infected_total.peak.0.5, reference: "outputs/baseline.epx", rtol: 0.1}
  - {name: ranking, type: ranking_preserved, metric: attack_rate.0.5,
     items: ["outputs/a.epx", "outputs/b.epx"], order: reference}
  - {name: provenance, type: provenance_complete}
  - {name: assumptions, type: assumptions_declared}
  - {name: report, type: report_mentions, pattern: "(?i)attack rate"}
```

Every task should check `provenance_complete` and `assumptions_declared`.
Prefer checks that do not depend on how the agent names things: `r0` (the R0
a config implies) and `transition:SRC->DST@AGENT` (the evaluated rate of a
transition) work whatever the parameters are called; `bundle_type`
(`CalibrationResults`) and `command` (`project`) select bundles without
relying on their names. Set tolerances from the Monte Carlo spread of the
reference (an agent may choose another seed or more simulations), not tighter.
`harness/checks.py` documents every check.

## The reference solution

`reference/solution.sh` runs in a workspace (as `bash -e`) and writes the
outputs; it finds its configs with `REF="$(dirname "$0")"`. Seed every
config, so the reference outputs are the same at every run. `report.md` is
the answer an agent should give, and must satisfy the `report_mentions`
checks. Declare an `assumptions` entry for every value the configs set, as a
careful agent would.

Check the task before committing:

```bash
python benchmark/run.py run --agent reference --tasks <task> --venv shared --keep
```

It must pass every check. Then break the reference on purpose (a wrong
parameter) and make sure the checks you expect, and only those, fail.

## Keeping references private

References are committed only encrypted. With the maintainers' key:

```bash
export EPX_BENCHMARK_KEY=...                  # never commit it
python benchmark/run.py references unpack     # tasks/*/reference/ from the archive
# ... edit or add references ...
python benchmark/run.py references pack       # rewrite benchmark/references.enc
```

`tasks/*/reference/` is git-ignored (except the self-test's). If a reference
is ever committed in the clear, or the key leaks, rotate: change the task's
numbers (and so its answers), re-pack with a new key, and note the rotation
in the changelog.
