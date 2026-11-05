# Design: a thin MCP server over the epydemix CLI

Status: **design only, not built.** The improvement plan says to build this
only once the benchmark (`benchmark/`) shows demand from agents that cannot
run shell commands. The benchmark exists, but it has not yet been run against
any real agent, so there is no such evidence. This document is the "design
PR"; the build is gated on the criterion below.

## Goal

Agents without a shell (chat clients, hosted assistants that speak only
MCP) get the same workflow as terminal agents: discover, write a config,
validate, run, sweep, calibrate, project, inspect, compare, verify, replay,
with identical semantics, and with no state on the server.

## When to build it

Build it when at least one of these holds:

1. A benchmark run includes an MCP-only client condition (or a terminal
   agent restricted to non-shell tools) and that condition's pass rate is
   limited by tool access rather than by modeling errors, i.e. sessions fail
   because the agent cannot run `epydemix`, not because it configures the
   model wrongly.
2. Users ask for it for a specific MCP-only client.

Until then, a terminal agent with the CLI and `AGENT.md` covers the
workflow, and every new surface adds a contract to keep in sync.

## Principles

- **The CLI stays the only implementation.** Each MCP tool runs the CLI as a
  subprocess and returns its stdout JSON. No second code path computes
  results, so outputs, errors and provenance are identical.
- **Stateless.** No sessions, caches or background processes. Every call is
  a function of its arguments and the files in the workspace, as every CLI
  command already is.
- **Generated, not hand-written.** Tool names, descriptions and argument
  schemas come from the click command tree, the same source `AGENT.md`'s CLI
  reference is generated from (`tools/build_contract.py`), so CLI, contract
  and MCP cannot drift.
- **Confined.** Every path an agent passes, or a config refers to, resolves
  inside one workspace root.

## Tools

One tool per CLI command, named `epydemix_<command>`:

| Tool | Arguments (from click) | Notes |
|---|---|---|
| `epydemix_models`, `epydemix_populations` | — | |
| `epydemix_schema` | `model_name`, `format`, `waning_immunity`, `vaccination`, `outcome` | `model_name` may be a workspace config path |
| `epydemix_defaults` | `disease?` | |
| `epydemix_validate` | `config_path` or `config_inline`, `prompt?`, `upgrade` | |
| `epydemix_derive` | `config_path` or `config_inline`, `raw` | |
| `epydemix_run` | `config_path` or `config_inline`, `output` | |
| `epydemix_sweep` | `config_path` or `config_inline`, `output`, `workers` | long-running (see Limits) |
| `epydemix_calibrate` | `config_path` or `config_inline`, `output` | long-running |
| `epydemix_project` | `calibration_bundle`, `config?` or `config_inline?`, `output` | |
| `epydemix_inspect` | `bundle_path`, `command`, `variables`, `quantiles`, `start`, `end`, `resample`, `round`, `raw`, `generation`, `format` | |
| `epydemix_compare` | `bundles` (array), `metrics`, `variables`, `names`, `round`, `raw`, `baseline` | |
| `epydemix_verify` | `bundle_path`, `offline` | |
| `epydemix_replay` | `bundle_path`, `tol`, `output?` | |

Schema generation maps click parameters to JSON Schema: an argument becomes a
required property (`nargs=-1` becomes an array); an option becomes an optional
property with its type (`int`, `float`, `string`, `boolean` for flags,
`enum` for `click.Choice`) and default; the help text becomes the
description (the command's first help paragraph becomes the tool's).

`config_inline` (a YAML string) is accepted wherever a command takes a config
file: the server writes it to `<root>/.mcp/<uuid>.yaml` and passes that path.
The file is deliberately left in place: the bundle's provenance records the
config path, and its hash is in the bundle anyway. Relative `base:` and data
paths in an inline config resolve from `<root>/.mcp/`, so they must name
files in the workspace (e.g. `../base.yaml`).

## Resources

Read-only, under `epydemix://<bundle path relative to the root>/...`:

- `summary.json`, `manifest.json`, `config.yaml`;
- `figures/<name>` (image MIME types from the extension).

Listing walks the root for directories holding a `manifest.json`. Nothing is
written through resources.

## Path confinement

The reference pseudocode compared string prefixes, which accepts
`/work/root-evil` for a root of `/work/root`. Instead:

```python
ROOT = Path(os.environ["EPX_WORKSPACE"]).resolve(strict=True)

def confine(path: str) -> Path:
    candidate = (ROOT / path).resolve()      # resolves symlinks and ".."
    if candidate != ROOT and ROOT not in candidate.parents:
        raise ToolError(f"{path!r} is outside the workspace")
    return candidate
```

Every argument that is a path (`config_path`, `prompt`, `output`, `bundle_path`,
`calibration_bundle`, `config`, every item of `bundles`) goes through
`confine`. Because a config can name other files (`base:`, `observed_data`,
`schedule`, a sweep `reference`), the server also validates those after
loading the config, and runs the CLI with the workspace as its working
directory. `../../etc/passwd` is rejected before any subprocess starts.

## Execution

```python
def call(command, arguments):
    argv = ["epydemix", command, *to_argv(command, confined(arguments))]
    proc = subprocess.run(argv, cwd=ROOT, env=scrubbed_env(), capture_output=True,
                          text=True, timeout=TIMEOUTS.get(command, 600), check=False)
    if proc.returncode == 0:
        return json.loads(proc.stdout)
    error = parse_error(proc.stderr) or {"code": "TOOL_ERROR", "message": proc.stderr[-2000:]}
    raise ToolError(error)          # the CLI's structured error, unchanged
```

`verify` and `replay` exit 1 on drift or a failed reproduction while still
printing a report; for them the server returns the report with `ok: false`
rather than an error. No `shell=True`. The environment passed on is scrubbed
to `PATH`, `HOME`, the proxy variables and `EPX_*`. `EPX_AGENT_RUNTIME` is set
to `mcp:<client name>`, so bundles record that they were produced over MCP.

## Limits

- Calibrations and sweeps can take minutes. A stateless server cannot return a
  job id to poll, so these calls are synchronous with a generous timeout
  (configurable per command). If clients need progress, the MCP progress
  notification can relay the CLI's stderr lines; that keeps the server
  stateless.
- Inline configs are limited to 256 KiB, and outputs returned by a tool to
  1 MiB (a larger `inspect` result tells the agent to narrow it with
  `--start/--end` or `--variables`).

## Tests (when built)

- The tool list and every tool's schema are generated from the click tree,
  and a test fails if a CLI option has no tool property (same drift check as
  `build_contract.py --check`).
- The vaccination case study runs end to end through MCP tools from an
  in-process MCP client with no shell access, and its bundles verify.
- `../../etc/passwd`, an absolute path outside the root, a symlink pointing
  out of the root, a prefix sibling (`<root>-evil`) and a config whose
  `base:` escapes the root are all rejected.
- Errors come back with the CLI's error codes.

## Out of scope

No caching, sessions, background jobs, authentication beyond what the MCP
transport provides, or tools that do not map to a CLI command.
