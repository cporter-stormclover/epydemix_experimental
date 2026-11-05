"""Run every worked example of the agent contract (AGENT.md) as written.

Each fenced block tagged ```bash example:NAME``` runs in a fresh directory.
It must succeed, and the JSON its last command prints must have the shape
recorded in ``tests/fixtures/contract_examples/NAME.json`` (keys and value
types, not values: runs are stochastic). Set EPX_UPDATE_CONTRACT_EXAMPLES=1
to record the shapes again after changing an example.

The examples run offline: a ``sitecustomize`` module put on the subprocesses'
path serves the Italy population from the test snapshot instead of
downloading it, and stands in for the population data files that provenance
hashes.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AGENT_MD = ROOT / "AGENT.md"
EXPECTED = Path(__file__).parent / "fixtures" / "contract_examples"
ITALY = Path(__file__).parent / "fixtures" / "italy_v1.2.0.json"
FENCE = re.compile(r"```bash example:([A-Za-z0-9_]+)\n(.*?)```", re.S)

SITECUSTOMIZE = """
import json, os
import numpy as np

def _offline():
    import epydemix.io.provenance as provenance
    import epydemix.model.epimodel as epimodel
    from epydemix.population import Population

    with open(os.environ["EPX_TEST_ITALY"]) as f:
        data = json.load(f)

    def load(name, *args, layers=None, **kwargs):
        if name != "Italy":
            raise ValueError("offline examples only have Italy")
        pop = Population(name="Italy")
        pop.add_population(np.array(data["Nk"]), Nk_names=data["Nk_names"])
        for layer in layers or ["school", "work", "home", "community"]:
            pop.add_contact_matrix(np.array(data["contact_matrices"][layer]), layer)
        return pop

    def fetch(url):
        if url.endswith("/locations.csv"):
            return b"location,primary_contact_source\\nItaly,mistry_2021\\n"
        return ("offline copy of " + url).encode()

    epimodel.load_epydemix_population = load
    provenance._fetch = fetch

_offline()
"""


def extract_examples(text):
    """``{name: script}`` for every tagged example, in order."""
    examples = {}
    for name, body in FENCE.findall(text):
        if name in examples:
            raise ValueError(f"two examples are named {name}")
        examples[name] = body
    return examples


def json_shape(value):
    """Keys and value types of a JSON document (a list by its first item)."""
    if isinstance(value, dict):
        return {k: json_shape(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_shape(value[0])] if value else []
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if value is None:
        return "null"
    return "str"


EXAMPLES = extract_examples(AGENT_MD.read_text())

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("epydemix") is None or os.name == "nt",
    reason="the contract examples are bash scripts calling the epydemix CLI",
)


def test_the_contract_has_its_worked_examples():
    assert {"run_seir", "case_study", "school_closure"} <= set(EXAMPLES)


@pytest.mark.parametrize("name", list(EXAMPLES))
def test_example(name, tmp_path):
    site = tmp_path / "_site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(SITECUSTOMIZE)
    work = tmp_path / "work"
    work.mkdir()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(site)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env["EPX_TEST_ITALY"] = str(ITALY)
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")

    lines = EXAMPLES[name].rstrip("\n").split("\n")
    setup, last = "\n".join(lines[:-1]), lines[-1]
    for script in (setup, last):
        result = subprocess.run(
            ["bash", "-e", "-c", script],
            cwd=work,
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        assert result.returncode == 0, (
            f"{name}: {script[:200]}\n{result.stderr[-3000:]}"
        )

    shape = json_shape(json.loads(result.stdout))
    expected_path = EXPECTED / f"{name}.json"
    if os.environ.get("EPX_UPDATE_CONTRACT_EXAMPLES"):
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        expected_path.write_text(json.dumps(shape, indent=1) + "\n")
    assert expected_path.exists(), (
        f"no recorded shape for {name}; run with EPX_UPDATE_CONTRACT_EXAMPLES=1"
    )
    assert shape == json.loads(expected_path.read_text()), name
