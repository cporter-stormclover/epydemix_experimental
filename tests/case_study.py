"""The vaccination case study of AGENT.md, as files.

An SEIR epidemic with R0 = 2.5 in Italy (all four contact layers), with a
vaccination campaign at three qualitative rates and without one: four
scenarios, four bundles. Kept in step with the case study in AGENT.md.
"""

import yaml

BASE = {
    "model": {"type": "SEIR", "vaccination": True},
    "population": {
        "name": "Italy",
        "contact_layers": ["home", "work", "school", "community"],
    },
    # transmission_rate = R0 * recovery_rate / rho(C) = 2.5 * 0.2 / 13.9 = 0.036
    "derived": {"transmission_rate": {"from": "R0", "value": 2.5}},
    "parameters": {
        "incubation_rate": 0.25,
        "recovery_rate": 0.2,
        "vaccine_efficacy": 0.8,
        "vaccination_rate": 0.0,
    },
    "simulation": {
        "start_date": "2026-10-01",
        "end_date": "2027-03-31",
        "n_simulations": 20,
        "seed": 2026,
    },
    "initial_conditions": {
        "Susceptible": 0.9999,
        "Exposed": 0.00005,
        "Infected": 0.00005,
        "Recovered": 0.0,
        "Vaccinated": 0.0,
    },
}

# Scenario name -> vaccination_rate (per day): none, and the midpoint of each
# qualitative band of the registry.
SCENARIOS = {
    "no_vaccination": 0.0,
    "low": 0.0015,
    "moderate": 0.0045,
    "high": 0.0115,
}

# The basis of every value the base config sets (the prompt gives R0 = 2.5, a
# 4-day latent period, a 5-day infectious period and an 80% efficacy).
BASE_ASSUMPTIONS = [
    {
        "parameter": "incubation_rate",
        "value": 0.25,
        "basis": "derived",
        "rationale": "4-day latent period",
    },
    {
        "parameter": "recovery_rate",
        "value": 0.2,
        "basis": "derived",
        "rationale": "5-day infectious period",
    },
    {
        "parameter": "vaccine_efficacy",
        "value": 0.8,
        "basis": "prompt_literal",
        "rationale": "80% efficacy",
    },
]


def _scenario_assumption(name, rate):
    if name == "no_vaccination":
        return {
            "parameter": "vaccination_rate",
            "value": rate,
            "basis": "prompt_literal",
            "rationale": "no campaign",
        }
    return {
        "parameter": "vaccination_rate",
        "value": rate,
        "basis": "agent_choice",
        "rationale": f"midpoint of the '{name}' band",
    }


def write_case_study(directory, n_simulations=None, declare=False):
    """Write base.yaml and one overlay per scenario; return {scenario: path}.

    With ``declare``, the configs carry an ``assumptions`` block for every
    value they set.
    """
    base = yaml.safe_load(yaml.safe_dump(BASE))
    if n_simulations is not None:
        base["simulation"]["n_simulations"] = n_simulations
    if declare:
        base["assumptions"] = BASE_ASSUMPTIONS
    with open(directory / "base.yaml", "w") as f:
        yaml.safe_dump(base, f, sort_keys=False)
    paths = {}
    for name, rate in SCENARIOS.items():
        overlay = {"base": "base.yaml", "parameters": {"vaccination_rate": rate}}
        if declare:
            overlay["assumptions"] = [_scenario_assumption(name, rate)]
        path = directory / f"{name}.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(overlay, f)
        paths[name] = path
    return paths
