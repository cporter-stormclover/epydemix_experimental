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
    "parameters": {
        # R0 = 2.5: transmission_rate = R0 * recovery_rate / rho(C), rho = 13.9
        "transmission_rate": 0.036,
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

# Scenario name -> vaccination_rate (per day): none, and one value inside each
# qualitative band of the registry.
SCENARIOS = {
    "no_vaccination": 0.0,
    "low": 0.0015,
    "moderate": 0.0045,
    "high": 0.0115,
}


def write_case_study(directory, n_simulations=None):
    """Write base.yaml and one overlay per scenario; return {scenario: path}."""
    base = yaml.safe_load(yaml.safe_dump(BASE))
    if n_simulations is not None:
        base["simulation"]["n_simulations"] = n_simulations
    with open(directory / "base.yaml", "w") as f:
        yaml.safe_dump(base, f, sort_keys=False)
    paths = {}
    for name, rate in SCENARIOS.items():
        path = directory / f"{name}.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(
                {"base": "base.yaml", "parameters": {"vaccination_rate": rate}}, f
            )
        paths[name] = path
    return paths
