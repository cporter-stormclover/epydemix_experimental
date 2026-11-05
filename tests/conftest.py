"""Shared fixtures for the agent-framework tests.

``offline_italy`` replaces the download of named populations with a snapshot
of the Italy data (epydemix-data v1.2.0, default age groups), so tests that use
``population: {name: Italy}`` run offline and see the real contact matrices.
"""

import json
from pathlib import Path

import numpy as np
import pytest

_ITALY = Path(__file__).parent / "fixtures" / "italy_v1.2.0.json"


def italy_population(layers=None):
    from epydemix.population import Population

    with open(_ITALY) as f:
        data = json.load(f)
    pop = Population(name="Italy")
    pop.add_population(np.array(data["Nk"]), Nk_names=data["Nk_names"])
    for layer in layers or ["school", "work", "home", "community"]:
        pop.add_contact_matrix(np.array(data["contact_matrices"][layer]), layer)
    return pop


@pytest.fixture
def offline_italy(monkeypatch):
    """Serve every named population from the Italy snapshot."""
    import epydemix.model.epimodel as epimodel

    def fake_load(population_name, *args, layers=None, **kwargs):
        if population_name != "Italy":
            raise ValueError(
                f"offline test fixture only has Italy, not {population_name}"
            )
        return italy_population(layers)

    monkeypatch.setattr(epimodel, "load_epydemix_population", fake_load)
    return fake_load
