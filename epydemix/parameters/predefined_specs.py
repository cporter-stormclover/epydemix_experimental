"""Canonical parameter specifications for predefined epidemic models.

Each function returns a list of :class:`ParameterSpec` objects describing
the parameters of a predefined model (SIR, SEIR, SIS, SEIAR) or a modular
extension (waning immunity, vaccination, outcome). These are registered
automatically when a predefined model is created.
"""

from .semantic import SemanticCheck
from .spec import ParameterSpec


def sir_specs(
    transmission_rate: float = 0.3,
    recovery_rate: float = 0.1,
) -> list:
    """Parameter specs for the SIR model."""
    return [
        ParameterSpec(
            name="transmission_rate",
            description=(
                "Rate at which susceptible individuals become infected through "
                "contact with infected individuals. Controls the speed of epidemic "
                "growth. Higher values produce faster, larger outbreaks."
            ),
            kind="rate",
            default=transmission_rate,
            min=0,
            max=10,
            units="1/days",
            precision=2,
            tags=["transmission", "SIR"],
        ),
        ParameterSpec(
            name="recovery_rate",
            description=(
                "Rate at which infected individuals recover and become immune. "
                "The inverse (1/recovery_rate) gives the mean infectious period "
                "in days."
            ),
            kind="rate",
            default=recovery_rate,
            min=0,
            max=10,
            units="1/days",
            precision=2,
            tags=["recovery", "SIR"],
        ),
    ]


def seir_specs(
    transmission_rate: float = 0.3,
    incubation_rate: float = 0.2,
    recovery_rate: float = 0.1,
) -> list:
    """Parameter specs for the SEIR model."""
    return [
        ParameterSpec(
            name="transmission_rate",
            description=(
                "Rate at which susceptible individuals become exposed through "
                "contact with infected individuals. Controls the force of "
                "infection. Higher values produce faster epidemic growth."
            ),
            kind="rate",
            default=transmission_rate,
            min=0,
            max=10,
            units="1/days",
            precision=2,
            tags=["transmission", "SEIR"],
        ),
        ParameterSpec(
            name="incubation_rate",
            description=(
                "Rate at which exposed individuals become infectious. "
                "The inverse (1/incubation_rate) gives the mean latent "
                "(incubation) period in days."
            ),
            kind="rate",
            default=incubation_rate,
            min=0,
            max=10,
            units="1/days",
            precision=2,
            tags=["incubation", "SEIR"],
        ),
        ParameterSpec(
            name="recovery_rate",
            description=(
                "Rate at which infected individuals recover and become immune. "
                "The inverse (1/recovery_rate) gives the mean infectious period "
                "in days."
            ),
            kind="rate",
            default=recovery_rate,
            min=0,
            max=10,
            units="1/days",
            precision=2,
            tags=["recovery", "SEIR"],
        ),
    ]


def sis_specs(
    transmission_rate: float = 0.3,
    recovery_rate: float = 0.1,
) -> list:
    """Parameter specs for the SIS model."""
    return [
        ParameterSpec(
            name="transmission_rate",
            description=(
                "Rate at which susceptible individuals become infected through "
                "contact with infected individuals. In the SIS model, recovered "
                "individuals return to susceptible, so this rate also governs "
                "reinfection dynamics."
            ),
            kind="rate",
            default=transmission_rate,
            min=0,
            max=10,
            units="1/days",
            precision=2,
            tags=["transmission", "SIS"],
        ),
        ParameterSpec(
            name="recovery_rate",
            description=(
                "Rate at which infected individuals recover and return to the "
                "susceptible state (no lasting immunity). The inverse "
                "(1/recovery_rate) gives the mean infectious period in days."
            ),
            kind="rate",
            default=recovery_rate,
            min=0,
            max=10,
            units="1/days",
            precision=2,
            tags=["recovery", "SIS"],
        ),
    ]


def seiar_specs(
    transmission_rate: float = 0.3,
    incubation_rate: float = 0.2,
    recovery_rate: float = 0.1,
    asymptomatic_fraction: float = 0.4,
    asymptomatic_recovery_rate: float = 0.14,
    asymptomatic_relative_infectivity: float = 0.5,
) -> list:
    """Parameter specs for the SEIAR model (SEIR with an asymptomatic branch)."""
    return [
        ParameterSpec(
            name="transmission_rate",
            description=(
                "Rate at which susceptible individuals become exposed through "
                "contact with symptomatic infected individuals. Asymptomatic "
                "contacts contribute at a reduced rate scaled by "
                "asymptomatic_relative_infectivity."
            ),
            kind="rate",
            default=transmission_rate,
            min=0,
            max=10,
            units="1/days",
            precision=2,
            tags=["transmission", "SEIAR"],
        ),
        ParameterSpec(
            name="incubation_rate",
            description=(
                "Rate at which exposed individuals leave the Exposed compartment. "
                "A fraction asymptomatic_fraction progresses to Asymptomatic; the "
                "rest progresses to Infected. The inverse (1/incubation_rate) is "
                "the mean latent period in days."
            ),
            kind="rate",
            default=incubation_rate,
            min=0,
            max=10,
            units="1/days",
            precision=2,
            tags=["incubation", "SEIAR"],
        ),
        ParameterSpec(
            name="recovery_rate",
            description=(
                "Recovery rate for symptomatic Infected individuals. The inverse "
                "(1/recovery_rate) is the mean symptomatic infectious period."
            ),
            kind="rate",
            default=recovery_rate,
            min=0,
            max=10,
            units="1/days",
            precision=2,
            tags=["recovery", "SEIAR"],
        ),
        ParameterSpec(
            name="asymptomatic_fraction",
            description=(
                "Fraction of Exposed individuals who progress to the "
                "Asymptomatic compartment rather than Infected. The remaining "
                "(1 - asymptomatic_fraction) develops symptoms."
            ),
            kind="proportion",
            default=asymptomatic_fraction,
            min=0,
            max=1,
            units="dimensionless",
            precision=2,
            tags=["SEIAR"],
        ),
        ParameterSpec(
            name="asymptomatic_recovery_rate",
            description=(
                "Recovery rate for Asymptomatic individuals. The inverse "
                "(1/asymptomatic_recovery_rate) is the mean asymptomatic "
                "infectious period in days."
            ),
            kind="rate",
            default=asymptomatic_recovery_rate,
            min=0,
            max=10,
            units="1/days",
            precision=2,
            tags=["recovery", "SEIAR"],
        ),
        ParameterSpec(
            name="asymptomatic_relative_infectivity",
            description=(
                "Infectivity of Asymptomatic individuals relative to symptomatic "
                "Infected. Multiplies transmission_rate on the S → E flow driven "
                "by Asymptomatic. 0 means asymptomatics do not transmit; 1 means "
                "they transmit as much as symptomatic cases."
            ),
            kind="proportion",
            default=asymptomatic_relative_infectivity,
            min=0,
            max=1,
            units="dimensionless",
            precision=2,
            tags=["transmission", "SEIAR"],
        ),
    ]


def waning_immunity_specs(waning_rate: float = 1.0 / 365) -> list:
    """Parameter specs for the waning-immunity module (R → S)."""
    return [
        ParameterSpec(
            name="waning_rate",
            description=(
                "Rate at which recovered individuals lose immunity and return to "
                "Susceptible. The inverse (1/waning_rate) is the mean immune "
                "duration in days."
            ),
            kind="rate",
            default=waning_rate,
            min=0,
            max=1,
            units="1/days",
            precision=2,
            tags=["waning", "module"],
        ),
    ]


# Qualitative vaccination paces. Paltiel et al. (2021) model the pace of a
# COVID-19 vaccination campaign as the share of the population vaccinated per
# day: base case 0.5%/day (the daily share reached in US influenza vaccination
# campaigns), range 0.1-2.0%/day, with ~0.1%/day the pace of the early US
# rollout and ~1%/day the pace hoped for. Their pace counts doses against the
# whole population; vaccination_rate applies to the susceptible compartment,
# so the two coincide only while nearly everyone is susceptible.
_PALTIEL_2021 = (
    "Paltiel AD, Schwartz JL, Zheng A, Walensky RP. Clinical outcomes of a "
    "COVID-19 vaccine: implementation over efficacy. Health Affairs "
    "2021;40(1):42-52. doi:10.1377/hlthaff.2020.02054"
)
VACCINATION_RATE_BANDS = {
    "low": {
        "min": 0.001,
        "max": 0.002,
        "source": _PALTIEL_2021,
        "note": "~0.1%/day: the early US COVID-19 rollout, the low end of the "
        "range modelled. Band edges are this package's reading of the source.",
    },
    "moderate": {
        "min": 0.003,
        "max": 0.006,
        "source": _PALTIEL_2021,
        "note": "0.5%/day base case, the daily share reached in US influenza "
        "vaccination. Band edges are this package's reading of the source.",
    },
    "high": {
        "min": 0.008,
        "max": 0.015,
        "source": _PALTIEL_2021,
        "note": "~1%/day, the pace hoped for in the US COVID-19 rollout, within "
        "the 2%/day upper end of the range modelled. Band edges are this "
        "package's reading of the source.",
    },
}


def vaccination_specs(
    vaccination_rate: float = 0.01,
    vaccine_efficacy: float = 0.9,
) -> list:
    """Parameter specs for the vaccination module (S → V, leaky V → I breakthrough)."""
    return [
        ParameterSpec(
            name="vaccination_rate",
            description=(
                "Per-capita rate at which Susceptible individuals are vaccinated "
                "and move to the Vaccinated compartment."
            ),
            kind="rate",
            default=vaccination_rate,
            min=0,
            max=1,
            units="1/days",
            precision=2,
            tags=["vaccination", "module"],
            qualitative_bands=VACCINATION_RATE_BANDS,
        ),
        ParameterSpec(
            name="vaccine_efficacy",
            description=(
                "Fractional reduction in transmission for Vaccinated individuals. "
                "Breakthrough transmission (V → Exposed if the backbone has an "
                "Exposed compartment, otherwise V → Infected) uses "
                "transmission_rate * (1 - vaccine_efficacy). 0 means no protection; "
                "1 means perfect protection."
            ),
            kind="proportion",
            default=vaccine_efficacy,
            min=0,
            max=1,
            units="dimensionless",
            precision=2,
            tags=["vaccination", "module"],
        ),
    ]


def outcome_specs(
    outcome: str,
    mortality_rate: float = 0.01,
    hospitalization_rate: float = 0.01,
    hospitalization_recovery_rate: float = 0.1,
) -> list:
    """Parameter specs for the outcome module ('deaths' or 'hospitalization')."""
    if outcome == "deaths":
        return [
            ParameterSpec(
                name="mortality_rate",
                description=(
                    "Fraction of the Infected outflow that results in death rather "
                    "than recovery. Adds an Infected → Dead spontaneous transition "
                    "and rescales the existing Infected → Recovered transition by "
                    "(1 - mortality_rate)."
                ),
                kind="proportion",
                default=mortality_rate,
                min=0,
                max=1,
                units="dimensionless",
                precision=2,
                tags=["outcome", "deaths", "module"],
            ),
        ]
    if outcome == "hospitalization":
        return [
            ParameterSpec(
                name="hospitalization_rate",
                description=(
                    "Fraction of the Infected outflow that results in hospitalization "
                    "rather than direct recovery. Adds an Infected → Hospitalized "
                    "spontaneous transition and rescales the existing Infected → "
                    "Recovered transition by (1 - hospitalization_rate)."
                ),
                kind="proportion",
                default=hospitalization_rate,
                min=0,
                max=1,
                units="dimensionless",
                precision=2,
                tags=["outcome", "hospitalization", "module"],
            ),
            ParameterSpec(
                name="hospitalization_recovery_rate",
                description=(
                    "Rate at which Hospitalized individuals recover. The inverse "
                    "(1/hospitalization_recovery_rate) is the mean length of stay."
                ),
                kind="rate",
                default=hospitalization_recovery_rate,
                min=0,
                max=10,
                units="1/days",
                precision=2,
                tags=["outcome", "hospitalization", "module"],
            ),
        ]
    raise ValueError(
        f"Unknown outcome '{outcome}'. Supported values: 'deaths', 'hospitalization'."
    )


def default_semantic_checks() -> list:
    """Cross-parameter rules for configs of the predefined models."""
    module_rules = [
        ("vaccination_rate", "model.vaccination", True, "vaccination"),
        ("waning_rate", "model.waning_immunity", True, "waning immunity"),
        ("mortality_rate", "model.outcome", "deaths", "the deaths outcome"),
        (
            "hospitalization_rate",
            "model.outcome",
            "hospitalization",
            "the hospitalization outcome",
        ),
    ]
    checks = [
        SemanticCheck(
            name=f"{parameter}_requires_module",
            rule="requires",
            parameter=parameter,
            greater_than=0.0,
            requires={path: value},
            message=f"a positive {parameter} has no effect without {label}",
        )
        for parameter, path, value, label in module_rules
    ]
    checks += [
        SemanticCheck(
            name=f"{section}_dates_ordered",
            rule="ordered_dates",
            section=section,
            message="A window that ends before it starts has no effect.",
        )
        for section in ("interventions", "overrides")
    ]
    checks.append(
        SemanticCheck(
            name="derived_r0_plausible",
            rule="range",
            severity="warning",
            quantity="derived.*.value",
            min=0.5,
            max=30.0,
            message="an R0 outside [0.5, 30] is implausible for a human "
            "infectious disease; check the value",
        )
    )
    return checks


# Map of model name → spec factory function
MODEL_SPEC_FACTORIES = {
    "SIR": sir_specs,
    "SEIR": seir_specs,
    "SIS": sis_specs,
    "SEIAR": seiar_specs,
}
