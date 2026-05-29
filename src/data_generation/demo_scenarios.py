from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data_generation.generate_pd_observations import generate_pd_observations


REQUIRED_COLUMNS = [
    "observation_id",
    "obligor_id",
    "facility_id",
    "portfolio",
    "segment",
    "product_type",
    "model_id",
    "model_version",
    "observation_date",
    "origination_date",
    "performance_window_months",
    "rating_grade",
    "score",
    "pd_estimate",
    "ead_at_observation",
    "default_flag_12m",
    "default_date",
    "exclusion_flag",
    "exclusion_reason",
    "snapshot_id",
    "as_of_date",
    "observation_year",
    "previous_observation_date",
    "regulatory_exposure_class",
    "irb_approach",
    "application_scope_flag",
    "model_applicability_flag",
    "exposure_status_at_observation",
    "performing_status_at_observation",
    "default_flag_at_observation",
    "data_complete_12m_flag",
    "maturity_before_12m_flag",
    "closure_date",
    "rating_system_id",
    "rating_method",
    "model_use_type",
    "model_philosophy",
    "pd_type",
    "pd_horizon_months",
    "calibration_pool_id",
    "rating_assignment_date",
    "last_review_date",
    "rating_age_months",
    "stale_rating_flag",
    "grade_order",
    "master_scale_grade",
    "score_bucket",
    "pd_bucket",
    "pd_raw",
    "pd_calibrated",
    "pd_regulatory",
    "pd_before_floor",
    "pd_after_floor",
    "pd_floor_applied_flag",
    "pd_floor_value",
    "margin_of_conservatism",
    "margin_of_conservatism_type",
    "previous_rating_grade",
    "previous_grade_order",
    "previous_score",
    "previous_pd_estimate",
    "previous_pd_bucket",
    "rating_migration",
    "notch_change",
    "rating_change_direction",
    "new_customer_flag",
    "exited_customer_flag",
    "country_code",
    "country_of_risk",
    "sector_code",
    "borrower_size_class",
    "secured_flag",
    "collateral_type",
    "ltv_bucket",
    "origination_channel",
    "watchlist_flag",
    "forbearance_flag_at_observation",
    "npe_flag_at_observation",
    "days_past_due_at_observation",
    "default_reason",
    "past_due_90_flag_12m",
    "utp_flag_12m",
    "max_days_past_due_12m",
    "months_to_default",
    "default_exit_date",
    "cure_flag",
    "exclusion_category",
    "exclusion_materiality",
    "exclusion_applied_by_rule",
    "exclusion_rule_id",
]


@dataclass(frozen=True)
class DemoScenario:
    key: str
    label: str
    description: str
    expected_observation: str


SCENARIOS = {
    "retail_well_calibrated": DemoScenario(
        "retail_well_calibrated",
        "Retail bien calibre",
        "Portefeuille Retail proche de la calibration attendue, avec volumes suffisants.",
        "Les alertes de calibration doivent majoritairement rester vertes.",
    ),
    "retail_underestimation": DemoScenario(
        "retail_underestimation",
        "Retail sous-estimation du risque",
        "Sous-estimation volontaire du risque sur les grades E, F et G du portefeuille Retail.",
        "L'ODR devrait depasser la PD moyenne sur les grades risqués et generer des alertes orange ou rouges.",
    ),
    "corporate_low_default": DemoScenario(
        "corporate_low_default",
        "Corporate low-default portfolio",
        "Portefeuille Corporate avec tres faibles volumes de defauts.",
        "Plusieurs tests Corporate doivent etre gris ou fragiles du fait des faibles volumes de defauts.",
    ),
    "corporate_degraded_discrimination": DemoScenario(
        "corporate_degraded_discrimination",
        "Corporate discrimination degradee",
        "Les scores Corporate sont volontairement brouilles pour degrader le pouvoir discriminant.",
        "L'AUC, le Gini ou le KS Corporate doivent se deteriorer.",
    ),
    "population_shift": DemoScenario(
        "population_shift",
        "Population shift PSI eleve",
        "La population des derniers millesimes migre vers des grades plus risques.",
        "Le PSI entre periode de reference et periode courante doit augmenter.",
    ),
    "rds_stable_population": DemoScenario(
        "rds_stable_population",
        "RDS population stable",
        "Population multi-periodes avec distributions de grades, scores et buckets de PD relativement stables.",
        "Le PSI doit rester faible et les changements de mix doivent etre limites.",
    ),
    "rds_population_shift": DemoScenario(
        "rds_population_shift",
        "RDS population shift",
        "La population courante migre vers des grades, secteurs et buckets de PD plus risques.",
        "Le PSI et les contributeurs de shift doivent ressortir en orange ou rouge.",
    ),
    "rating_migration_deterioration": DemoScenario(
        "rating_migration_deterioration",
        "Migration rating degradee",
        "Les migrations de rating sont volontairement degradees sur la derniere periode.",
        "La matrice de migration doit afficher davantage de downgrades et de changements de crans.",
    ),
    "pit_vs_ttc_comparison": DemoScenario(
        "pit_vs_ttc_comparison",
        "Comparaison PIT / TTC / Hybrid",
        "La base contient des philosophies de modeles contrastees pour illustrer leur lecture validation.",
        "Les commentaires PIT/TTC doivent differencier variation attendue et stabilite relative.",
    ),
    "pit_ttc_coherent": DemoScenario(
        "pit_ttc_coherent",
        "PIT / TTC coherent",
        "Les PD PIT varient davantage avec la periode courante, les PD TTC restent plus stables et les modeles Hybrid ont un comportement intermediaire.",
        "La volatilite PD doit etre plus elevee en PIT qu'en TTC, sans conclure qu'une philosophie est meilleure.",
    ),
    "pit_ttc_incoherent": DemoScenario(
        "pit_ttc_incoherent",
        "PIT / TTC incoherence apparente",
        "Des modeles declares TTC presentent une forte variation temporelle ou une derive de distribution.",
        "Le commentaire doit recommander une investigation prudente sur la coherence entre philosophie declaree et comportement observe.",
    ),
    "pd_floor_and_moc": DemoScenario(
        "pd_floor_and_moc",
        "PD floors et MoC",
        "Les PD reglementaires integrent davantage de floors et de marges de conservatisme.",
        "Les impacts floor/MoC doivent etre visibles par segment et portfolio.",
    ),
    "regulatory_eligibility_issues": DemoScenario(
        "regulatory_eligibility_issues",
        "Eligibilite reglementaire fragile",
        "Le perimetre contient davantage d'expositions hors scope ou sans horizon complet.",
        "Le waterfall population doit faire apparaitre des exclusions significatives.",
    ),
    "exclusion_analysis": DemoScenario(
        "exclusion_analysis",
        "Analyse fine des exclusions",
        "Les exclusions sont variees par categorie, materialite et regle appliquee.",
        "La page Data Quality doit prioriser les categories d'exclusion et leur impact EAD.",
    ),
    "corporate_low_default_portfolio": DemoScenario(
        "corporate_low_default_portfolio",
        "Corporate low-default portfolio avance",
        "Portefeuille Corporate avec tres peu de defauts et historique fragile.",
        "Les analyses LDP doivent recommander prudence, regroupement et analyse pluriannuelle.",
    ),
    "data_quality_issues": DemoScenario(
        "data_quality_issues",
        "Dataset avec anomalies data quality",
        "Le jeu de donnees contient davantage d'anomalies de qualite.",
        "Les controles data quality doivent remonter plusieurs anomalies visibles.",
    ),
}


def scenario_catalog() -> dict[str, DemoScenario]:
    """Return available demo scenarios."""
    return SCENARIOS


def _set_defaults(frame: pd.DataFrame, mask: pd.Series, multiplier: float, rng: np.random.Generator) -> None:
    valid_mask = mask & frame["pd_estimate"].notna() & frame["observation_date"].notna()
    probs = np.clip(frame.loc[valid_mask, "pd_estimate"].astype(float) * multiplier, 0.0001, 0.95)
    frame.loc[valid_mask, "default_flag_12m"] = rng.binomial(1, probs).astype(float)
    offsets = rng.integers(1, 366, size=int(valid_mask.sum())).astype("timedelta64[D]")
    default_dates = pd.to_datetime(frame.loc[valid_mask, "observation_date"].to_numpy() + offsets)
    default_index = frame.loc[valid_mask].index
    frame.loc[default_index, "default_date"] = pd.NaT
    default_index = frame.loc[valid_mask & (frame["default_flag_12m"] == 1)].index
    frame.loc[default_index, "default_date"] = default_dates[
        frame.loc[valid_mask, "default_flag_12m"].to_numpy() == 1
    ]


def _add_extra_anomalies(frame: pd.DataFrame, rng: np.random.Generator, level: str) -> pd.DataFrame:
    counts = {"none": 0, "low": 30, "medium": 120, "high": 350}
    count = counts.get(level, counts["medium"])
    if count == 0:
        return frame
    output = frame.copy()
    indexes = rng.choice(output.index, size=min(count, len(output)), replace=False)
    chunks = np.array_split(indexes, 7)
    output.loc[chunks[0], "pd_estimate"] = np.nan
    output.loc[chunks[1], "pd_estimate"] = -0.02
    output.loc[chunks[2], "default_flag_12m"] = np.nan
    output.loc[chunks[3], "rating_grade"] = np.nan
    output.loc[chunks[4], "portfolio"] = np.nan
    output.loc[chunks[5], "performance_window_months"] = 6
    output.loc[chunks[6], "observation_date"] = pd.NaT
    return output


def generate_demo_scenario(
    scenario: str,
    retail_observations: int = 30000,
    corporate_observations: int = 5000,
    start_year: int = 2019,
    years: int = 5,
    data_quality_level: str = "low",
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate a documented, reproducible demo scenario."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")

    rng = np.random.default_rng(random_seed)
    include_base_anomalies = scenario == "data_quality_issues" or data_quality_level != "none"
    frame = generate_pd_observations(
        retail_observations=retail_observations,
        corporate_observations=corporate_observations,
        seed=random_seed,
        include_anomalies=include_base_anomalies,
        observation_start_year=start_year,
        observation_years=years,
    )

    if scenario == "retail_well_calibrated":
        pass
    elif scenario == "retail_underestimation":
        mask = (frame["portfolio"] == "Retail") & frame["rating_grade"].isin(["E", "F", "G"])
        _set_defaults(frame, mask, 2.8, rng)
        frame.loc[mask, "pd_estimate"] = np.clip(frame.loc[mask, "pd_estimate"] * 0.65, 0.0001, 1.0)
    elif scenario == "corporate_low_default":
        mask = frame["portfolio"] == "Corporate"
        _set_defaults(frame, mask, 0.25, rng)
    elif scenario == "corporate_degraded_discrimination":
        mask = frame["portfolio"] == "Corporate"
        shuffled_pd = frame.loc[mask, "pd_estimate"].sample(frac=1, random_state=random_seed).to_numpy()
        frame.loc[mask, "pd_estimate"] = shuffled_pd
        frame.loc[mask, "score"] = rng.normal(650, 55, size=int(mask.sum())).round(0)
    elif scenario in {"population_shift", "rds_population_shift"}:
        last_year = start_year + years - 1
        mask = frame["observation_date"].dt.year.eq(last_year)
        retail_shift = mask & frame["portfolio"].eq("Retail")
        corporate_shift = mask & frame["portfolio"].eq("Corporate")
        frame.loc[retail_shift, "rating_grade"] = rng.choice(["D", "E", "F", "G"], size=int(retail_shift.sum()), p=[0.20, 0.35, 0.30, 0.15])
        frame.loc[corporate_shift, "rating_grade"] = rng.choice(["C", "D", "E", "F"], size=int(corporate_shift.sum()), p=[0.20, 0.35, 0.30, 0.15])
        frame.loc[mask, "pd_estimate"] = np.clip(frame.loc[mask, "pd_estimate"] * 1.8, 0.0001, 1.0)
        frame.loc[mask, "pd_bucket"] = pd.cut(frame.loc[mask, "pd_estimate"], [0, 0.005, 0.015, 0.04, 0.10, 1.0], labels=["PD1", "PD2", "PD3", "PD4", "PD5"], include_lowest=True).astype(object)
        frame.loc[mask, "segment"] = rng.choice(["Consumer Finance", "Credit Card", "SME", "Specialized Lending"], size=int(mask.sum()))
        frame.loc[mask, "country_code"] = rng.choice(["IT", "ES", "BE"], size=int(mask.sum()), p=[0.45, 0.35, 0.20])
        frame.loc[mask, "country_of_risk"] = frame.loc[mask, "country_code"]
    elif scenario == "data_quality_issues":
        frame = _add_extra_anomalies(frame, rng, "high")
    elif scenario == "rds_stable_population":
        pass
    elif scenario == "rating_migration_deterioration":
        mask = frame["observation_date"].dt.year.eq(start_year + years - 1)
        frame.loc[mask, "previous_grade_order"] = np.clip(frame.loc[mask, "grade_order"].astype(float) - rng.choice([1, 2, 3], size=int(mask.sum()), p=[0.50, 0.35, 0.15]), 1, 7)
        frame.loc[mask, "notch_change"] = frame.loc[mask, "grade_order"].astype(float) - frame.loc[mask, "previous_grade_order"].astype(float)
        frame.loc[mask, "rating_migration"] = "downgrade"
        frame.loc[mask, "rating_change_direction"] = "worse"
    elif scenario in {"pit_vs_ttc_comparison", "pit_ttc_coherent"}:
        frame.loc[frame["portfolio"].eq("Retail"), "model_philosophy"] = "PIT"
        frame.loc[frame["portfolio"].eq("Corporate"), "model_philosophy"] = "TTC"
        hybrid_mask = frame["segment"].astype(str).isin(["Consumer Finance", "SME"])
        frame.loc[hybrid_mask, "model_philosophy"] = "Hybrid"
        last_year = frame["observation_year"].max()
        pit_mask = frame["model_philosophy"].eq("PIT") & frame["observation_year"].eq(last_year)
        frame.loc[pit_mask, "pd_estimate"] = np.clip(frame.loc[pit_mask, "pd_estimate"] * 1.45, 0.0001, 1.0)
        hybrid_mask = frame["model_philosophy"].eq("Hybrid") & frame["observation_year"].eq(last_year)
        frame.loc[hybrid_mask, "pd_estimate"] = np.clip(frame.loc[hybrid_mask, "pd_estimate"] * 1.18, 0.0001, 1.0)
    elif scenario == "pit_ttc_incoherent":
        frame["model_philosophy"] = "TTC"
        last_year = frame["observation_year"].max()
        stressed = frame["observation_year"].eq(last_year)
        frame.loc[stressed, "pd_estimate"] = np.clip(frame.loc[stressed, "pd_estimate"] * 1.90, 0.0001, 1.0)
        frame.loc[stressed, "rating_grade"] = rng.choice(["D", "E", "F", "G"], size=int(stressed.sum()), p=[0.25, 0.35, 0.25, 0.15])
    elif scenario == "pd_floor_and_moc":
        impacted = rng.choice(frame.index, size=max(1, int(len(frame) * 0.18)), replace=False)
        frame.loc[impacted, "pd_floor_value"] = np.maximum(frame.loc[impacted, "pd_floor_value"], frame.loc[impacted, "pd_before_floor"] * 1.25)
        frame.loc[impacted, "pd_after_floor"] = np.maximum(frame.loc[impacted, "pd_before_floor"], frame.loc[impacted, "pd_floor_value"])
        frame.loc[impacted, "pd_regulatory"] = frame.loc[impacted, "pd_after_floor"]
        frame.loc[impacted, "pd_floor_applied_flag"] = 1
        frame.loc[impacted, "margin_of_conservatism"] = frame.loc[impacted, "margin_of_conservatism"] + rng.uniform(0.003, 0.012, size=len(impacted))
        frame.loc[impacted, "margin_of_conservatism_type"] = "conservatism_overlay"
    elif scenario in {"regulatory_eligibility_issues", "exclusion_analysis"}:
        impacted = rng.choice(frame.index, size=max(1, int(len(frame) * 0.14)), replace=False)
        categories = ["out_of_scope", "defaulted_at_observation", "incomplete_performance_window", "closed_before_horizon", "model_not_applicable", "data_quality"]
        frame.loc[impacted, "exclusion_flag"] = 1
        frame.loc[impacted, "exclusion_category"] = rng.choice(categories, size=len(impacted))
        frame.loc[impacted, "exclusion_reason"] = frame.loc[impacted, "exclusion_category"]
        frame.loc[impacted, "exclusion_materiality"] = rng.choice(["medium", "high"], size=len(impacted), p=[0.55, 0.45])
        frame.loc[impacted, "exclusion_applied_by_rule"] = 1
        frame.loc[impacted, "exclusion_rule_id"] = "RULE_" + frame.loc[impacted, "exclusion_category"].astype(str).str.upper()
    elif scenario == "corporate_low_default_portfolio":
        mask = frame["portfolio"] == "Corporate"
        _set_defaults(frame, mask, 0.18, rng)

    if scenario != "data_quality_issues" and data_quality_level in {"medium", "high"}:
        frame = _add_extra_anomalies(frame, rng, data_quality_level)

    return frame.reset_index(drop=True)
