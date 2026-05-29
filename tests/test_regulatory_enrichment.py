import pandas as pd
import pytest

from src.data_generation.demo_scenarios import REQUIRED_COLUMNS, generate_demo_scenario
from src.data_quality.exclusion_analysis import calculate_exclusion_ead_impact, generate_exclusion_findings, summarize_exclusions_by_category
from src.pd_backtesting.eligibility import build_population_waterfall, identify_eligible_population, summarize_population_eligibility
from src.pd_backtesting.low_default import identify_low_default_segments, recommend_ldp_treatment
from src.pd_backtesting.migration import build_rating_migration_matrix, calculate_upgrade_downgrade_rates
from src.pd_backtesting.pd_adjustments import analyze_moc_impact, analyze_pd_floor_impact, compare_raw_calibrated_regulatory_pd
from src.pd_backtesting.philosophy import (
    analyze_pd_volatility_by_philosophy,
    compare_pd_by_philosophy,
    compare_rds_by_philosophy,
    generate_philosophy_commentary,
    summarize_model_philosophy,
)
from src.pd_backtesting.population_shift import diagnose_population_change, summarize_new_and_exited_customers
from src.pd_backtesting.rds import calculate_rds_psi, diagnose_rds_change, summarize_rds_stability


def enriched_frame(scenario="rds_stable_population"):
    return generate_demo_scenario(scenario, retail_observations=800, corporate_observations=200, data_quality_level="none", random_seed=123)


def test_enriched_schema_columns_and_categories():
    frame = enriched_frame()

    assert set(REQUIRED_COLUMNS).issubset(frame.columns)
    assert set(frame["model_philosophy"].dropna().unique()).issubset({"PIT", "TTC", "Hybrid"})
    assert set(frame["pd_type"].dropna().unique()).issubset({"regulatory_12m", "origination_12m", "behavioural_12m", "monitoring_12m"})
    assert frame["pd_horizon_months"].dropna().eq(12).all()
    assert set(frame["rating_migration"].dropna().unique()).issubset({"stable", "upgrade", "downgrade", "new", "exited"})
    assert not any("lifetime" in column.lower() for column in frame.columns)


@pytest.mark.parametrize(
    "scenario",
    [
        "rds_stable_population",
        "rds_population_shift",
        "rating_migration_deterioration",
        "pit_vs_ttc_comparison",
        "pit_ttc_coherent",
        "pit_ttc_incoherent",
        "pd_floor_and_moc",
        "regulatory_eligibility_issues",
        "exclusion_analysis",
        "corporate_low_default_portfolio",
    ],
)
def test_new_scenarios_generate_without_error(scenario):
    frame = enriched_frame(scenario)

    assert not frame.empty
    assert set(REQUIRED_COLUMNS).issubset(frame.columns)


def test_population_eligibility_and_waterfall():
    frame = enriched_frame("regulatory_eligibility_issues")
    eligible = identify_eligible_population(frame)
    summary = summarize_population_eligibility(frame)
    waterfall = build_population_waterfall(frame)

    assert "eligible_flag" in eligible.columns
    assert summary.iloc[0]["excluded_observations"] > 0
    assert waterfall.iloc[0]["step"] == "Population initiale"
    assert waterfall.iloc[-1]["step"] == "Population eligible backtesting"


def test_rds_psi_and_top_contributors():
    frame = enriched_frame("rds_population_shift")
    periods = sorted(frame["observation_year"].dropna().unique())
    reference, current = periods[0], periods[-1]
    summary = summarize_rds_stability(frame, reference, current)
    drivers = diagnose_rds_change(frame, reference, current)
    psi = calculate_rds_psi(frame[frame["observation_year"] == reference], frame[frame["observation_year"] == current], "rating_grade")

    assert not summary.empty
    assert psi["psi"] >= 0
    assert not drivers.empty


def test_migration_matrix_and_rates():
    frame = enriched_frame("rating_migration_deterioration")
    matrices = build_rating_migration_matrix(frame)
    rates = calculate_upgrade_downgrade_rates(frame)

    assert not matrices["count_matrix"].empty
    assert 0 <= rates["downgrade_rate"] <= 1


def test_pit_ttc_commentary_without_abusive_conclusion():
    frame = enriched_frame("pit_vs_ttc_comparison")
    summary = summarize_model_philosophy(frame)
    comment = generate_philosophy_commentary(frame)

    assert not summary.empty
    assert "meilleur" not in comment.lower()


def test_pit_ttc_metrics_by_philosophy():
    frame = enriched_frame("pit_ttc_coherent")
    summary = compare_pd_by_philosophy(frame)

    assert {"pd_mean", "odr", "calibration_gap", "calibration_ratio"}.issubset(summary.columns)
    assert summary["model_philosophy"].isin(["PIT", "TTC", "Hybrid"]).any()


def test_pit_ttc_volatility_and_rds_by_philosophy():
    frame = enriched_frame("pit_ttc_incoherent")
    volatility = analyze_pd_volatility_by_philosophy(frame)
    periods = sorted(frame["observation_year"].dropna().unique())
    rds = compare_rds_by_philosophy(frame, periods[0], periods[-1])

    assert not volatility.empty
    assert "pd_mean_volatility" in volatility.columns
    assert not rds.empty
    assert {"psi", "status"}.issubset(rds.columns)


def test_model_philosophy_missing_is_unknown():
    frame = enriched_frame().drop(columns=["model_philosophy"])
    summary = summarize_model_philosophy(frame)
    comment = generate_philosophy_commentary(frame)

    assert summary["model_philosophy"].tolist() == ["Unknown"]
    assert "n'est pas renseignée" in comment


def test_floor_and_moc_impacts():
    frame = enriched_frame("pd_floor_and_moc")
    floor = analyze_pd_floor_impact(frame)
    moc = analyze_moc_impact(frame)
    layers = compare_raw_calibrated_regulatory_pd(frame)

    assert floor.iloc[0]["floor_count"] > 0
    assert not moc.empty
    assert {"pd_raw", "pd_calibrated", "pd_regulatory"}.issubset(layers.columns)


def test_low_default_detection_and_recommendation():
    frame = enriched_frame("corporate_low_default_portfolio")
    ldp = identify_low_default_segments(frame)

    assert ldp["status"].isin(["orange", "grey", "green"]).all()
    assert "pluriannuel" in recommend_ldp_treatment(frame)


def test_exclusion_analysis_and_findings():
    frame = enriched_frame("exclusion_analysis")
    by_category = summarize_exclusions_by_category(frame)
    ead = calculate_exclusion_ead_impact(frame)
    findings = generate_exclusion_findings(frame)

    assert not by_category.empty
    assert ead.iloc[0]["excluded_ead"] >= 0
    assert isinstance(findings, pd.DataFrame)


def test_population_shift_diagnostic_new_and_exited():
    frame = enriched_frame("rds_population_shift")
    periods = sorted(frame["observation_year"].dropna().unique())
    reference, current = periods[0], periods[-1]
    diagnostic = diagnose_population_change(frame, reference, current)
    new_exited = summarize_new_and_exited_customers(frame)

    assert diagnostic["status"] in {"green", "orange", "red", "grey"}
    assert new_exited.iloc[0]["observations"] == len(frame)
