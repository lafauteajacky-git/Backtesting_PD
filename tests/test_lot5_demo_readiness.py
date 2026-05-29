import pandas as pd
import pytest

from src.alerting.calibration_alerts import build_calibration_alerts
from src.config_utils import apply_threshold_profile
from src.data_generation.demo_scenarios import REQUIRED_COLUMNS, generate_demo_scenario, scenario_catalog
from src.pd_backtesting.validation import validate_observation_schema
from src.reporting.demo_narrative import generate_demo_narrative


def thresholds():
    return {
        "pd_backtesting": {"confidence_level": 0.95},
        "calibration_tests": {
            "binomial": {
                "alpha_orange": 0.05,
                "alpha_red": 0.01,
                "default_test_type": "one_sided_high",
                "confidence_interval_method": "wilson",
            }
        },
        "minimum_volume": {"min_observations": 30, "min_defaults": 5, "min_non_defaults": 30},
        "discrimination": {"auc_orange": 0.70, "auc_red": 0.60, "gini_orange": 0.40, "gini_red": 0.20, "ks_orange": 0.30, "ks_red": 0.20},
        "stability": {"psi_orange": 0.10, "psi_red": 0.25},
        "threshold_profiles": {
            "standard": {"calibration_tests": {"binomial": {"alpha_orange": 0.05, "alpha_red": 0.01}}},
            "conservative": {"calibration_tests": {"binomial": {"alpha_orange": 0.10, "alpha_red": 0.05}}},
        },
    }


@pytest.mark.parametrize("scenario", sorted(scenario_catalog()))
def test_each_demo_scenario_generates_required_columns(scenario):
    frame = generate_demo_scenario(
        scenario,
        retail_observations=1200,
        corporate_observations=600,
        data_quality_level="none",
        random_seed=123,
    )

    assert set(REQUIRED_COLUMNS).issubset(frame.columns)
    assert len(frame) == 1800
    assert frame["observation_date"].dt.year.nunique() >= 5


def test_demo_scenario_is_reproducible_with_seed():
    first = generate_demo_scenario("retail_underestimation", 1200, 600, random_seed=99, data_quality_level="none")
    second = generate_demo_scenario("retail_underestimation", 1200, 600, random_seed=99, data_quality_level="none")

    pd.testing.assert_frame_equal(first, second)


def test_problematic_scenario_produces_calibration_alerts():
    frame = generate_demo_scenario("retail_underestimation", 5000, 1000, random_seed=7, data_quality_level="none")
    alerts = build_calibration_alerts(frame, thresholds())

    assert alerts["status"].isin(["orange", "red"]).any()


def test_threshold_profile_selection():
    selected = apply_threshold_profile(thresholds(), "conservative")

    assert selected["calibration_tests"]["binomial"]["alpha_orange"] == 0.10
    assert selected["calibration_tests"]["binomial"]["alpha_red"] == 0.05
    assert selected["active_threshold_profile"] == "conservative"


def test_demo_narrative_generation():
    alerts = pd.DataFrame([{"status": "red"}, {"status": "orange"}, {"status": "grey"}])
    narrative = generate_demo_narrative("Scenario test", "Observer une alerte", alerts, pd.DataFrame(), pd.DataFrame())

    assert "Scenario presente" in narrative
    assert "1 alerte(s) rouge(s)" in narrative


def test_empty_dataset_and_missing_columns_validation():
    empty = pd.DataFrame(columns=REQUIRED_COLUMNS)
    missing = validate_observation_schema(["portfolio", "pd_estimate"])

    assert empty.empty
    assert "observation_id" in missing
    assert "default_flag_12m" in missing
