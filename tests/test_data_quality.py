import pandas as pd

from src.data_quality.checks import run_data_quality_checks


def test_data_quality_checks_detect_expected_failures():
    observations = pd.DataFrame(
        {
            "observation_id": [1, 1, None],
            "pd_estimate": [0.01, 1.2, None],
            "default_flag_12m": [1, None, 0],
            "observation_date": [pd.Timestamp("2022-01-01"), pd.NaT, pd.Timestamp("2022-01-03")],
            "portfolio": ["Retail", None, "Corporate"],
            "rating_grade": ["A", "B", None],
            "performance_window_months": [12, 6, 12],
            "default_date": [pd.NaT, pd.NaT, pd.NaT],
        }
    )

    results = run_data_quality_checks(observations)
    failures = dict(zip(results["check_id"], results["failing_rows"]))

    assert failures["missing_observation_id"] == 1
    assert failures["duplicate_observation_id"] == 2
    assert failures["missing_pd_estimate"] == 1
    assert failures["pd_estimate_out_of_bounds"] == 1
    assert failures["missing_default_flag_12m"] == 1
    assert failures["missing_observation_date"] == 1
    assert failures["missing_portfolio"] == 1
    assert failures["missing_rating_grade"] == 1
    assert failures["invalid_performance_window_months"] == 1
    assert failures["missing_default_date_for_default"] == 1
