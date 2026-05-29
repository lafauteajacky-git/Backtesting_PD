import pandas as pd

from src.pd_backtesting.calibration import compute_pd_backtesting_metrics


def _thresholds():
    return {
        "pd_backtesting": {
            "confidence_level": 0.95,
            "status": {
                "grey_min_observations": 5,
                "grey_min_defaults": 1,
                "orange_abs_calibration_gap": 0.10,
                "red_abs_calibration_gap": 0.20,
                "orange_p_value": 0.05,
                "red_p_value": 0.01,
            },
        }
    }


def test_pd_backtesting_metrics_compute_core_indicators():
    observations = pd.DataFrame(
        {
            "portfolio": ["Retail"] * 10,
            "segment": ["Consumer"] * 10,
            "product_type": ["Loan"] * 10,
            "rating_grade": ["B"] * 10,
            "observation_date": [pd.Timestamp("2024-01-15")] * 10,
            "pd_estimate": [0.10] * 10,
            "default_flag_12m": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        }
    )

    result = compute_pd_backtesting_metrics(observations, thresholds=_thresholds())
    row = result.iloc[0]

    assert row["portfolio"] == "Retail"
    assert row["period"] == "2024-01"
    assert row["observations"] == 10
    assert row["observed_defaults"] == 1
    assert row["pd_mean"] == 0.10
    assert row["expected_defaults"] == 1.0
    assert row["odr"] == 0.10
    assert row["calibration_gap"] == 0.0
    assert row["calibration_ratio"] == 1.0
    assert 0 <= row["binomial_p_value"] <= 1
    assert 0 <= row["ci_lower"] <= row["odr"] <= row["ci_upper"] <= 1
    assert row["status"] == "green"


def test_pd_backtesting_metrics_aggregate_by_requested_dimensions():
    observations = pd.DataFrame(
        {
            "portfolio": ["Retail", "Retail", "Corporate"],
            "segment": ["A", "A", "B"],
            "product_type": ["Loan", "Loan", "Credit Line"],
            "rating_grade": ["B", "B", "C"],
            "observation_date": [
                pd.Timestamp("2024-01-15"),
                pd.Timestamp("2024-01-20"),
                pd.Timestamp("2024-02-01"),
            ],
            "pd_estimate": [0.10, 0.20, 0.05],
            "default_flag_12m": [0, 1, 0],
        }
    )

    result = compute_pd_backtesting_metrics(observations, thresholds=_thresholds())

    assert len(result) == 2
    assert set(result["portfolio"]) == {"Retail", "Corporate"}
    assert set(result["period"]) == {"2024-01", "2024-02"}


def test_pd_backtesting_metrics_apply_grey_status_for_low_volume():
    observations = pd.DataFrame(
        {
            "portfolio": ["Corporate"] * 3,
            "segment": ["SME"] * 3,
            "product_type": ["Loan"] * 3,
            "rating_grade": ["A"] * 3,
            "observation_date": [pd.Timestamp("2024-01-15")] * 3,
            "pd_estimate": [0.01] * 3,
            "default_flag_12m": [0, 0, 0],
        }
    )

    result = compute_pd_backtesting_metrics(observations, thresholds=_thresholds())

    assert result.iloc[0]["status"] == "grey"


def test_pd_backtesting_metrics_apply_red_status_for_large_gap():
    observations = pd.DataFrame(
        {
            "portfolio": ["Retail"] * 20,
            "segment": ["Cards"] * 20,
            "product_type": ["Card"] * 20,
            "rating_grade": ["D"] * 20,
            "observation_date": [pd.Timestamp("2024-01-15")] * 20,
            "pd_estimate": [0.05] * 20,
            "default_flag_12m": [1] * 10 + [0] * 10,
        }
    )

    result = compute_pd_backtesting_metrics(observations, thresholds=_thresholds())

    assert result.iloc[0]["status"] == "red"
