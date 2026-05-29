import pandas as pd
import pytest

from src.alerting.calibration_alerts import assign_status, build_calibration_alerts
from src.pd_backtesting.aggregation import build_standard_aggregations
from src.pd_backtesting.metrics import calculate_pd_metrics
from src.pd_backtesting.stat_tests import binomial_calibration_test, hosmer_lemeshow_test


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
        "minimum_volume": {"min_observations": 30, "min_defaults": 5},
    }


def sample_frame(defaults=10, observations=100, pd_value=0.10, portfolio="Retail"):
    flags = [1] * defaults + [0] * (observations - defaults)
    return pd.DataFrame(
        {
            "portfolio": [portfolio] * observations,
            "segment": ["Segment A"] * observations,
            "product_type": ["Loan"] * observations,
            "rating_grade": ["B"] * observations,
            "observation_date": [pd.Timestamp("2024-01-01")] * observations,
            "pd_estimate": [pd_value] * observations,
            "ead_at_observation": [100.0] * observations,
            "default_flag_12m": flags,
        }
    )


def test_core_pd_metrics():
    metrics = calculate_pd_metrics(sample_frame(defaults=8, observations=100, pd_value=0.05))

    assert metrics["observations"] == 100
    assert metrics["observed_defaults"] == 8
    assert metrics["odr"] == 0.08
    assert metrics["pd_mean"] == pytest.approx(0.05)
    assert metrics["ead_weighted_pd"] == pytest.approx(0.05)
    assert metrics["expected_defaults"] == pytest.approx(5.0)
    assert metrics["calibration_gap"] == pytest.approx(0.03)
    assert metrics["calibration_ratio"] == pytest.approx(1.6)
    assert metrics["ead_total"] == 10000.0
    assert metrics["represented_grades"] == 1


def test_standard_aggregations_include_requested_levels():
    aggregations = build_standard_aggregations(sample_frame())

    assert set(aggregations) == {
        "global",
        "portfolio",
        "segment",
        "product_type",
        "rating_grade",
        "observation_year",
        "portfolio_segment",
        "portfolio_rating_grade",
    }
    assert aggregations["portfolio"].iloc[0]["perimeter"] == "Retail"


def test_binomial_test_simple_case():
    result = binomial_calibration_test(
        observations=100,
        observed_defaults=20,
        pd_mean=0.10,
        min_observations=30,
        min_defaults=5,
    )

    assert result["test_interpretable"] is True
    assert 0 <= result["p_value_two_sided"] <= 1
    assert 0 <= result["p_value_one_sided_high"] <= 1
    assert result["p_value"] == result["p_value_one_sided_high"]
    assert 0 <= result["ci_lower"] <= result["ci_upper"] <= 1


def test_hosmer_lemeshow_test_simple_case():
    frame = pd.concat(
        [
            sample_frame(defaults=2, observations=40, pd_value=0.04),
            sample_frame(defaults=4, observations=40, pd_value=0.08),
            sample_frame(defaults=6, observations=40, pd_value=0.14),
            sample_frame(defaults=8, observations=40, pd_value=0.20),
        ],
        ignore_index=True,
    )

    result = hosmer_lemeshow_test(frame, n_buckets=4, min_observations=30, min_defaults=5)

    assert result["test_interpretable"] is True
    assert 0 <= result["p_value"] <= 1
    assert result["hl_degrees_freedom"] >= 1
    assert set(["bucket", "observed_defaults", "expected_defaults", "odr"]).issubset(result["hl_buckets"].columns)


def test_status_green_orange_red_grey():
    th = thresholds()

    assert assign_status(0.20, True, th) == "green"
    assert assign_status(0.03, True, th) == "orange"
    assert assign_status(0.005, True, th) == "red"
    assert assign_status(float("nan"), False, th) == "grey"


def test_low_volume_segment_is_grey():
    alerts = build_calibration_alerts(sample_frame(defaults=1, observations=20), thresholds())
    segment_alert = alerts[alerts["aggregation_level"] == "segment"].iloc[0]

    assert segment_alert["status"] == "grey"
    assert not segment_alert["test_interpretable"]
