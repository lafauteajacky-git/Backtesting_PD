import pandas as pd
import pytest

from src.alerting.discrimination_alerts import assign_discrimination_status
from src.alerting.global_status import combine_statuses
from src.alerting.stability_alerts import assign_stability_status
from app.streamlit_app import build_model_summary
from src.pd_backtesting.discrimination import calculate_discrimination_metrics
from src.pd_backtesting.stability import calculate_psi


def thresholds():
    return {
        "minimum_volume": {"min_observations": 4, "min_defaults": 1, "min_non_defaults": 1},
        "discrimination": {
            "auc_orange": 0.70,
            "auc_red": 0.60,
            "gini_orange": 0.40,
            "gini_red": 0.20,
            "ks_orange": 0.30,
            "ks_red": 0.20,
        },
        "stability": {"psi_orange": 0.10, "psi_red": 0.25},
    }


def discrimination_frame(defaults=(1, 1), non_defaults=(0, 0), pd_values=(0.9, 0.8, 0.2, 0.1)):
    flags = list(defaults) + list(non_defaults)
    return pd.DataFrame(
        {
            "default_flag_12m": flags,
            "pd_estimate": list(pd_values),
            "portfolio": ["Retail"] * len(flags),
            "segment": ["A"] * len(flags),
            "model_id": ["M1"] * len(flags),
            "observation_date": [pd.Timestamp("2024-01-01")] * len(flags),
        }
    )


def test_auc_gini_ks_simple_case():
    result = calculate_discrimination_metrics(
        discrimination_frame(),
        min_observations=4,
        min_defaults=1,
        min_non_defaults=1,
    )

    assert result["is_interpretable"] is True
    assert result["auc"] == pytest.approx(1.0)
    assert result["gini"] == pytest.approx(1.0)
    assert result["ks"] == pytest.approx(1.0)
    assert not result["roc_curve"].empty
    assert not result["cap_curve"].empty


def test_discrimination_without_defaults_is_grey():
    frame = discrimination_frame(defaults=(0, 0), non_defaults=(0, 0))
    result = calculate_discrimination_metrics(frame, 4, 1, 1)

    assert result["is_interpretable"] is False
    assert result["status"] == "grey"


def test_discrimination_without_non_defaults_is_grey():
    frame = discrimination_frame(defaults=(1, 1), non_defaults=(1, 1))
    result = calculate_discrimination_metrics(frame, 4, 1, 1)

    assert result["is_interpretable"] is False
    assert result["status"] == "grey"


def test_discrimination_low_volume_is_grey():
    frame = discrimination_frame().head(3)
    result = calculate_discrimination_metrics(frame, 4, 1, 1)

    assert result["is_interpretable"] is False
    assert result["status"] == "grey"


def test_psi_simple_distributions():
    reference = pd.Series(["A", "A", "B", "B"])
    current = pd.Series(["A", "B", "B", "B"])
    result = calculate_psi(reference, current, min_observations=4)

    assert result["is_calculable"] is True
    assert result["psi"] > 0
    assert "bucket" in result["reference_distribution"].columns
    assert "bucket" in result["current_distribution"].columns


def test_stability_status_green_orange_red_grey():
    th = thresholds()

    assert assign_stability_status(0.05, True, th) == "green"
    assert assign_stability_status(0.15, True, th) == "orange"
    assert assign_stability_status(0.30, True, th) == "red"
    assert assign_stability_status(float("nan"), False, th) == "grey"


def test_discrimination_status_green_orange_red_grey():
    th = thresholds()

    assert assign_discrimination_status({"auc": 0.75, "gini": 0.50, "ks": 0.35, "is_interpretable": True}, th) == "green"
    assert assign_discrimination_status({"auc": 0.65, "gini": 0.30, "ks": 0.25, "is_interpretable": True}, th) == "orange"
    assert assign_discrimination_status({"auc": 0.55, "gini": 0.10, "ks": 0.10, "is_interpretable": True}, th) == "red"
    assert assign_discrimination_status({"auc": float("nan"), "gini": float("nan"), "ks": float("nan"), "is_interpretable": False}, th) == "grey"


def test_global_status_rule():
    assert combine_statuses(["green", "orange", "green"]) == "orange"
    assert combine_statuses(["green", "red", "orange"]) == "red"
    assert combine_statuses(["grey", "grey"]) == "grey"
    assert combine_statuses(["green", "green", "grey"]) == "green"


def test_model_summary_handles_empty_stability_table():
    observations = pd.DataFrame(
        {
            "model_id": ["M1"] * 4,
            "portfolio": ["Retail"] * 4,
            "observation_date": [pd.Timestamp("2024-01-01")] * 4,
            "pd_estimate": [0.1] * 4,
            "default_flag_12m": [1, 0, 0, 0],
            "rating_grade": ["A", "B", "B", "C"],
        }
    )
    calibration = pd.DataFrame(
        [{"aggregation_level": "portfolio", "perimeter": "Retail", "status": "green"}]
    )
    discrimination = pd.DataFrame(
        [
            {
                "aggregation_level": "portfolio",
                "perimeter": "Retail",
                "status": "green",
                "auc": 0.75,
                "gini": 0.50,
                "ks": 0.35,
            }
        ]
    )

    summary = build_model_summary(observations, calibration, discrimination, pd.DataFrame(), 2024)

    assert summary.iloc[0]["stability_status"] == "grey"
    assert summary.iloc[0]["statut_global"] == "green"
