import pandas as pd

from src.alerting.monotonicity_alerts import assign_monotonicity_status, build_monotonicity_alert
from src.pd_backtesting.interpretability import add_interpretability
from src.pd_backtesting.monotonicity import calculate_grade_monotonicity
from src.reporting.findings import generate_validation_findings
from src.reporting.test_mapping import build_test_mapping


def thresholds():
    return {
        "minimum_volume": {"min_observations": 30, "min_defaults": 5, "min_non_defaults": 30},
        "monotonicity": {
            "min_observations_per_grade": 10,
            "min_defaults_total": 1,
            "material_odr_decrease": 0.05,
        },
    }


def frame_from_odrs(odrs):
    rows = []
    grades = ["A", "B", "C", "D"][: len(odrs)]
    for grade, odr in zip(grades, odrs):
        defaults = int(odr * 100)
        rows.extend(
            {
                "rating_grade": grade,
                "default_flag_12m": 1 if idx < defaults else 0,
                "pd_estimate": max(0.001, odr),
            }
            for idx in range(100)
        )
    return pd.DataFrame(rows)


def test_monotonicity_without_violation():
    result, alert = build_monotonicity_alert(frame_from_odrs([0.01, 0.03, 0.06, 0.10]), thresholds())

    assert not result["violation_monotonicity"].any()
    assert alert["status"] == "green"


def test_monotonicity_with_one_violation():
    result = calculate_grade_monotonicity(frame_from_odrs([0.01, 0.08, 0.07, 0.10]), min_observations_per_grade=10)

    assert result["violation_monotonicity"].sum() == 1
    assert assign_monotonicity_status(result, thresholds()) == "orange"


def test_monotonicity_with_multiple_violations():
    result = calculate_grade_monotonicity(frame_from_odrs([0.10, 0.08, 0.06, 0.05]), min_observations_per_grade=10)

    assert result["violation_monotonicity"].sum() >= 2
    assert assign_monotonicity_status(result, thresholds()) == "red"


def test_monotonicity_low_volume_is_grey():
    small = frame_from_odrs([0.01, 0.02]).head(8)
    result = calculate_grade_monotonicity(small, min_observations_per_grade=10)

    assert assign_monotonicity_status(result, thresholds()) == "grey"


def test_generate_validation_findings():
    alerts = pd.DataFrame(
        [
            {
                "test_family": "Calibration",
                "perimeter": "Retail",
                "status": "red",
                "comment": "Ecart significatif",
                "p_value": 0.001,
            }
        ]
    )

    findings = generate_validation_findings(alerts)

    assert findings.iloc[0]["finding_id"] == "F-001"
    assert findings.iloc[0]["niveau_de_severite"] == "Haute"


def test_test_mapping_is_present():
    mapping = build_test_mapping()

    assert {"theme", "nom_du_test", "objectif", "niveau_analyse", "indicateur_produit", "statut"}.issubset(mapping.columns)
    assert "Data quality" in set(mapping["theme"])


def test_interpretability_column_is_added():
    frame = pd.DataFrame([{"observations": 100, "observed_defaults": 10, "pd_mean": 0.1, "odr": 0.1}])
    result = add_interpretability(frame, thresholds(), "calibration")

    assert "interpretabilite" in result.columns
    assert result.iloc[0]["interpretabilite"] == "Interprétable"
