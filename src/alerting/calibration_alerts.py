import pandas as pd

from src.pd_backtesting.aggregation import concat_standard_aggregations
from src.pd_backtesting.stat_tests import binomial_calibration_test


COMMENTS = {
    "green": "Aucun écart statistiquement significatif n’est détecté au seuil paramétré.",
    "orange": "Un écart de calibration est détecté au seuil de 5 %. Une analyse complémentaire est recommandée.",
    "red": "Un écart de calibration significatif est détecté au seuil de 1 %. Une investigation approfondie est recommandée.",
    "grey": "Le test n’est pas interprétable en raison d’un volume insuffisant ou de données manquantes.",
}


def assign_status(p_value: float, test_interpretable: bool, thresholds: dict) -> str:
    """Assign a traffic-light status from thresholds."""
    if not test_interpretable or pd.isna(p_value):
        return "grey"

    config = thresholds["calibration_tests"]["binomial"]
    alpha_orange = config["alpha_orange"]
    alpha_red = config["alpha_red"]

    if p_value < alpha_red:
        return "red"
    if p_value < alpha_orange:
        return "orange"
    return "green"


def build_calibration_alerts(
    observations: pd.DataFrame,
    thresholds: dict,
    aggregations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Produce calibration alerts for each aggregated perimeter."""
    config = thresholds["calibration_tests"]["binomial"]
    volume = thresholds["minimum_volume"]
    confidence_level = thresholds.get("pd_backtesting", {}).get("confidence_level", 0.95)
    aggregated = aggregations if aggregations is not None else concat_standard_aggregations(observations)

    rows = []
    for _, row in aggregated.iterrows():
        test = binomial_calibration_test(
            observations=int(row["observations"]),
            observed_defaults=int(row["observed_defaults"]),
            pd_mean=float(row["pd_mean"]) if pd.notna(row["pd_mean"]) else float("nan"),
            confidence_level=confidence_level,
            method=config.get("confidence_interval_method", "wilson"),
            test_type=config.get("default_test_type", "one_sided_high"),
            min_observations=volume["min_observations"],
            min_defaults=volume["min_defaults"],
        )
        status = assign_status(test["p_value"], test["test_interpretable"], thresholds)

        output = row.to_dict()
        output.update(test)
        output["status"] = status
        output["comment"] = COMMENTS[status]
        rows.append(output)

    preferred_columns = [
        "perimeter",
        "aggregation_level",
        "observations",
        "observed_defaults",
        "pd_mean",
        "odr",
        "expected_defaults",
        "p_value",
        "p_value_two_sided",
        "p_value_one_sided_high",
        "ci_lower",
        "ci_upper",
        "status",
        "comment",
    ]
    result = pd.DataFrame(rows)
    remaining = [column for column in result.columns if column not in preferred_columns]
    return result[preferred_columns + remaining]
