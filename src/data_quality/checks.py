import pandas as pd


DQ_CHECKS = {
    "missing_observation_id": "observation_id manquant",
    "duplicate_observation_id": "doublons observation_id",
    "missing_pd_estimate": "pd_estimate manquante",
    "pd_estimate_out_of_bounds": "pd_estimate hors bornes 0 < PD <= 1",
    "missing_default_flag_12m": "default_flag_12m manquant",
    "missing_observation_date": "observation_date manquante",
    "missing_portfolio": "portfolio manquant",
    "missing_rating_grade": "rating_grade manquant",
    "invalid_performance_window_months": "performance_window_months invalide",
    "missing_default_date_for_default": "default_date manquante lorsque default_flag_12m = 1",
}


def run_data_quality_checks(
    observations: pd.DataFrame,
    expected_performance_window_months: int = 12,
) -> pd.DataFrame:
    """Return one row per data quality control."""
    masks = {
        "missing_observation_id": observations["observation_id"].isna(),
        "duplicate_observation_id": observations["observation_id"].duplicated(keep=False)
        & observations["observation_id"].notna(),
        "missing_pd_estimate": observations["pd_estimate"].isna(),
        "pd_estimate_out_of_bounds": observations["pd_estimate"].notna()
        & ~observations["pd_estimate"].between(0, 1, inclusive="right"),
        "missing_default_flag_12m": observations["default_flag_12m"].isna(),
        "missing_observation_date": observations["observation_date"].isna(),
        "missing_portfolio": observations["portfolio"].isna(),
        "missing_rating_grade": observations["rating_grade"].isna(),
        "invalid_performance_window_months": observations[
            "performance_window_months"
        ].isna()
        | (observations["performance_window_months"] != expected_performance_window_months),
        "missing_default_date_for_default": (observations["default_flag_12m"] == 1)
        & observations["default_date"].isna(),
    }

    total = len(observations)
    rows = []
    for check_id, mask in masks.items():
        failing_rows = int(mask.sum())
        rows.append(
            {
                "check_id": check_id,
                "description": DQ_CHECKS[check_id],
                "failing_rows": failing_rows,
                "failure_rate": failing_rows / total if total else 0.0,
                "status": "KO" if failing_rows else "OK",
            }
        )

    return pd.DataFrame(rows)
