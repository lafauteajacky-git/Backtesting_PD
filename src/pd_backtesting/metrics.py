import numpy as np
import pandas as pd


def _valid_pd_flags(observations: pd.DataFrame) -> pd.DataFrame:
    """Return rows usable for PD calibration metrics."""
    return observations[
        observations["pd_estimate"].notna()
        & observations["pd_estimate"].between(0, 1, inclusive="right")
        & observations["default_flag_12m"].isin([0, 1])
    ].copy()


def calculate_pd_metrics(observations: pd.DataFrame) -> dict:
    """Compute core PD calibration metrics for a pandas DataFrame.

    Invalid PD values and missing default flags are excluded from rate-based
    metrics. The raw input count is preserved in `raw_observations`, while
    `observations` is the count used in calculations.
    """
    valid = _valid_pd_flags(observations)
    observations_count = int(len(valid))
    observed_defaults = int(valid["default_flag_12m"].sum()) if observations_count else 0
    pd_mean = float(valid["pd_estimate"].mean()) if observations_count else np.nan
    odr = observed_defaults / observations_count if observations_count else np.nan
    expected_defaults = float(valid["pd_estimate"].sum()) if observations_count else 0.0
    calibration_gap = odr - pd_mean if pd.notna(odr) and pd.notna(pd_mean) else np.nan
    calibration_ratio = odr / pd_mean if pd.notna(odr) and pd_mean and pd_mean > 0 else np.nan

    if "ead_at_observation" in valid.columns:
        ead = pd.to_numeric(valid["ead_at_observation"], errors="coerce")
        ead_total = float(ead.sum(skipna=True))
        weighted_pd = (
            float((valid["pd_estimate"] * ead).sum(skipna=True) / ead_total)
            if ead_total > 0
            else np.nan
        )
    else:
        ead_total = np.nan
        weighted_pd = np.nan

    represented_grades = (
        int(valid["rating_grade"].dropna().nunique()) if "rating_grade" in valid.columns else 0
    )

    return {
        "raw_observations": int(len(observations)),
        "observations": observations_count,
        "observed_defaults": observed_defaults,
        "defaults": observed_defaults,
        "odr": odr,
        "observed_default_rate": odr,
        "pd_mean": pd_mean,
        "mean_pd": pd_mean,
        "ead_weighted_pd": weighted_pd,
        "expected_defaults": expected_defaults,
        "calibration_gap": calibration_gap,
        "calibration_ratio": calibration_ratio,
        "ead_total": ead_total,
        "represented_grades": represented_grades,
    }


def portfolio_metrics(observations: pd.DataFrame) -> dict:
    """Backward-compatible alias for headline PD metrics."""
    return calculate_pd_metrics(observations)


def metrics_by_portfolio(observations: pd.DataFrame) -> pd.DataFrame:
    """Compute core metrics grouped by portfolio."""
    rows = []
    for portfolio, group in observations.groupby("portfolio", dropna=False):
        metrics = calculate_pd_metrics(group)
        metrics["portfolio"] = "Missing" if pd.isna(portfolio) else portfolio
        rows.append(metrics)

    return pd.DataFrame(rows)[
        [
            "portfolio",
            "observations",
            "observed_defaults",
            "pd_mean",
            "odr",
            "expected_defaults",
            "calibration_gap",
            "calibration_ratio",
            "ead_total",
            "represented_grades",
            "mean_pd",
            "observed_default_rate",
            "defaults",
        ]
    ]
