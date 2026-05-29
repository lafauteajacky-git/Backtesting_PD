from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest


DEFAULT_GROUP_COLUMNS = ["portfolio", "segment", "product_type", "rating_grade", "period"]


def _status(
    observations: int,
    defaults_observed: int,
    calibration_gap: float,
    p_value: float,
    thresholds: dict,
) -> str:
    status_thresholds = thresholds.get("pd_backtesting", {}).get("status", {})

    min_observations = status_thresholds.get("grey_min_observations", 30)
    min_defaults = status_thresholds.get("grey_min_defaults", 1)
    if observations < min_observations or defaults_observed < min_defaults:
        return "grey"

    abs_gap = abs(calibration_gap)
    red_abs_gap = status_thresholds.get("red_abs_calibration_gap", 0.03)
    orange_abs_gap = status_thresholds.get("orange_abs_calibration_gap", 0.015)
    red_p_value = status_thresholds.get("red_p_value", 0.01)
    orange_p_value = status_thresholds.get("orange_p_value", 0.05)

    if abs_gap > red_abs_gap or p_value < red_p_value:
        return "red"
    if abs_gap > orange_abs_gap or p_value < orange_p_value:
        return "orange"
    return "green"


def _prepare_observations(observations: pd.DataFrame, period_frequency: str) -> pd.DataFrame:
    prepared = observations.copy()
    prepared["period"] = pd.to_datetime(prepared["observation_date"], errors="coerce").dt.to_period(
        period_frequency
    ).astype("string")

    valid = (
        prepared["pd_estimate"].notna()
        & prepared["pd_estimate"].between(0, 1, inclusive="right")
        & prepared["default_flag_12m"].isin([0, 1])
    )
    return prepared.loc[valid].copy()


def compute_pd_backtesting_metrics(
    observations: pd.DataFrame,
    thresholds: dict | None = None,
    group_columns: Sequence[str] | None = None,
    period_frequency: str = "M",
    confidence_level: float | None = None,
) -> pd.DataFrame:
    """Compute PD 12M backtesting indicators with binomial tests by group."""
    thresholds = thresholds or {}
    confidence_level = confidence_level or thresholds.get("pd_backtesting", {}).get(
        "confidence_level", 0.95
    )
    group_columns = list(group_columns or DEFAULT_GROUP_COLUMNS)

    prepared = _prepare_observations(observations, period_frequency)
    if prepared.empty:
        return pd.DataFrame(
            columns=[
                *group_columns,
                "observations",
                "pd_mean",
                "expected_defaults",
                "observed_defaults",
                "odr",
                "calibration_gap",
                "calibration_ratio",
                "binomial_p_value",
                "ci_lower",
                "ci_upper",
                "status",
            ]
        )

    rows = []
    for group_values, group in prepared.groupby(group_columns, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        observations_count = int(len(group))
        observed_defaults = int(group["default_flag_12m"].sum())
        pd_mean = float(group["pd_estimate"].mean())
        expected_defaults = float(group["pd_estimate"].sum())
        odr = observed_defaults / observations_count if observations_count else np.nan
        calibration_gap = odr - pd_mean
        calibration_ratio = odr / pd_mean if pd_mean > 0 else np.nan

        test = binomtest(observed_defaults, observations_count, pd_mean)
        confidence_interval = test.proportion_ci(
            confidence_level=confidence_level,
            method="exact",
        )

        row = dict(zip(group_columns, group_values))
        row.update(
            {
                "observations": observations_count,
                "pd_mean": pd_mean,
                "expected_defaults": expected_defaults,
                "observed_defaults": observed_defaults,
                "odr": odr,
                "calibration_gap": calibration_gap,
                "calibration_ratio": calibration_ratio,
                "binomial_p_value": float(test.pvalue),
                "ci_lower": float(confidence_interval.low),
                "ci_upper": float(confidence_interval.high),
                "status": _status(
                    observations_count,
                    observed_defaults,
                    calibration_gap,
                    float(test.pvalue),
                    thresholds,
                ),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def compute_portfolio_calibration(
    observations: pd.DataFrame,
    thresholds: dict | None = None,
    period_frequency: str = "M",
) -> pd.DataFrame:
    return compute_pd_backtesting_metrics(
        observations,
        thresholds=thresholds,
        group_columns=["portfolio", "period"],
        period_frequency=period_frequency,
    )
