from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binom, binomtest, chi2


def binomial_calibration_test(
    observations: int,
    observed_defaults: int,
    pd_mean: float,
    confidence_level: float = 0.95,
    method: str = "wilson",
    test_type: str = "one_sided_high",
    min_observations: int = 30,
    min_defaults: int = 5,
) -> dict:
    """Run a robust binomial calibration test for PD backtesting.

    `one_sided_high` tests whether observed defaults are unusually high versus
    the average PD, which is the conservative signal for risk underestimation.
    """
    if (
        observations is None
        or observed_defaults is None
        or pd_mean is None
        or observations < min_observations
        or observed_defaults < min_defaults
        or not np.isfinite(pd_mean)
        or pd_mean <= 0
        or pd_mean > 1
    ):
        return {
            "test_interpretable": False,
            "p_value": np.nan,
            "p_value_two_sided": np.nan,
            "p_value_one_sided_high": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "test_type": test_type,
            "ci_method": method,
        }

    two_sided = float(binomtest(observed_defaults, observations, pd_mean).pvalue)
    one_sided_high = float(binom.sf(observed_defaults - 1, observations, pd_mean))
    selected = one_sided_high if test_type == "one_sided_high" else two_sided

    ci = binomtest(observed_defaults, observations, pd_mean).proportion_ci(
        confidence_level=confidence_level,
        method=method,
    )

    return {
        "test_interpretable": True,
        "p_value": selected,
        "p_value_two_sided": two_sided,
        "p_value_one_sided_high": one_sided_high,
        "ci_lower": float(ci.low),
        "ci_upper": float(ci.high),
        "test_type": test_type,
        "ci_method": method,
    }


def hosmer_lemeshow_test(
    observations: pd.DataFrame,
    n_buckets: int = 10,
    min_observations: int = 30,
    min_defaults: int = 5,
) -> dict:
    """Run a Hosmer-Lemeshow style calibration test by ordered PD buckets.

    The test compares observed and expected defaults across ordered buckets of
    `pd_estimate`. It is returned as non-interpretable when the filtered sample
    is too small, has too few defaults, or cannot be split into risk buckets.
    """
    frame = observations[
        observations["pd_estimate"].notna()
        & observations["pd_estimate"].between(0, 1, inclusive="right")
        & observations["default_flag_12m"].isin([0, 1])
    ].copy()
    observed_defaults = int(frame["default_flag_12m"].sum()) if not frame.empty else 0
    if (
        len(frame) < min_observations
        or observed_defaults < min_defaults
        or frame["pd_estimate"].nunique() < 2
    ):
        return {
            "test_interpretable": False,
            "hl_statistic": np.nan,
            "p_value": np.nan,
            "hl_p_value": np.nan,
            "hl_buckets": pd.DataFrame(),
        }

    bucket_count = min(n_buckets, frame["pd_estimate"].nunique(), len(frame))
    frame["bucket"] = pd.qcut(frame["pd_estimate"], q=bucket_count, duplicates="drop", labels=False)
    if frame["bucket"].nunique(dropna=True) < 2:
        return {
            "test_interpretable": False,
            "hl_statistic": np.nan,
            "p_value": np.nan,
            "hl_p_value": np.nan,
            "hl_buckets": pd.DataFrame(),
        }
    frame["bucket"] = frame["bucket"].astype(int) + 1

    buckets = (
        frame.groupby("bucket")
        .agg(
            observations=("default_flag_12m", "size"),
            observed_defaults=("default_flag_12m", "sum"),
            expected_defaults=("pd_estimate", "sum"),
            pd_mean=("pd_estimate", "mean"),
        )
        .reset_index()
    )
    buckets["observed_non_defaults"] = buckets["observations"] - buckets["observed_defaults"]
    buckets["expected_non_defaults"] = buckets["observations"] - buckets["expected_defaults"]
    eps = 1e-9
    statistic = (
        ((buckets["observed_defaults"] - buckets["expected_defaults"]) ** 2)
        / buckets["expected_defaults"].clip(lower=eps)
        + ((buckets["observed_non_defaults"] - buckets["expected_non_defaults"]) ** 2)
        / buckets["expected_non_defaults"].clip(lower=eps)
    ).sum()
    degrees = max(len(buckets) - 2, 1)
    p_value = float(chi2.sf(statistic, degrees))
    buckets["odr"] = buckets["observed_defaults"] / buckets["observations"]
    buckets["calibration_gap"] = buckets["odr"] - buckets["pd_mean"]

    return {
        "test_interpretable": True,
        "hl_statistic": float(statistic),
        "p_value": p_value,
        "hl_p_value": p_value,
        "hl_degrees_freedom": int(degrees),
        "hl_buckets": buckets,
    }
