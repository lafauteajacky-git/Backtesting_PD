from __future__ import annotations

import numpy as np
import pandas as pd

from src.pd_backtesting.metrics import calculate_pd_metrics


SHIFT_VARIABLES = [
    "portfolio",
    "segment",
    "product_type",
    "country_code",
    "sector_code",
    "borrower_size_class",
    "secured_flag",
    "collateral_type",
    "ltv_bucket",
    "rating_grade",
    "score_bucket",
    "pd_bucket",
]


def compare_population_mix(reference_df: pd.DataFrame, current_df: pd.DataFrame, variables: list[str] | None = None) -> pd.DataFrame:
    """Compare population shares between reference and current datasets."""
    rows = []
    for variable in variables or SHIFT_VARIABLES:
        if variable not in reference_df.columns or variable not in current_df.columns:
            continue
        ref = reference_df[variable].fillna("Missing").astype(str).value_counts(normalize=True)
        cur = current_df[variable].fillna("Missing").astype(str).value_counts(normalize=True)
        for bucket in sorted(set(ref.index).union(cur.index)):
            rows.append({"variable": variable, "bucket": bucket, "reference_share": ref.get(bucket, 0.0), "current_share": cur.get(bucket, 0.0), "share_change": cur.get(bucket, 0.0) - ref.get(bucket, 0.0)})
    return pd.DataFrame(rows)


def identify_population_shift_drivers(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    """Return top contributors to population mix changes."""
    mix = compare_population_mix(reference_df, current_df)
    if mix.empty:
        return mix
    mix["abs_share_change"] = mix["share_change"].abs()
    return mix.sort_values("abs_share_change", ascending=False).head(20)


def summarize_new_and_exited_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize new and exited customer flags."""
    return pd.DataFrame(
        [
            {
                "observations": len(df),
                "new_customers": int(df.get("new_customer_flag", pd.Series(dtype=int)).fillna(0).sum()),
                "exited_customers": int(df.get("exited_customer_flag", pd.Series(dtype=int)).fillna(0).sum()),
            }
        ]
    )


def diagnose_population_change(df: pd.DataFrame, reference_period, current_period, thresholds: dict | None = None) -> dict:
    """Diagnose whether changes are driven by mix, migration or population flow."""
    reference = df[df["observation_year"] == reference_period]
    current = df[df["observation_year"] == current_period]
    drivers = identify_population_shift_drivers(reference, current)
    ref_metrics = calculate_pd_metrics(reference) if not reference.empty else {}
    cur_metrics = calculate_pd_metrics(current) if not current.empty else {}
    max_change = float(drivers["abs_share_change"].max()) if not drivers.empty else np.nan
    cfg = (thresholds or {}).get("population_shift", {})
    orange = cfg.get("share_change_orange", 0.10)
    red = cfg.get("share_change_red", 0.20)
    status = "grey" if pd.isna(max_change) else "red" if max_change >= red else "orange" if max_change >= orange else "green"
    return {
        "drivers": drivers,
        "status": status,
        "pd_mean_change": cur_metrics.get("pd_mean", np.nan) - ref_metrics.get("pd_mean", np.nan),
        "odr_change": cur_metrics.get("odr", np.nan) - ref_metrics.get("odr", np.nan),
        "comment": "Les principaux changements de population sont listés par variable et bucket.",
    }
