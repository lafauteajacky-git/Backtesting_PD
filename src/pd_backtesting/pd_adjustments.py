from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_pd_components(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize raw, calibrated, floored and regulatory PD components."""
    columns = ["pd_raw", "pd_calibrated", "pd_before_floor", "pd_after_floor", "pd_regulatory"]
    rows = []
    for column in columns:
        if column in df.columns:
            rows.append({"component": column, "mean": df[column].mean(), "median": df[column].median(), "missing_rate": df[column].isna().mean()})
    return pd.DataFrame(rows)


def _status(value: float, thresholds: dict | None = None) -> str:
    cfg = (thresholds or {}).get("pd_adjustments", {})
    orange = cfg.get("impact_orange", 0.01)
    red = cfg.get("impact_red", 0.03)
    if pd.isna(value):
        return "grey"
    if value >= red:
        return "red"
    if value >= orange:
        return "orange"
    return "green"


def analyze_pd_floor_impact(df: pd.DataFrame, thresholds: dict | None = None) -> pd.DataFrame:
    """Measure impact of PD floors."""
    if "pd_floor_applied_flag" not in df.columns:
        return pd.DataFrame([{"status": "grey"}])
    flag = df["pd_floor_applied_flag"].fillna(0).astype(int)
    impact = (df["pd_after_floor"] - df["pd_before_floor"]).clip(lower=0) if {"pd_after_floor", "pd_before_floor"}.issubset(df.columns) else pd.Series(dtype=float)
    ead = pd.to_numeric(df.get("ead_at_observation", pd.Series(dtype=float)), errors="coerce")
    mean_impact = float(impact.mean()) if len(impact) else np.nan
    return pd.DataFrame([{"floor_count": int(flag.sum()), "floor_rate": float(flag.mean()), "ead_impacted": float(ead[flag.eq(1)].sum()), "mean_floor_impact": mean_impact, "status": _status(mean_impact, thresholds)}])


def analyze_moc_impact(df: pd.DataFrame, thresholds: dict | None = None) -> pd.DataFrame:
    """Measure margin of conservatism impact overall and by type."""
    if "margin_of_conservatism" not in df.columns:
        return pd.DataFrame([{"status": "grey"}])
    grouped = (
        df.groupby("margin_of_conservatism_type", dropna=False)
        .agg(observations=("observation_id", "size"), moc_mean=("margin_of_conservatism", "mean"), ead=("ead_at_observation", "sum"))
        .reset_index()
    )
    grouped["status"] = grouped["moc_mean"].map(lambda value: _status(value, thresholds))
    return grouped


def compare_raw_calibrated_regulatory_pd(df: pd.DataFrame) -> pd.DataFrame:
    """Compare PD layers by portfolio and segment."""
    group_cols = [col for col in ["portfolio", "segment"] if col in df.columns]
    return (
        df.groupby(group_cols, dropna=False)
        .agg(
            observations=("observation_id", "size"),
            pd_raw=("pd_raw", "mean"),
            pd_calibrated=("pd_calibrated", "mean"),
            pd_regulatory=("pd_regulatory", "mean"),
            floor_rate=("pd_floor_applied_flag", "mean"),
            moc_mean=("margin_of_conservatism", "mean"),
        )
        .reset_index()
    )
