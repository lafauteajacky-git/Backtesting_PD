from __future__ import annotations

import numpy as np
import pandas as pd


ESSENTIAL_COLUMNS = ["pd_estimate", "default_flag_12m", "observation_date", "rating_grade", "portfolio"]


def identify_eligible_population(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Flag observations eligible for regulatory PD backtesting."""
    frame = df.copy()
    frame["eligibility_exclusion_category"] = "eligible"
    invalid_pd = frame["pd_estimate"].isna() | ~frame["pd_estimate"].between(0, 1, inclusive="right")
    missing_essential = frame[[col for col in ESSENTIAL_COLUMNS if col in frame.columns]].isna().any(axis=1)
    conditions = [
        (frame.get("default_flag_at_observation", 0) == 1, "defaulted_at_observation"),
        (frame.get("application_scope_flag", 1) == 0, "out_of_scope"),
        (frame.get("model_applicability_flag", 1) == 0, "model_not_applicable"),
        (frame.get("data_complete_12m_flag", 1) == 0, "incomplete_performance_window"),
        (frame.get("maturity_before_12m_flag", 0) == 1, "closed_before_horizon"),
        (missing_essential, "missing_essential_data"),
        (invalid_pd, "invalid_pd"),
        (~frame["default_flag_12m"].isin([0, 1]), "missing_default_flag"),
    ]
    for mask, category in conditions:
        frame.loc[mask, "eligibility_exclusion_category"] = category
    frame["eligible_flag"] = (frame["eligibility_exclusion_category"] == "eligible").astype(int)
    return frame


def _status(rate: float, thresholds: dict | None = None) -> str:
    cfg = (thresholds or {}).get("eligibility", {})
    orange = cfg.get("exclusion_rate_orange", 0.05)
    red = cfg.get("exclusion_rate_red", 0.15)
    if pd.isna(rate):
        return "grey"
    if rate >= red:
        return "red"
    if rate >= orange:
        return "orange"
    return "green"


def summarize_population_eligibility(df: pd.DataFrame, thresholds: dict | None = None) -> pd.DataFrame:
    """Summarize eligible and excluded population volumes."""
    frame = identify_eligible_population(df, thresholds)
    total = len(frame)
    eligible = int(frame["eligible_flag"].sum())
    excluded = total - eligible
    rate = excluded / total if total else np.nan
    ead = pd.to_numeric(frame.get("ead_at_observation", pd.Series(dtype=float)), errors="coerce")
    return pd.DataFrame(
        [
            {
                "initial_observations": total,
                "eligible_observations": eligible,
                "excluded_observations": excluded,
                "exclusion_rate": rate,
                "excluded_ead": float(ead[frame["eligible_flag"].eq(0)].sum()) if total else 0.0,
                "status": _status(rate, thresholds),
            }
        ]
    )


def summarize_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize exclusions by category, portfolio and segment."""
    frame = identify_eligible_population(df)
    excluded = frame[frame["eligible_flag"].eq(0)].copy()
    if excluded.empty:
        return pd.DataFrame(columns=["exclusion_category", "portfolio", "segment", "observations", "ead_at_observation"])
    return (
        excluded.groupby(["eligibility_exclusion_category", "portfolio", "segment"], dropna=False)
        .agg(observations=("observation_id", "size"), ead_at_observation=("ead_at_observation", "sum"))
        .reset_index()
        .rename(columns={"eligibility_exclusion_category": "exclusion_category"})
    )


def build_population_waterfall(df: pd.DataFrame) -> pd.DataFrame:
    """Build a sequential waterfall from initial population to eligible population."""
    frame = identify_eligible_population(df)
    rows = [{"step": "Population initiale", "observations": len(frame)}]
    remaining = len(frame)
    for category, group in frame[frame["eligible_flag"].eq(0)].groupby("eligibility_exclusion_category", dropna=False):
        count = int(len(group))
        remaining -= count
        rows.append({"step": f"Exclusion - {category}", "observations": -count})
    rows.append({"step": "Population eligible backtesting", "observations": max(remaining, 0)})
    return pd.DataFrame(rows)
