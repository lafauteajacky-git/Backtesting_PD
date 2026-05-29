from __future__ import annotations

import numpy as np
import pandas as pd


def _distribution(df: pd.DataFrame, period_col: str, variable: str) -> pd.DataFrame:
    data = df[[period_col, variable]].copy()
    data[variable] = data[variable].fillna("Missing").astype(str)
    dist = data.groupby([period_col, variable], dropna=False).size().reset_index(name="count")
    dist["share"] = dist["count"] / dist.groupby(period_col)["count"].transform("sum")
    return dist


def calculate_rating_distribution(df: pd.DataFrame, period_col: str = "observation_year", grade_col: str = "rating_grade") -> pd.DataFrame:
    """Calculate rating distribution by period."""
    return _distribution(df, period_col, grade_col)


def calculate_score_distribution(df: pd.DataFrame, period_col: str = "observation_year", score_bucket_col: str = "score_bucket") -> pd.DataFrame:
    """Calculate score-bucket distribution by period."""
    return _distribution(df, period_col, score_bucket_col)


def calculate_pd_bucket_distribution(df: pd.DataFrame, period_col: str = "observation_year", pd_bucket_col: str = "pd_bucket") -> pd.DataFrame:
    """Calculate PD-bucket distribution by period."""
    return _distribution(df, period_col, pd_bucket_col)


def calculate_rds_psi(reference_df: pd.DataFrame, current_df: pd.DataFrame, variable: str) -> dict:
    """Calculate PSI and bucket contributions for one distribution variable."""
    if reference_df.empty or current_df.empty or variable not in reference_df.columns or variable not in current_df.columns:
        return {"psi": np.nan, "status": "grey", "contributors": pd.DataFrame()}
    ref = reference_df[variable].fillna("Missing").astype(str).value_counts(normalize=True)
    cur = current_df[variable].fillna("Missing").astype(str).value_counts(normalize=True)
    buckets = sorted(set(ref.index).union(cur.index))
    eps = 1e-6
    rows = []
    for bucket in buckets:
        ref_share = float(ref.get(bucket, 0.0))
        cur_share = float(cur.get(bucket, 0.0))
        contribution = (cur_share - ref_share) * np.log((cur_share + eps) / (ref_share + eps))
        rows.append({"variable": variable, "bucket": bucket, "reference_share": ref_share, "current_share": cur_share, "psi_contribution": contribution})
    contributors = pd.DataFrame(rows).sort_values("psi_contribution", ascending=False)
    psi = float(contributors["psi_contribution"].sum())
    return {"psi": psi, "status": assign_rds_status(psi), "contributors": contributors}


def assign_rds_status(psi: float, thresholds: dict | None = None) -> str:
    cfg = (thresholds or {}).get("rds", (thresholds or {}).get("stability", {}))
    orange = cfg.get("psi_orange", 0.10)
    red = cfg.get("psi_red", 0.25)
    if pd.isna(psi):
        return "grey"
    if psi >= red:
        return "red"
    if psi >= orange:
        return "orange"
    return "green"


def summarize_rds_stability(df: pd.DataFrame, reference_period, current_period, thresholds: dict | None = None) -> pd.DataFrame:
    """Summarize RDS PSI for rating, master scale, score bucket and PD bucket."""
    reference = df[df["observation_year"] == reference_period]
    current = df[df["observation_year"] == current_period]
    rows = []
    for variable in ["rating_grade", "master_scale_grade", "score_bucket", "pd_bucket"]:
        result = calculate_rds_psi(reference, current, variable)
        rows.append({"variable": variable, "reference_period": reference_period, "current_period": current_period, "psi": result["psi"], "status": assign_rds_status(result["psi"], thresholds)})
    return pd.DataFrame(rows)


def diagnose_rds_change(df: pd.DataFrame, reference_period, current_period) -> pd.DataFrame:
    """Identify top PSI contributors and population-flow drivers."""
    reference = df[df["observation_year"] == reference_period]
    current = df[df["observation_year"] == current_period]
    rows = []
    for variable in ["rating_grade", "score_bucket", "pd_bucket", "segment", "product_type", "country_code", "sector_code"]:
        if variable in df.columns:
            result = calculate_rds_psi(reference, current, variable)
            top = result["contributors"].head(5).copy()
            top["driver_type"] = "mix_change"
            rows.append(top)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
