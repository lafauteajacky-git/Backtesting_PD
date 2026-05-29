from __future__ import annotations

import numpy as np
import pandas as pd


def _valid(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["previous_rating_grade"].notna() & df["rating_grade"].notna()].copy()


def build_rating_migration_matrix(df: pd.DataFrame) -> dict:
    """Build rating migration matrices in counts and row percentages."""
    data = _valid(df)
    if data.empty:
        empty = pd.DataFrame()
        return {"count_matrix": empty, "percentage_matrix": empty}
    count = pd.crosstab(data["previous_rating_grade"], data["rating_grade"])
    pct = count.div(count.sum(axis=1), axis=0).fillna(0.0)
    return {"count_matrix": count, "percentage_matrix": pct}


def calculate_upgrade_downgrade_rates(df: pd.DataFrame) -> dict:
    """Calculate stable, upgrade and downgrade rates."""
    data = df[df["notch_change"].notna()].copy()
    total = len(data)
    if total == 0:
        return {"stable_rate": np.nan, "upgrade_rate": np.nan, "downgrade_rate": np.nan, "large_migration_rate": np.nan}
    return {
        "stable_rate": float((data["notch_change"] == 0).mean()),
        "upgrade_rate": float((data["notch_change"] < 0).mean()),
        "downgrade_rate": float((data["notch_change"] > 0).mean()),
        "large_migration_rate": float((data["notch_change"].abs() > 2).mean()),
    }


def calculate_notch_migration_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return distribution of notch changes."""
    data = df[df["notch_change"].notna()].copy()
    if data.empty:
        return pd.DataFrame(columns=["notch_change", "count", "share"])
    dist = data["notch_change"].value_counts().sort_index().reset_index()
    dist.columns = ["notch_change", "count"]
    dist["share"] = dist["count"] / dist["count"].sum()
    return dist


def summarize_migration(df: pd.DataFrame, thresholds: dict | None = None) -> pd.DataFrame:
    """Summarize migration rates and status."""
    rates = calculate_upgrade_downgrade_rates(df)
    data = df[df["notch_change"].notna()]
    if data.empty:
        status = "grey"
    else:
        cfg = (thresholds or {}).get("migration", {})
        orange = cfg.get("downgrade_rate_orange", 0.25)
        red = cfg.get("downgrade_rate_red", 0.40)
        status = "red" if rates["downgrade_rate"] >= red else "orange" if rates["downgrade_rate"] >= orange else "green"
    return pd.DataFrame([{**rates, "observations": len(data), "mean_notch_change": float(data["notch_change"].mean()) if len(data) else np.nan, "status": status}])


def identify_material_migration_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Identify segments with material downgrades."""
    data = df[df["notch_change"].notna()].copy()
    if data.empty or "segment" not in data.columns:
        return pd.DataFrame()
    return (
        data.assign(downgrade=data["notch_change"] > 0)
        .groupby("segment")
        .agg(observations=("observation_id", "size"), downgrade_rate=("downgrade", "mean"), mean_notch_change=("notch_change", "mean"))
        .reset_index()
        .sort_values(["downgrade_rate", "observations"], ascending=False)
    )
