from __future__ import annotations

import pandas as pd


def identify_low_default_segments(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Identify low-default segments using configurable volume thresholds."""
    cfg = (config or {}).get("low_default", {})
    min_defaults = cfg.get("min_defaults", 5)
    min_observations = cfg.get("min_observations", 30)
    rows = []
    for keys, group in df.groupby([col for col in ["portfolio", "segment"] if col in df.columns], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        defaults = int(group["default_flag_12m"].fillna(0).sum())
        observations = int(len(group))
        years = int(group["observation_year"].nunique()) if "observation_year" in group.columns else 0
        flag = defaults < min_defaults or observations < min_observations
        status = "grey" if observations < min_observations else "orange" if defaults < min_defaults else "green"
        row = {"observations": observations, "defaults": defaults, "years_covered": years, "low_default_flag": flag, "status": status}
        for column, value in zip([col for col in ["portfolio", "segment"] if col in df.columns], keys):
            row[column] = value
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_low_default_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize defaults by portfolio over multiple years."""
    return (
        df.groupby("portfolio", dropna=False)
        .agg(observations=("observation_id", "size"), defaults=("default_flag_12m", "sum"), years_covered=("observation_year", "nunique"))
        .reset_index()
    )


def calculate_multi_year_default_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate defaults by portfolio, segment, year and grade."""
    cols = [col for col in ["portfolio", "segment", "observation_year", "rating_grade"] if col in df.columns]
    return df.groupby(cols, dropna=False).agg(observations=("observation_id", "size"), defaults=("default_flag_12m", "sum")).reset_index()


def recommend_ldp_treatment(df: pd.DataFrame) -> str:
    """Generate standard low-default portfolio recommendations."""
    return "Regrouper les grades lorsque pertinent, analyser en pluriannuel, comparer a des benchmarks externes et documenter le jugement expert."
