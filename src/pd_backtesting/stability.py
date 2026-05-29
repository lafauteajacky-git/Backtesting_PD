from __future__ import annotations

import numpy as np
import pandas as pd


def add_observation_year(observations: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with `observation_year` derived from `observation_date`."""
    frame = observations.copy()
    frame["observation_year"] = pd.to_datetime(
        frame["observation_date"], errors="coerce"
    ).dt.year.astype("Int64")
    return frame


def psi_from_distributions(reference: pd.Series, current: pd.Series, epsilon: float = 1e-6) -> float:
    """Compute Population Stability Index from aligned distribution shares."""
    aligned = pd.concat([reference, current], axis=1).fillna(0.0)
    ref = aligned.iloc[:, 0].clip(lower=epsilon)
    cur = aligned.iloc[:, 1].clip(lower=epsilon)
    return float(((cur - ref) * np.log(cur / ref)).sum())


def calculate_psi(
    reference: pd.Series,
    current: pd.Series,
    categories: list | None = None,
    min_observations: int = 30,
) -> dict:
    """Calculate PSI for two categorical/bucketed pandas Series."""
    ref = reference.dropna()
    cur = current.dropna()
    if len(ref) < min_observations or len(cur) < min_observations:
        return {"psi": np.nan, "is_calculable": False, "comment": "Volume insuffisant."}

    if categories is None:
        categories = sorted(set(ref.unique()).union(set(cur.unique())))
    ref_dist = ref.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    cur_dist = cur.value_counts(normalize=True).reindex(categories, fill_value=0.0)

    reference_distribution = pd.DataFrame(
        {"bucket": ref_dist.index.astype(str), "reference_share": ref_dist.to_numpy()}
    )
    current_distribution = pd.DataFrame(
        {"bucket": cur_dist.index.astype(str), "current_share": cur_dist.to_numpy()}
    )

    return {
        "psi": psi_from_distributions(ref_dist, cur_dist),
        "is_calculable": True,
        "comment": "",
        "reference_distribution": reference_distribution,
        "current_distribution": current_distribution,
    }


def make_numeric_buckets(series: pd.Series, bins: int = 10) -> pd.Series:
    """Bucket a numeric variable into quantile-like intervals."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().nunique() <= 1:
        return pd.Series(pd.NA, index=series.index, dtype="object")
    return pd.qcut(numeric, q=min(bins, numeric.dropna().nunique()), duplicates="drop").astype("string")


def calculate_stability_table(
    observations: pd.DataFrame,
    reference_period,
    current_period,
    variable: str = "rating_grade",
    group_columns: list[str] | None = None,
    min_observations: int = 30,
) -> pd.DataFrame:
    """Compute PSI between two observation years globally or by group."""
    frame = add_observation_year(observations)
    group_columns = group_columns or []
    frame = frame[frame["observation_year"].isin([reference_period, current_period])].copy()

    if variable == "pd_bucket":
        frame["_stability_variable"] = make_numeric_buckets(frame["pd_estimate"])
    elif variable == "score_bucket":
        frame["_stability_variable"] = make_numeric_buckets(frame["score"])
    else:
        frame["_stability_variable"] = frame[variable]

    rows = []
    groups = [((), frame)] if not group_columns else frame.groupby(group_columns, dropna=False)
    for values, group in groups:
        if not isinstance(values, tuple):
            values = (values,)
        ref = group.loc[group["observation_year"] == reference_period, "_stability_variable"]
        cur = group.loc[group["observation_year"] == current_period, "_stability_variable"]
        result = calculate_psi(ref, cur, min_observations=min_observations)
        row = {
            "perimeter": "Global",
            "aggregation_level": "global" if not group_columns else "_".join(group_columns),
            "reference_period": reference_period,
            "current_period": current_period,
            "variable": variable,
            "reference_observations": int(ref.dropna().shape[0]),
            "current_observations": int(cur.dropna().shape[0]),
            "psi": result["psi"],
            "is_calculable": result["is_calculable"],
        }
        for column, value in zip(group_columns, values):
            row[column] = "Missing" if pd.isna(value) else value
        if group_columns:
            row["perimeter"] = " x ".join(str(row[column]) for column in group_columns)
        rows.append(row)
    return pd.DataFrame(rows)
