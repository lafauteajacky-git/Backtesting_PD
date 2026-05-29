from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.pd_backtesting.metrics import calculate_pd_metrics


AGGREGATION_LEVELS = {
    "global": [],
    "portfolio": ["portfolio"],
    "segment": ["segment"],
    "product_type": ["product_type"],
    "rating_grade": ["rating_grade"],
    "observation_year": ["observation_year"],
    "portfolio_segment": ["portfolio", "segment"],
    "portfolio_rating_grade": ["portfolio", "rating_grade"],
}


def add_observation_year(observations: pd.DataFrame) -> pd.DataFrame:
    """Add an `observation_year` column derived from `observation_date`."""
    enriched = observations.copy()
    enriched["observation_year"] = pd.to_datetime(
        enriched["observation_date"], errors="coerce"
    ).dt.year.astype("Int64")
    return enriched


def aggregate_pd_metrics(
    observations: pd.DataFrame,
    group_columns: Sequence[str] | None = None,
    level_name: str = "global",
) -> pd.DataFrame:
    """Aggregate core PD metrics globally or by the provided columns."""
    enriched = add_observation_year(observations)
    group_columns = list(group_columns or [])

    if not group_columns:
        row = calculate_pd_metrics(enriched)
        row["aggregation_level"] = level_name
        row["perimeter"] = "Global"
        return pd.DataFrame([row])

    rows = []
    for group_values, group in enriched.groupby(group_columns, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = calculate_pd_metrics(group)
        for column, value in zip(group_columns, group_values):
            row[column] = "Missing" if pd.isna(value) else value
        row["aggregation_level"] = level_name
        row["perimeter"] = " x ".join(str(row[column]) for column in group_columns)
        rows.append(row)

    return pd.DataFrame(rows)


def build_standard_aggregations(observations: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build the standard Lot 2 aggregations."""
    return {
        level: aggregate_pd_metrics(observations, columns, level)
        for level, columns in AGGREGATION_LEVELS.items()
    }


def concat_standard_aggregations(observations: pd.DataFrame) -> pd.DataFrame:
    """Return all standard aggregations as a single DataFrame."""
    return pd.concat(build_standard_aggregations(observations).values(), ignore_index=True)
