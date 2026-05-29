from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


DEFAULT_GROUP_COLUMNS = ["portfolio", "segment", "model_id", "observation_year"]


def add_observation_year(observations: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with `observation_year` derived from `observation_date`."""
    frame = observations.copy()
    frame["observation_year"] = pd.to_datetime(
        frame["observation_date"], errors="coerce"
    ).dt.year.astype("Int64")
    return frame


def _risk_score(frame: pd.DataFrame, score_column: str | None = None) -> pd.Series:
    if score_column and score_column in frame.columns:
        return pd.to_numeric(frame[score_column], errors="coerce")
    if "pd_estimate" in frame.columns:
        return pd.to_numeric(frame["pd_estimate"], errors="coerce")
    if "score" in frame.columns:
        return -pd.to_numeric(frame["score"], errors="coerce")
    return pd.Series(np.nan, index=frame.index)


def _non_interpretable(reason: str) -> dict:
    return {
        "auc": np.nan,
        "gini": np.nan,
        "ks": np.nan,
        "status": "grey",
        "comment": reason,
        "is_interpretable": False,
        "roc_curve": pd.DataFrame(columns=["fpr", "tpr", "threshold"]),
        "cap_curve": pd.DataFrame(columns=["population_share", "default_share"]),
    }


def calculate_discrimination_metrics(
    observations: pd.DataFrame,
    min_observations: int = 30,
    min_defaults: int = 5,
    min_non_defaults: int = 30,
    score_column: str | None = None,
) -> dict:
    """Compute AUC, Gini, KS, ROC points and CAP points for a DataFrame."""
    frame = observations.copy()
    frame["_risk_score"] = _risk_score(frame, score_column)
    frame = frame[
        frame["default_flag_12m"].isin([0, 1]) & frame["_risk_score"].notna()
    ].copy()

    observations_count = int(len(frame))
    defaults = int(frame["default_flag_12m"].sum()) if observations_count else 0
    non_defaults = observations_count - defaults

    if observations_count < min_observations:
        result = _non_interpretable("Volume insuffisant pour calculer la discrimination.")
    elif defaults < min_defaults:
        result = _non_interpretable("Absence ou volume insuffisant de defauts.")
    elif non_defaults < min_non_defaults:
        result = _non_interpretable("Absence ou volume insuffisant de non-defauts.")
    elif frame["_risk_score"].nunique(dropna=True) <= 1:
        result = _non_interpretable("Un seul niveau de score ou de PD est disponible.")
    else:
        y_true = frame["default_flag_12m"].astype(int)
        y_score = frame["_risk_score"].astype(float)
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        auc = float(roc_auc_score(y_true, y_score))
        ks = float(np.max(tpr - fpr))

        sorted_frame = frame.sort_values("_risk_score", ascending=False).reset_index(drop=True)
        cap_curve = pd.DataFrame(
            {
                "population_share": np.arange(1, len(sorted_frame) + 1) / len(sorted_frame),
                "default_share": sorted_frame["default_flag_12m"].cumsum() / defaults,
            }
        )
        result = {
            "auc": auc,
            "gini": float(2 * auc - 1),
            "ks": ks,
            "status": None,
            "comment": "",
            "is_interpretable": True,
            "roc_curve": pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}),
            "cap_curve": cap_curve,
        }

    result.update(
        {
            "observations": observations_count,
            "defaults": defaults,
            "non_defaults": non_defaults,
        }
    )
    return result


def calculate_discrimination_by_group(
    observations: pd.DataFrame,
    group_columns: Sequence[str] | None,
    min_observations: int = 30,
    min_defaults: int = 5,
    min_non_defaults: int = 30,
) -> pd.DataFrame:
    """Compute discrimination metrics globally or by requested dimensions."""
    frame = add_observation_year(observations)
    group_columns = list(group_columns or [])

    if not group_columns:
        metrics = calculate_discrimination_metrics(
            frame, min_observations, min_defaults, min_non_defaults
        )
        metrics.update({"aggregation_level": "global", "perimeter": "Global"})
        return pd.DataFrame([{k: v for k, v in metrics.items() if not k.endswith("_curve")}])

    rows = []
    for values, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        metrics = calculate_discrimination_metrics(
            group, min_observations, min_defaults, min_non_defaults
        )
        row = {k: v for k, v in metrics.items() if not k.endswith("_curve")}
        for column, value in zip(group_columns, values):
            row[column] = "Missing" if pd.isna(value) else value
        row["aggregation_level"] = "_".join(group_columns)
        row["perimeter"] = " x ".join(str(row[column]) for column in group_columns)
        rows.append(row)
    return pd.DataFrame(rows)
