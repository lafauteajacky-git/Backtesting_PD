from __future__ import annotations

import numpy as np
import pandas as pd

from src.pd_backtesting.rds import calculate_rds_psi


PHILOSOPHY_COMMENTS = {
    "PIT": (
        "PIT : une variation de PD dans le temps peut être cohérente si elle reflète "
        "l'évolution conjoncturelle ou la dégradation du risque."
    ),
    "TTC": (
        "TTC : une forte variation de PD ou de distribution de grades doit être investiguée, "
        "car une plus grande stabilité relative est généralement attendue."
    ),
    "Hybrid": (
        "Hybrid : l'interprétation doit tenir compte à la fois de la composante cyclique "
        "et de la stabilité attendue du rating."
    ),
    "Unknown": (
        "Unknown : la philosophie du modèle n'est pas renseignée, ce qui limite "
        "l'interprétation des résultats."
    ),
}


def _with_philosophy(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a normalized model_philosophy column."""
    output = df.copy()
    if "model_philosophy" not in output.columns:
        output["model_philosophy"] = "Unknown"
    output["model_philosophy"] = output["model_philosophy"].fillna("Unknown").replace("", "Unknown")
    return output


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.divide(denominator.replace(0, np.nan))


def summarize_model_philosophy(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize observations, default rates and PD metrics by model philosophy."""
    if df.empty:
        return pd.DataFrame()
    data = _with_philosophy(df)
    pd_col = "pd_estimate" if "pd_estimate" in data.columns else "pd_regulatory"
    if pd_col not in data.columns or "default_flag_12m" not in data.columns:
        return pd.DataFrame()

    grouped = (
        data.assign(
            _default=pd.to_numeric(data["default_flag_12m"], errors="coerce"),
            _pd=pd.to_numeric(data[pd_col], errors="coerce"),
            _pd_raw=pd.to_numeric(data.get("pd_raw", np.nan), errors="coerce"),
            _pd_calibrated=pd.to_numeric(data.get("pd_calibrated", np.nan), errors="coerce"),
            _pd_regulatory=pd.to_numeric(data.get("pd_regulatory", np.nan), errors="coerce"),
        )
        .groupby("model_philosophy", dropna=False)
        .agg(
            observations=("model_philosophy", "size"),
            defaults=("_default", "sum"),
            pd_mean=("_pd", "mean"),
            pd_raw_mean=("_pd_raw", "mean"),
            pd_calibrated_mean=("_pd_calibrated", "mean"),
            pd_regulatory_mean=("_pd_regulatory", "mean"),
        )
        .reset_index()
    )
    grouped["odr"] = _safe_ratio(grouped["defaults"], grouped["observations"])
    grouped["expected_defaults"] = grouped["pd_mean"] * grouped["observations"]
    grouped["calibration_gap"] = grouped["odr"] - grouped["pd_mean"]
    grouped["calibration_ratio"] = _safe_ratio(grouped["odr"], grouped["pd_mean"])
    return grouped


def compare_pd_by_philosophy(df: pd.DataFrame) -> pd.DataFrame:
    """Return the core 12-month PD backtesting metrics by philosophy."""
    return summarize_model_philosophy(df)


def compare_pit_ttc_behaviour(df: pd.DataFrame) -> pd.DataFrame:
    """Compare 12-month PD sensitivity through time by philosophy."""
    if df.empty or "observation_year" not in df.columns:
        return pd.DataFrame()
    data = _with_philosophy(df)
    if "pd_estimate" not in data.columns:
        return pd.DataFrame()
    yearly = (
        data.assign(_pd=pd.to_numeric(data["pd_estimate"], errors="coerce"))
        .groupby(["model_philosophy", "observation_year"], dropna=False)
        .agg(
            observations=("model_philosophy", "size"),
            pd_mean=("_pd", "mean"),
            grade_order_mean=("grade_order", "mean") if "grade_order" in data.columns else ("model_philosophy", "size"),
        )
        .reset_index()
        .sort_values(["model_philosophy", "observation_year"])
    )
    yearly["pd_change_vs_previous"] = yearly.groupby("model_philosophy")["pd_mean"].diff()
    return yearly


def analyze_pd_volatility_by_philosophy(df: pd.DataFrame) -> pd.DataFrame:
    """Measure volatility of mean 12-month PD by philosophy across observation years."""
    yearly = compare_pit_ttc_behaviour(df)
    if yearly.empty:
        return pd.DataFrame()
    summary = (
        yearly.groupby("model_philosophy", dropna=False)
        .agg(
            periods=("observation_year", "nunique"),
            pd_mean_average=("pd_mean", "mean"),
            pd_mean_volatility=("pd_mean", "std"),
            pd_mean_min=("pd_mean", "min"),
            pd_mean_max=("pd_mean", "max"),
        )
        .reset_index()
    )
    summary["pd_mean_volatility"] = summary["pd_mean_volatility"].fillna(0.0)
    summary["pd_mean_range"] = summary["pd_mean_max"] - summary["pd_mean_min"]
    summary["volatility_ratio"] = _safe_ratio(summary["pd_mean_volatility"], summary["pd_mean_average"])
    return summary


def compare_rds_by_philosophy(
    df: pd.DataFrame,
    reference_period=None,
    current_period=None,
    variable: str = "rating_grade",
) -> pd.DataFrame:
    """Calculate PSI/RDS by model philosophy for a selected distribution variable."""
    if df.empty or "observation_year" not in df.columns:
        return pd.DataFrame()
    data = _with_philosophy(df)
    if variable not in data.columns:
        return pd.DataFrame()
    periods = sorted(data["observation_year"].dropna().unique().tolist())
    if len(periods) < 2:
        return pd.DataFrame()
    reference_period = periods[0] if reference_period is None else reference_period
    current_period = periods[-1] if current_period is None else current_period
    rows = []
    for philosophy, group in data.groupby("model_philosophy", dropna=False):
        reference = group[group["observation_year"] == reference_period]
        current = group[group["observation_year"] == current_period]
        result = calculate_rds_psi(reference, current, variable)
        rows.append(
            {
                "model_philosophy": philosophy,
                "variable": variable,
                "reference_period": reference_period,
                "current_period": current_period,
                "reference_observations": len(reference),
                "current_observations": len(current),
                "psi": result["psi"],
                "status": result["status"],
            }
        )
    return pd.DataFrame(rows)


def compare_migration_by_philosophy(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize migration directions by philosophy when migration fields exist."""
    if df.empty or "rating_migration" not in df.columns:
        return pd.DataFrame()
    data = _with_philosophy(df)
    migration = (
        data.assign(_migration=data["rating_migration"].fillna("Missing").astype(str))
        .groupby(["model_philosophy", "_migration"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    migration["share"] = migration["count"] / migration.groupby("model_philosophy")["count"].transform("sum")
    return migration.rename(columns={"_migration": "rating_migration"})


def generate_philosophy_commentary(
    df: pd.DataFrame,
    calibration_results: pd.DataFrame | None = None,
    stability_results: pd.DataFrame | None = None,
) -> str:
    """Generate prudent PIT/TTC/Hybrid comments without ranking philosophies."""
    data = _with_philosophy(df)
    philosophies = sorted(data["model_philosophy"].dropna().unique().tolist())
    if not philosophies:
        philosophies = ["Unknown"]
    comments = [PHILOSOPHY_COMMENTS.get(philosophy, PHILOSOPHY_COMMENTS["Unknown"]) for philosophy in philosophies]
    comments.append(
        "Cette lecture porte uniquement sur des PD 12 mois ; PIT/TTC/Hybrid est une clé d'interprétation, "
        "pas un test autonome de conformité ni une comparaison de qualité entre philosophies."
    )
    return " ".join(comments)
