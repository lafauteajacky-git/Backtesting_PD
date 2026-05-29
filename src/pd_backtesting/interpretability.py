import pandas as pd


def calibration_interpretability(row: pd.Series, thresholds: dict) -> str:
    """Return a business-readable interpretability label for calibration results."""
    volume = thresholds["minimum_volume"]
    if row.get("observations", 0) < volume["min_observations"]:
        return "Volume insuffisant"
    if row.get("observed_defaults", 0) < volume["min_defaults"]:
        return "Défauts insuffisants"
    if pd.isna(row.get("pd_mean")) or pd.isna(row.get("odr")):
        return "Données manquantes"
    if row.get("observations", 0) < volume["min_observations"] * 3:
        return "À interpréter avec prudence"
    return "Interprétable"


def generic_interpretability(row: pd.Series, thresholds: dict) -> str:
    """Return an interpretability label for test outputs with standard flags."""
    if "is_interpretable" in row and not bool(row["is_interpretable"]):
        if row.get("defaults", row.get("observed_defaults", 0)) < thresholds["minimum_volume"]["min_defaults"]:
            return "Défauts insuffisants"
        return "Non interprétable"
    if "is_calculable" in row and not bool(row["is_calculable"]):
        return "Non interprétable"
    if row.get("observations", thresholds["minimum_volume"]["min_observations"]) < thresholds["minimum_volume"]["min_observations"]:
        return "Volume insuffisant"
    return "Interprétable"


def add_interpretability(frame: pd.DataFrame, thresholds: dict, family: str) -> pd.DataFrame:
    """Attach an `interpretabilite` column to a result DataFrame."""
    output = frame.copy()
    if output.empty:
        output["interpretabilite"] = []
        return output
    if family == "calibration":
        output["interpretabilite"] = output.apply(lambda row: calibration_interpretability(row, thresholds), axis=1)
    else:
        output["interpretabilite"] = output.apply(lambda row: generic_interpretability(row, thresholds), axis=1)
    return output
