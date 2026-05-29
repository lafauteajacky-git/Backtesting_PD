import pandas as pd

from src.pd_backtesting.stability import calculate_stability_table


COMMENTS = {
    "green": "La distribution de population ne présente pas de dérive significative selon le seuil PSI paramétré.",
    "orange": "Une dérive modérée de population est détectée. Les résultats de backtesting doivent être interprétés avec prudence.",
    "red": "Une dérive significative de population est détectée. Une analyse des changements de population, d’octroi ou de scoring est recommandée.",
    "grey": "Le PSI n’est pas calculable en raison d’une donnée manquante, d’une période de référence absente ou d’un volume insuffisant.",
}


def assign_stability_status(psi: float, is_calculable: bool, thresholds: dict) -> str:
    """Assign a PSI traffic-light status."""
    if not is_calculable or pd.isna(psi):
        return "grey"

    config = thresholds["stability"]
    if psi >= config["psi_red"]:
        return "red"
    if psi >= config["psi_orange"]:
        return "orange"
    return "green"


def build_stability_alerts(
    observations: pd.DataFrame,
    thresholds: dict,
    reference_period,
    current_period,
    variable: str = "rating_grade",
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Compute PSI metrics and attach traffic-light statuses."""
    table = calculate_stability_table(
        observations,
        reference_period=reference_period,
        current_period=current_period,
        variable=variable,
        group_columns=group_columns,
        min_observations=thresholds["minimum_volume"]["min_observations"],
    )
    table["status"] = table.apply(
        lambda row: assign_stability_status(row["psi"], row["is_calculable"], thresholds),
        axis=1,
    )
    table["comment"] = table["status"].map(COMMENTS)
    return table
