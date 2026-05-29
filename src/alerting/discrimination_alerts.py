import pandas as pd

from src.pd_backtesting.discrimination import calculate_discrimination_by_group


COMMENTS = {
    "green": "Le pouvoir discriminant du modèle est supérieur au seuil cible défini.",
    "orange": "Le pouvoir discriminant est inférieur au niveau cible. Une analyse complémentaire par segment, grade et période est recommandée.",
    "red": "Le pouvoir discriminant est faible au regard des seuils paramétrés. Une investigation approfondie est recommandée.",
    "grey": "L’indicateur de discrimination n’est pas interprétable en raison d’un volume insuffisant ou d’une absence de défauts/non-défauts.",
}


def assign_discrimination_status(metrics: dict | pd.Series, thresholds: dict) -> str:
    """Assign a discrimination traffic-light status from AUC, Gini and KS."""
    if not bool(metrics.get("is_interpretable", False)) or pd.isna(metrics.get("auc")):
        return "grey"

    config = thresholds["discrimination"]
    auc = metrics.get("auc")
    gini = metrics.get("gini")
    ks = metrics.get("ks")

    if auc < config["auc_red"] or gini < config["gini_red"] or ks < config["ks_red"]:
        return "red"
    if auc < config["auc_orange"] or gini < config["gini_orange"] or ks < config["ks_orange"]:
        return "orange"
    return "green"


def build_discrimination_alerts(
    observations: pd.DataFrame,
    thresholds: dict,
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Compute discrimination metrics and attach traffic-light statuses."""
    volume = thresholds["minimum_volume"]
    metrics = calculate_discrimination_by_group(
        observations,
        group_columns=group_columns,
        min_observations=volume["min_observations"],
        min_defaults=volume["min_defaults"],
        min_non_defaults=volume["min_non_defaults"],
    )
    if metrics.empty:
        return metrics

    metrics["status"] = metrics.apply(lambda row: assign_discrimination_status(row, thresholds), axis=1)
    metrics["comment"] = metrics["status"].map(COMMENTS)
    return metrics
