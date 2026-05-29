import pandas as pd

from src.pd_backtesting.monotonicity import calculate_grade_monotonicity


COMMENTS = {
    "green": "Aucune violation de monotonie des grades n'est détectée.",
    "orange": "Une violation locale de monotonie est détectée.",
    "red": "Plusieurs violations ou une violation matérielle de monotonie sont détectées.",
    "grey": "Le test de monotonie n'est pas interprétable en raison de volumes insuffisants.",
}


def assign_monotonicity_status(result: pd.DataFrame, thresholds: dict) -> str:
    """Assign traffic-light status for grade monotonicity."""
    config = thresholds["monotonicity"]
    interpretable = result[result["observations"] >= config["min_observations_per_grade"]]
    if interpretable.empty or result["defaults"].sum() < config["min_defaults_total"]:
        return "grey"
    violations = interpretable[interpretable["violation_monotonicity"]]
    if violations.empty:
        return "green"
    previous_odr = None
    material = False
    for _, row in interpretable.sort_values("grade_order").iterrows():
        if previous_odr is not None and row["odr"] < previous_odr - config["material_odr_decrease"]:
            material = True
        previous_odr = row["odr"]
    if len(violations) > 1 or material:
        return "red"
    return "orange"


def build_monotonicity_alert(
    observations: pd.DataFrame,
    thresholds: dict,
    perimeter: str = "Global",
) -> tuple[pd.DataFrame, dict]:
    """Return monotonicity rows and one summary alert."""
    result = calculate_grade_monotonicity(
        observations,
        min_observations_per_grade=thresholds["monotonicity"]["min_observations_per_grade"],
    )
    status = assign_monotonicity_status(result, thresholds)
    result["interpretabilite"] = result["observations"].map(
        lambda value: "Volume insuffisant"
        if value < thresholds["monotonicity"]["min_observations_per_grade"]
        else "Interprétable"
    )
    alert = {
        "test_family": "Monotonie",
        "perimeter": perimeter,
        "aggregation_level": "rating_grade",
        "status": status,
        "comment": COMMENTS[status],
        "interpretabilite": "Non interprétable" if status == "grey" else "Interprétable",
    }
    return result, alert
