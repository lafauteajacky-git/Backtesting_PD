import pandas as pd


DEFAULT_GRADE_ORDER = ["A", "B", "C", "D", "E", "F", "G"]


def calculate_grade_monotonicity(
    observations: pd.DataFrame,
    grade_order: list[str] | None = None,
    min_observations_per_grade: int = 30,
) -> pd.DataFrame:
    """Check whether observed default rate increases with grade risk order."""
    order = grade_order or DEFAULT_GRADE_ORDER
    valid = observations[
        observations["rating_grade"].notna()
        & observations["default_flag_12m"].isin([0, 1])
        & observations["pd_estimate"].notna()
    ].copy()
    rows = []
    previous_odr = None
    for idx, grade in enumerate(order, start=1):
        group = valid[valid["rating_grade"] == grade]
        observations_count = int(len(group))
        defaults = int(group["default_flag_12m"].sum()) if observations_count else 0
        odr = defaults / observations_count if observations_count else pd.NA
        pd_mean = float(group["pd_estimate"].mean()) if observations_count else pd.NA
        insufficient = observations_count < min_observations_per_grade
        violation = False
        if previous_odr is not None and not pd.isna(odr) and observations_count >= min_observations_per_grade:
            violation = odr < previous_odr
        if observations_count == 0:
            comment = "Grade absent du périmètre."
        elif insufficient:
            comment = "Volume insuffisant pour interpréter ce grade."
        elif violation:
            comment = "Violation locale : l'ODR diminue alors que le grade est plus risqué."
        else:
            comment = "Ordre de risque cohérent avec l'ODR observé."
        rows.append(
            {
                "rating_grade": grade,
                "grade_order": idx,
                "observations": observations_count,
                "defaults": defaults,
                "odr": odr,
                "pd_mean": pd_mean,
                "violation_monotonicity": bool(violation),
                "commentaire": comment,
            }
        )
        if observations_count >= min_observations_per_grade and not pd.isna(odr):
            previous_odr = odr
    return pd.DataFrame(rows)
