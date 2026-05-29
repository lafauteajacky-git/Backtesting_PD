import pandas as pd


SEVERITY = {"red": "Haute", "orange": "Moyenne", "grey": "À qualifier", "green": "Information"}


def generate_validation_findings(*alert_frames: pd.DataFrame) -> pd.DataFrame:
    """Transform main alerts into validation findings."""
    rows = []
    finding_id = 1
    for frame in alert_frames:
        if frame.empty or "status" not in frame.columns:
            continue
        for _, row in frame[frame["status"].isin(["red", "orange", "grey"])].iterrows():
            theme = row.get("test_family", row.get("aggregation_level", "Backtesting"))
            perimeter = row.get("perimeter", "Global")
            status = row.get("status", "grey")
            rows.append(
                {
                    "finding_id": f"F-{finding_id:03d}",
                    "theme": theme,
                    "perimetre": perimeter,
                    "constat": row.get("comment", "Signal de validation à analyser."),
                    "niveau_de_severite": SEVERITY.get(status, "Information"),
                    "justification_statistique": _justification(row),
                    "recommandation": _recommendation(status, theme),
                }
            )
            finding_id += 1
    return pd.DataFrame(rows)


def _justification(row: pd.Series) -> str:
    parts = []
    for column in ["p_value", "auc", "gini", "ks", "psi", "odr", "pd_mean"]:
        if column in row and pd.notna(row[column]):
            value = row[column]
            parts.append(f"{column}={value:.4f}" if isinstance(value, (int, float)) else f"{column}={value}")
    return "; ".join(parts) if parts else "Test non interprétable ou information qualitative."


def _recommendation(status: str, theme: str) -> str:
    if status == "red":
        return "Ouvrir une investigation prioritaire et documenter l'impact validation."
    if status == "orange":
        return "Analyser par segment, grade et millésime avant conclusion."
    if status == "grey":
        return "Documenter les limites de volume et consolider le périmètre si possible."
    return "Surveiller dans le prochain cycle de monitoring."
