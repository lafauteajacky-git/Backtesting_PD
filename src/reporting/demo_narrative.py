import pandas as pd


def generate_demo_narrative(
    scenario_description: str,
    expected_observation: str,
    calibration_alerts: pd.DataFrame,
    discrimination_alerts: pd.DataFrame,
    stability_alerts: pd.DataFrame,
) -> str:
    """Generate a short business narrative for a client demo."""
    red_alerts = int(
        sum((frame.get("status") == "red").sum() for frame in [calibration_alerts, discrimination_alerts, stability_alerts] if not frame.empty)
    )
    orange_alerts = int(
        sum((frame.get("status") == "orange").sum() for frame in [calibration_alerts, discrimination_alerts, stability_alerts] if not frame.empty)
    )
    grey_alerts = int(
        sum((frame.get("status") == "grey").sum() for frame in [calibration_alerts, discrimination_alerts, stability_alerts] if not frame.empty)
    )

    if red_alerts:
        validation_read = "La lecture validation met en evidence des signaux critiques a investiguer prioritairement."
    elif orange_alerts:
        validation_read = "La lecture validation identifie des signaux d'alerte moderes qui justifient une analyse complementaire."
    elif grey_alerts:
        validation_read = "Certains tests sont non interpretables, principalement du fait de faibles volumes ou de donnees manquantes."
    else:
        validation_read = "Les indicateurs disponibles ne mettent pas en evidence d'alerte majeure."

    return (
        f"Scenario presente : {scenario_description}\n\n"
        f"Resultats attendus : {expected_observation}\n\n"
        f"Resultats observes : {red_alerts} alerte(s) rouge(s), {orange_alerts} alerte(s) orange(s) "
        f"et {grey_alerts} alerte(s) grise(s) sur les controles statistiques affiches.\n\n"
        f"Lecture validation : {validation_read}\n\n"
        "Investigations recommandees : analyser les resultats par portefeuille, segment, rating grade et millesime ; "
        "documenter les faibles volumes ; rapprocher les signaux de calibration des changements de politique d'octroi, "
        "de population et de scoring."
    )
