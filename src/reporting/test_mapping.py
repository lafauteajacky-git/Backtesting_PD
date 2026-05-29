import pandas as pd


def build_test_mapping() -> pd.DataFrame:
    """Return the methodological test map displayed in the demo."""
    rows = [
        ("Data quality", "Observation ID manquant", "Vérifier l'identifiant unique", "Ligne", "Nombre d'anomalies", "inclus"),
        ("Data quality", "PD hors bornes", "Contrôler 0 < PD <= 1", "Ligne", "Taux d'anomalies", "inclus"),
        ("Quantification / calibration", "ODR vs PD moyenne", "Comparer défaut observé et risque estimé", "Portfolio / grade / période", "Gap, ratio", "inclus"),
        ("Quantification / calibration", "Test binomial", "Tester la significativité de l'écart", "Tout niveau agrégé", "p-value, IC", "inclus"),
        ("Discrimination", "AUC / Gini", "Mesurer le pouvoir de classement", "Global / portfolio", "AUC, Gini", "inclus"),
        ("Discrimination", "KS", "Mesurer la séparation défauts / non-défauts", "Global / portfolio", "KS", "inclus"),
        ("Stabilité Rating / Score Distribution", "PSI rating", "Comparer les distributions de grades", "Période / portfolio / segment", "PSI", "inclus"),
        ("Stabilité Rating / Score Distribution", "PSI score / PD buckets", "Comparer scores ou buckets de PD", "Période / portfolio / segment", "PSI", "optionnel"),
        ("Faibles volumes et robustesse", "Seuils minimums", "Eviter la sur-interprétation", "Tout niveau", "Interprétabilité", "inclus"),
        ("Faibles volumes et robustesse", "Monotonie des grades", "Vérifier l'ordre de risque des grades", "Rating grade", "Violations", "inclus"),
        ("Reporting et alerting", "Traffic light", "Prioriser les constats", "Test / périmètre", "Vert / orange / rouge / gris", "inclus"),
        ("Reporting et alerting", "Exports Excel/HTML", "Partager les résultats", "Rapport", "Fichiers exportés", "inclus"),
    ]
    return pd.DataFrame(rows, columns=["theme", "nom_du_test", "objectif", "niveau_analyse", "indicateur_produit", "statut"])
