from __future__ import annotations

import pandas as pd


FIELD_METADATA = {
    "observation_id": ("Identifiant observation", "Identifiant unique de la ligne d'observation utilisée pour le backtesting.", "Identifiant", "obligatoire, unique attendu"),
    "obligor_id": ("Identifiant contrepartie", "Identifiant de l'emprunteur ou groupe de contreparties.", "Identifiant", "clé de suivi client"),
    "facility_id": ("Identifiant facilité", "Identifiant de la facilité ou exposition observée.", "Identifiant", "peut être multiple par contrepartie"),
    "portfolio": ("Portefeuille", "Portefeuille réglementaire ou métier analysé.", "Catégoriel", "Retail, Corporate"),
    "segment": ("Segment", "Sous-population métier utilisée pour les analyses de robustesse.", "Catégoriel", "Mortgage, SME, Large Corporate..."),
    "product_type": ("Type de produit", "Nature du produit ou de l'exposition crédit.", "Catégoriel", "loan, mortgage, credit line..."),
    "model_id": ("Identifiant modèle", "Modèle PD appliqué à l'observation.", "Catégoriel", "dimension de monitoring modèle"),
    "model_version": ("Version modèle", "Version du modèle ou de la calibration utilisée.", "Catégoriel", "utile pour comparaison de versions"),
    "observation_date": ("Date d'observation", "Date à laquelle la PD, le rating et l'exposition sont observés.", "Date", "point de départ de la fenêtre 12 mois"),
    "origination_date": ("Date d'origination", "Date d'entrée en portefeuille ou d'octroi.", "Date", "support des analyses vintage"),
    "performance_window_months": ("Fenêtre de performance", "Horizon de performance utilisé pour observer le défaut.", "Numérique", "12 mois attendu dans le MVP"),
    "rating_grade": ("Grade de rating", "Classe de risque attribuée par le système de notation.", "Ordinal", "A à G, risque croissant"),
    "score": ("Score", "Score numérique associé au modèle ou au rating.", "Numérique", "souvent inversement lié au risque"),
    "pd_estimate": ("PD estimée", "Probabilité de défaut 12 mois utilisée comme estimation modèle principale.", "Numérique", "0 < PD <= 1"),
    "ead_at_observation": ("EAD à l'observation", "Exposition au défaut observée au début de la fenêtre.", "Numérique", "utilisée pour pondérations et impacts EAD"),
    "default_flag_12m": ("Défaut observé 12 mois", "Indicateur de défaut réalisé dans la fenêtre de performance.", "Binaire", "0/1"),
    "default_date": ("Date de défaut", "Date de survenance du défaut lorsque le flag 12 mois vaut 1.", "Date", "obligatoire si défaut observé"),
    "exclusion_flag": ("Flag exclusion", "Indique si l'observation est exclue ou marquée hors population backtestée.", "Binaire", "0/1"),
    "exclusion_reason": ("Motif exclusion", "Motif textuel ou catégorie d'exclusion.", "Catégoriel", "support de traçabilité"),
    "snapshot_id": ("Identifiant snapshot", "Identifiant du snapshot ou arrêté de données.", "Identifiant", "utile pour analyses N vs N-1"),
    "as_of_date": ("Date d'arrêté", "Date d'arrêté du snapshot.", "Date", "souvent fin d'année"),
    "observation_year": ("Année d'observation", "Année dérivée de la date d'observation.", "Numérique", "dimension temporelle"),
    "previous_observation_date": ("Date observation précédente", "Date de référence précédente pour migrations et RDS.", "Date", "souvent N-1"),
    "regulatory_exposure_class": ("Classe réglementaire", "Classe d'exposition réglementaire de l'observation.", "Catégoriel", "Retail, Corporate, Institution..."),
    "irb_approach": ("Approche IRB", "Approche réglementaire appliquée à l'exposition.", "Catégoriel", "AIRB, FIRB, Standardised, Slotting"),
    "application_scope_flag": ("Scope réglementaire", "Indique si l'exposition appartient au périmètre réglementaire analysé.", "Binaire", "1 éligible au scope"),
    "model_applicability_flag": ("Applicabilité modèle", "Indique si le modèle est applicable à l'exposition.", "Binaire", "1 modèle applicable"),
    "exposure_status_at_observation": ("Statut exposition", "Statut de l'exposition à la date d'observation.", "Catégoriel", "active, closed, defaulted..."),
    "performing_status_at_observation": ("Statut performing", "Statut performing/non-performing/forborne/defaulted.", "Catégoriel", "lecture réglementaire"),
    "default_flag_at_observation": ("Défaut à l'observation", "Indique si l'exposition est déjà en défaut au départ.", "Binaire", "doit être exclu du backtesting PD performing"),
    "data_complete_12m_flag": ("Données complètes 12 mois", "Indique si la fenêtre de performance est observable entièrement.", "Binaire", "condition d'éligibilité"),
    "maturity_before_12m_flag": ("Maturité avant horizon", "Indique si l'exposition arrive à maturité avant 12 mois.", "Binaire", "peut justifier exclusion"),
    "closure_date": ("Date clôture", "Date de clôture, vente ou maturité de l'exposition.", "Date", "utilisée pour horizon incomplet"),
    "rating_system_id": ("Système de notation", "Identifiant du système de notation crédit.", "Catégoriel", "distinct du modèle individuel"),
    "rating_method": ("Méthode de rating", "Méthode d'attribution du rating.", "Catégoriel", "scorecard, expert judgement, hybrid..."),
    "model_use_type": ("Type d'usage modèle", "Usage principal du modèle dans le cycle crédit.", "Catégoriel", "application, behavioural, monitoring..."),
    "model_philosophy": ("Philosophie modèle", "Philosophie de notation du modèle.", "Catégoriel", "PIT, TTC, Hybrid"),
    "pd_type": ("Type de PD", "Usage de la PD 12 mois dans le dispositif.", "Catégoriel", "regulatory_12m, origination_12m, behavioural_12m, monitoring_12m"),
    "pd_horizon_months": ("Horizon PD", "Horizon de prédiction de la PD en mois.", "Numérique", "valeur attendue 12 dans le MVP"),
    "calibration_pool_id": ("Pool de calibration", "Pool utilisé pour calibrer les PD.", "Catégoriel", "support de gouvernance calibration"),
    "rating_assignment_date": ("Date attribution rating", "Date d'attribution du rating courant.", "Date", "support stale rating"),
    "last_review_date": ("Date dernière revue", "Date de dernière revue du rating.", "Date", "support gouvernance rating"),
    "rating_age_months": ("Âge rating", "Ancienneté du rating en mois.", "Numérique", "utilisé pour stale rating"),
    "stale_rating_flag": ("Rating obsolète", "Indique si le rating dépasse le seuil d'ancienneté.", "Binaire", "signal qualité rating"),
    "grade_order": ("Ordre de grade", "Rang numérique du grade de rating.", "Ordinal", "risque croissant"),
    "master_scale_grade": ("Grade master scale", "Grade aligné sur une échelle maître.", "Catégoriel ordinal", "support comparabilité"),
    "score_bucket": ("Bucket score", "Classe discrète de score.", "Catégoriel ordinal", "support RDS"),
    "pd_bucket": ("Bucket PD", "Classe discrète de PD.", "Catégoriel ordinal", "support RDS"),
    "pd_raw": ("PD brute", "PD issue brute du modèle avant calibration finale.", "Numérique", "composante PD"),
    "pd_calibrated": ("PD calibrée", "PD après calibration statistique.", "Numérique", "avant floor/MoC selon convention"),
    "pd_regulatory": ("PD réglementaire", "PD utilisée pour lecture réglementaire après ajustements.", "Numérique", "après floors/MoC"),
    "pd_before_floor": ("PD avant floor", "PD avant application du floor réglementaire/interne.", "Numérique", "base d'impact floor"),
    "pd_after_floor": ("PD après floor", "PD après application du floor.", "Numérique", ">= PD before floor"),
    "pd_floor_applied_flag": ("Floor PD appliqué", "Indique si un floor a relevé la PD.", "Binaire", "impact conservatisme"),
    "pd_floor_value": ("Valeur floor PD", "Niveau minimal de PD applicable.", "Numérique", "seuil de floor"),
    "margin_of_conservatism": ("Marge de conservatisme", "Ajustement prudentiel ajouté à la PD.", "Numérique", "MoC"),
    "margin_of_conservatism_type": ("Type de MoC", "Nature de la marge de conservatisme.", "Catégoriel", "data_quality, methodology..."),
    "previous_rating_grade": ("Grade précédent", "Grade observé sur la période précédente.", "Ordinal", "support migration"),
    "previous_grade_order": ("Ordre grade précédent", "Rang numérique du grade précédent.", "Ordinal", "support notch change"),
    "previous_score": ("Score précédent", "Score observé sur la période précédente.", "Numérique", "support migration score"),
    "previous_pd_estimate": ("PD précédente", "PD estimée sur la période précédente.", "Numérique", "support RDS/migration PD"),
    "previous_pd_bucket": ("Bucket PD précédent", "Bucket de PD sur la période précédente.", "Catégoriel ordinal", "support migration"),
    "rating_migration": ("Migration rating", "Type de migration entre rating précédent et courant.", "Catégoriel", "stable, upgrade, downgrade, new, exited"),
    "notch_change": ("Changement de cran", "Différence de rang entre grade courant et précédent.", "Numérique", "positif = dégradation"),
    "rating_change_direction": ("Direction migration", "Sens métier de la migration.", "Catégoriel", "better, worse, stable..."),
    "new_customer_flag": ("Nouvel entrant", "Indique un client entrant dans la population.", "Binaire", "support population shift"),
    "exited_customer_flag": ("Client sorti", "Indique un client sortant de la population.", "Binaire", "support population shift"),
    "country_code": ("Pays", "Pays de rattachement client ou exposition.", "Catégoriel", "code pays"),
    "country_of_risk": ("Pays de risque", "Pays du risque économique principal.", "Catégoriel", "support concentration"),
    "sector_code": ("Secteur", "Code secteur économique.", "Catégoriel", "support mix sectoriel"),
    "borrower_size_class": ("Taille emprunteur", "Classe de taille de l'emprunteur.", "Catégoriel", "micro, small, medium, large"),
    "secured_flag": ("Exposition sécurisée", "Indique la présence d'une sûreté ou garantie.", "Binaire", "0/1"),
    "collateral_type": ("Type de collatéral", "Nature du collatéral principal.", "Catégoriel", "real estate, guarantee..."),
    "ltv_bucket": ("Bucket LTV", "Classe de loan-to-value.", "Catégoriel ordinal", "support risque garanti"),
    "origination_channel": ("Canal d'origination", "Canal d'entrée de l'exposition.", "Catégoriel", "branch, broker, digital..."),
    "watchlist_flag": ("Watchlist", "Indique une surveillance renforcée.", "Binaire", "signal risque"),
    "forbearance_flag_at_observation": ("Forbearance observation", "Indique une mesure de forbearance à l'observation.", "Binaire", "signal réglementaire"),
    "npe_flag_at_observation": ("NPE observation", "Indique une exposition non-performing à l'observation.", "Binaire", "signal réglementaire"),
    "days_past_due_at_observation": ("Jours d'impayé observation", "Nombre de jours de retard à l'observation.", "Numérique", "support défaut/NPE"),
    "default_reason": ("Raison défaut", "Motif principal du défaut observé.", "Catégoriel", "past_due_90, UTP..."),
    "past_due_90_flag_12m": ("Past due 90j 12m", "Indique un défaut 90 jours dans la fenêtre.", "Binaire", "composante défaut"),
    "utp_flag_12m": ("UTP 12m", "Indique un défaut unlikely-to-pay dans la fenêtre.", "Binaire", "composante défaut"),
    "max_days_past_due_12m": ("Max DPD 12m", "Maximum des jours d'impayé sur l'horizon 12 mois.", "Numérique", "support défaut"),
    "months_to_default": ("Mois avant défaut", "Nombre de mois entre observation et défaut.", "Numérique", "renseigné si défaut"),
    "default_exit_date": ("Date sortie défaut", "Date de sortie du statut défaut.", "Date", "support cure"),
    "cure_flag": ("Cure", "Indique un retour hors défaut après défaut.", "Binaire", "0/1"),
    "exclusion_category": ("Catégorie exclusion", "Catégorie structurée d'exclusion.", "Catégoriel", "data_quality, out_of_scope..."),
    "exclusion_materiality": ("Matérialité exclusion", "Niveau de matérialité de l'exclusion.", "Catégoriel", "low, medium, high"),
    "exclusion_applied_by_rule": ("Exclusion par règle", "Indique si une règle automatique a appliqué l'exclusion.", "Binaire", "0/1"),
    "exclusion_rule_id": ("Règle exclusion", "Identifiant de la règle d'exclusion.", "Catégoriel", "traçabilité"),
}


def _infer_category(field: str) -> str:
    if field in {"observation_id", "obligor_id", "facility_id", "snapshot_id"}:
        return "Identifiants et snapshots"
    if field.startswith("previous_") or field in {"rating_migration", "notch_change", "rating_change_direction", "new_customer_flag", "exited_customer_flag"}:
        return "Historique rating / score / PD"
    if field.startswith("pd_") or field in {"score", "score_bucket", "rating_grade", "grade_order", "master_scale_grade"}:
        return "Rating, score, PD et buckets"
    if "exclusion" in field:
        return "Exclusions"
    if "default" in field or field in {"cure_flag", "months_to_default", "max_days_past_due_12m", "utp_flag_12m", "past_due_90_flag_12m"}:
        return "Défaut et outcome"
    if field in {"country_code", "country_of_risk", "sector_code", "borrower_size_class", "secured_flag", "collateral_type", "ltv_bucket", "origination_channel", "watchlist_flag", "forbearance_flag_at_observation", "npe_flag_at_observation", "days_past_due_at_observation"}:
        return "Segmentation crédit"
    if field in {"rating_system_id", "rating_method", "model_use_type", "model_philosophy", "calibration_pool_id", "rating_assignment_date", "last_review_date", "rating_age_months", "stale_rating_flag"}:
        return "Rating system et philosophie"
    if field in {"regulatory_exposure_class", "irb_approach", "application_scope_flag", "model_applicability_flag", "exposure_status_at_observation", "performing_status_at_observation", "data_complete_12m_flag", "maturity_before_12m_flag", "closure_date"}:
        return "Périmètre réglementaire"
    return "Socle PD"


def build_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """Build a business data dictionary for all fields in the provided dataframe."""
    rows = []
    total = len(df)
    for field in df.columns:
        label, description, data_type, characteristics = FIELD_METADATA.get(
            field,
            (field, "Champ présent dans la base PD.", str(df[field].dtype), "à documenter selon le mapping source"),
        )
        series = df[field]
        examples = ", ".join(series.dropna().astype(str).drop_duplicates().head(5).tolist())
        rows.append(
            {
                "champ": field,
                "libelle": label,
                "categorie": _infer_category(field),
                "type_technique": str(series.dtype),
                "type_metier": data_type,
                "description_metier": description,
                "caracteristiques": characteristics,
                "taux_completude": series.notna().mean() if total else pd.NA,
                "valeurs_distinctes": int(series.nunique(dropna=True)),
                "exemples_valeurs": examples,
            }
        )
    return pd.DataFrame(rows).sort_values(["categorie", "champ"]).reset_index(drop=True)
