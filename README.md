# Backtesting PD 12 mois

Démonstrateur Python / Streamlit de backtesting réglementaire PD 12 mois.

L'application couvre les principaux volets d'une validation de modèle PD :
calibration, discrimination, stabilité Rating / Score Distribution, data quality,
population éligible, alerting, exports et lecture PIT / TTC / Hybrid.

## Lancement local

```powershell
streamlit run app.py
```

ou :

```powershell
python -m streamlit run app.py
```

## Installation locale

```powershell
python -m pip install -r requirements.txt
```

## Streamlit Community Cloud

Cette application est prévue pour un déploiement sur Streamlit Community Cloud.

Paramètres de déploiement recommandés :

- Repository : `lafauteajacky-git/Backtesting_PD`
- Branch : `main`
- Main file path : `app.py`

## Données

Les données utilisées par défaut sont fictives et générées par l'application via
les scénarios de démonstration. Les fichiers CSV générés localement dans
`data/generated/` ne sont pas nécessaires au déploiement et sont exclus de Git.

## Tests

```powershell
python -m pytest
```
