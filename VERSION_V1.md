# Version V1 - MVP backtesting reglementaire PD

Date de sauvegarde : 2026-05-28

Cette version fige le MVP avant modifications structurantes.

## Contenu fonctionnel

- Generateur de donnees fictives Retail / Corporate avec observation multi-annees et `origination_date`.
- Scenarios de demonstration reproductibles.
- Chargement des donnees et controles data quality.
- Moteur de calibration PD 12 mois : ODR, PD moyenne, PD ponderee EAD, defauts attendus, calibration gap, calibration ratio, tests binomiaux, intervalle de confiance et Hosmer-Lemeshow.
- Indicateurs de discrimination : AUC, Gini, KS, ROC, CAP, bad rate par decile.
- Indicateurs de stabilite Rating / Score Distribution : PSI, distributions, migrations simplifiees, mix produit / segment.
- Test de monotonie des grades.
- Alerting traffic light et findings de validation.
- Application Streamlit aux couleurs Auria Advisory avec sections Accueil, Donnees & qualite, Stabilite RDS, Calibration PD, Discrimination, Retail vs Corporate, Alertes, Rapport & export, Methodologie.
- Exports CSV, Excel et HTML.
- Documentation utilisateur, methodologique et script de demonstration.

## Etat de validation

- Tests unitaires pytest : 42 tests passes.
- Compilation Python de `app/streamlit_app.py` verifiee.
- Dataset courant regenere avec `origination_date`.

## Commandes de reprise

```powershell
python -m src.data_generation.generate_pd_observations --output data/generated/pd_observations.csv --retail 30000 --corporate 5000 --seed 42 --start-year 2019 --years 5
python -m pytest
python -m streamlit run app/streamlit_app.py
```
