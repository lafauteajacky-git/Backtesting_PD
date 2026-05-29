# Version V0 - MVP Backtesting PD

Date de sauvegarde : 2026-05-26

## Contenu fonctionnel

- Socle Python/Streamlit.
- Generation de donnees fictives Retail et Corporate.
- Scenarios de demonstration reproductibles.
- Controles data quality.
- Moteur de calibration PD 12 mois.
- Tests binomiaux et alertes traffic light.
- Indicateurs de discrimination : AUC, Gini, KS, ROC, CAP.
- Indicateurs de stabilite : PSI.
- Synthese executive Streamlit.
- Narrative demo.
- Exports CSV/HTML simples.
- Documentation utilisateur, methodologique et script demo.

## Validation

Derniere validation connue :

```text
34 tests passed
Streamlit HTTP 200
CLI scenario OK
```

## Commandes de reference

```powershell
python -m src.data_generation.generate_sample_data --scenario retail_underestimation --output data/generated/demo_retail_underestimation.csv
python -m pytest
python -m streamlit run app/streamlit_app.py
```
