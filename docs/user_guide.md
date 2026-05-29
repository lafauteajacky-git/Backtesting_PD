# Guide utilisateur

## Objectif

Ce demonstrateur illustre le backtesting reglementaire de modeles PD a horizon 12 mois. Il couvre la calibration, la discrimination, la stabilite de population, les alertes traffic light et une synthese executive utilisable en rendez-vous client.

## Prerequis

- Python 3.10 ou plus recent.
- Installation des dependances du projet.

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Generer les donnees

```powershell
python -m src.data_generation.generate_sample_data --scenario retail_underestimation --output data/generated/demo_retail_underestimation.csv
```

Scenarios disponibles :

- `retail_well_calibrated` : portefeuille Retail bien calibre.
- `retail_underestimation` : sous-estimation du risque sur des grades Retail risqués.
- `corporate_low_default` : portefeuille Corporate a faibles defauts.
- `corporate_degraded_discrimination` : discrimination Corporate degradee.
- `population_shift` : derive de population avec PSI eleve.
- `data_quality_issues` : anomalies de qualite de donnees visibles.
- `pit_ttc_coherent` : philosophies PIT, TTC et Hybrid avec comportements temporels coherents.
- `pit_ttc_incoherent` : philosophie TTC declaree mais variations de PD et de distribution a investiguer.

## Lancer Streamlit

```powershell
python -m streamlit run app/streamlit_app.py
```

## Lire les resultats

L'onglet Accueil presente la synthese executive. Les onglets suivants detaillent la qualite des donnees, la calibration PD, la discrimination, la stabilite, les alertes, les exports et la methodologie.

La section Cartographie des tests presente les controles inclus dans le demonstrateur et leur objectif. La section Retail vs Corporate aide a distinguer les limites d'interpretation entre un portefeuille volumineux et un portefeuille low-default.

Les statuts traffic light se lisent ainsi :

- Vert : pas de signal majeur.
- Orange : signal a analyser.
- Rouge : signal prioritaire.
- Gris : test non interpretable.

## Exports

L'onglet Rapport & export permet de telecharger la synthese modele, les alertes et les findings au format CSV, ainsi qu'un rapport HTML et Excel lorsque les dependances sont installees.

## Limites MVP

Les donnees sont fictives. Le perimetre couvre uniquement la PD 12 mois. Les extensions LGD/EAD, l'authentification et une base de donnees ne sont pas incluses.
# Analyses enrichies V2

Les donnees generees contiennent desormais des champs reglementaires supplementaires : perimetre eligible, systeme de notation, philosophie PIT/TTC/Hybrid, composantes de PD, floors, MoC, historique rating et exclusions detaillees.

Dans Streamlit :

- **Donnees & qualite** : consultez le waterfall de population eligible, les exclusions et leur impact EAD.
- **Stabilite Rating / Score Distribution** : consultez RDS robuste, PSI, contributeurs, matrice de migration, nouveaux entrants et sorties.
- **Calibration PD** : consultez la lecture PIT/TTC/Hybrid sur PD 12 mois, PD raw/calibrated/regulatory, floors et MoC.
- **Stabilite Rating / Score Distribution** : consultez aussi le PSI/RDS par philosophie de modele lorsque plusieurs periodes sont disponibles.
- **Discrimination** : interpretez AUC/Gini/KS avec prudence si un segment est low-default ou si la population derive fortement.

Scenarios recommandes :

- `rds_population_shift` pour illustrer une derive de population.
- `rating_migration_deterioration` pour illustrer une matrice de migration degradee.
- `pd_floor_and_moc` pour illustrer floors et marges de conservatisme.
- `corporate_low_default_portfolio` pour illustrer les limites statistiques Corporate.
