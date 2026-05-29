# Script de demonstration commerciale - 20 minutes

## 0-3 min - Introduction

Presenter l'objectif : disposer d'un cockpit simple pour backtester des modeles PD reglementaires, identifier les signaux de calibration, de discrimination et de stabilite, puis produire une lecture validation exploitable.

## 3-5 min - Scenario recommande

Selectionner `retail_underestimation`.

Message : ce scenario illustre une sous-estimation du risque sur certains grades Retail. Le demonstrateur doit faire ressortir le depassement du taux de defaut observe par rapport a la PD moyenne.

## 5-10 min - Points a montrer

- Synthese executive : statut global et compte des alertes.
- Calibration PD : PD moyenne vs ODR par rating grade.
- Alertes : priorisation rouge/orange/gris.
- Monitoring temporel : evolution par millesime et portefeuille.

## 10-14 min - Discrimination et stabilite

Montrer AUC, Gini, KS, ROC, CAP, puis PSI par portfolio et segment.

Expliquer que calibration, discrimination et stabilite repondent a trois questions differentes : le niveau de risque, le pouvoir de tri, et la derive de population.

## 14-17 min - Rapport & export

Montrer la narrative automatique et les exports CSV/HTML.

## 17-20 min - Questions probables

Q : Peut-on brancher de vraies donnees ?
R : Oui, sous reserve de respecter le schema minimal documente.

Q : Les seuils sont-ils configurables ?
R : Oui, via les profils standard, conservative et tolerant.

Q : Comment traiter les portefeuilles low-default ?
R : Les tests non interpretables sont grises et documentes pour eviter une interpretation excessive.

Q : Peut-on etendre vers LGD/EAD ?
R : L'architecture separe deja generation, data quality, calculs, alerting et reporting ; LGD/EAD peuvent etre ajoutes dans des modules dedies.
# Sequence V2 - demonstration reglementaire enrichie

Pour une demonstration client de 20 minutes, le scenario recommande est `rds_population_shift`.

1. Montrer la synthese executive et rappeler que les donnees sont fictives.
2. Ouvrir **Donnees & qualite** pour presenter le waterfall population eligible et l'analyse des exclusions.
3. Ouvrir **Stabilite Rating / Score Distribution** pour montrer le PSI, les contributeurs, la matrice de migration et les nouveaux entrants/sorties.
4. Ouvrir **Calibration PD** pour expliquer la lecture PIT/TTC, les PD raw/calibrated/regulatory, les floors et MoC.
5. Ouvrir **Alertes** pour montrer la consolidation traffic light et les findings.

Message commercial : l'outil ne se limite pas a calculer des indicateurs ; il structure une lecture validation autour du perimetre eligible, de la stabilite de population, des migrations et des ajustements reglementaires de PD.
