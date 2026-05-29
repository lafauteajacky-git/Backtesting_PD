# Methodologie

## Calibration

ODR = defauts observes / observations.

PD moyenne = moyenne arithmetique des PD individuelles.

Defauts attendus = somme des PD individuelles.

Calibration gap = ODR - PD moyenne.

Calibration ratio = ODR / PD moyenne.

Le test binomial compare les defauts observes aux defauts attendus sous l'hypothese de PD moyenne. Le test unilateral haut sert a detecter une sous-estimation du risque.

## Discrimination

AUC mesure la capacite du score ou de la PD a ordonner defauts et non-defauts.

Gini = 2 x AUC - 1.

KS mesure l'ecart maximum entre les distributions cumulees des defauts et des non-defauts.

## Stabilite

PSI mesure la derive entre une population de reference et une population courante.

PSI < 10 % : derive faible. 10 % a 25 % : derive moderee. Au-dela de 25 % : derive significative.

## Monotonie des grades

Le test de monotonie verifie que le taux de defaut observe augmente lorsque le grade devient plus risque. Une violation locale indique qu'un grade plus risque presente un ODR plus faible que le grade precedent. Plusieurs violations ou une baisse materielle doivent etre investiguees.

## Interpretabilite

Chaque famille de tests distingue les resultats interpretables, les resultats a interpreter avec prudence, les volumes insuffisants, les defauts insuffisants et les donnees manquantes. Cette colonne evite de traiter un signal statistiquement fragile comme un constat ferme.

## Cartographie des tests

La cartographie liste les controles inclus, optionnels ou prevus par theme : data quality, calibration, discrimination, stabilite Rating / Score Distribution, faibles volumes, reporting et alerting.

## Seuils et limites

Les seuils du demonstrateur sont parametrables. Ils ne constituent pas des seuils reglementaires universels et doivent etre adaptes a la politique de validation, au portefeuille et a la materialite du modele.

## Faibles volumes

Les tests sont classes gris lorsque le nombre d'observations, de defauts ou de non-defauts est insuffisant. Cette regle evite de sur-interpreter des resultats statistiquement fragiles, notamment sur les portefeuilles low-default.

## Limites statistiques

Les tests ne remplacent pas l'analyse experte. Les resultats doivent etre rapproches des changements de politique d'octroi, de population, de definition du defaut, de methodologie modele et de cycle economique.
# Complements V2 - lecture reglementaire PD

## RDS

La Rating / Score Distribution Stability compare les distributions de ratings, scores et buckets de PD entre une periode de reference et une periode courante. Le PSI permet de quantifier la derive ; les contributeurs PSI indiquent les grades ou buckets qui expliquent le mouvement.

## Matrice de migration

La matrice de migration compare le rating precedent au rating courant. Elle permet de mesurer la stabilite, les upgrades, les downgrades et les changements de plusieurs crans, en particulier dans les periodes de deterioration.

## Population reglementaire eligible

La population eligible au backtesting PD exclut les expositions deja en defaut a l'observation, hors scope reglementaire, hors scope modele, sans horizon 12 mois complet, cloturees avant horizon ou avec donnees essentielles manquantes.

## PIT / TTC / Hybrid

La lecture PIT / TTC / Hybrid reste limitee aux PD 12 mois du demonstrateur. Elle ne cree pas de PD lifetime et ne constitue pas un module IFRS 9.

Un modele PIT peut varier avec le cycle economique ou la degradation courante du risque. Un modele TTC doit etre relativement plus stable dans le temps ; une forte variation de PD ou de distribution de grades doit donc etre investiguee. Un modele Hybrid combine une composante cyclique et une composante plus stable. Le demonstrateur ne conclut pas qu'une philosophie est meilleure ; il adapte la lecture validation.

Les indicateurs presentes par philosophie sont : nombre d'observations, ODR, PD moyenne, defauts attendus, calibration gap, calibration ratio, volatilite temporelle de la PD moyenne et PSI/RDS lorsque plusieurs periodes sont disponibles.

## PD raw, calibrated, regulatory

La PD raw represente le signal initial. La PD calibrated correspond a la PD ajustee au niveau de calibration cible. La PD regulatory integre les floors et marges de conservatisme applicables.

## Floors et MoC

Les floors imposent un niveau minimal de PD. Les marges de conservatisme couvrent notamment les limites de donnees, de methodologie ou de representativite. Leur impact doit etre trace, quantifie et interprete par segment.

## Low-default portfolios

Les portefeuilles avec peu de defauts exigent une lecture prudente : regroupement de grades, analyse pluriannuelle, benchmark externe et jugement expert peuvent etre necessaires.

## Diagnostic population shift

Un changement de resultat peut venir du modele, mais aussi d'un changement de mix produit, pays, secteur, segment, entrants/sortants ou migrations de rating. Le diagnostic population shift vise a separer ces effets.
