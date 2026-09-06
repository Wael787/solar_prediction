# ☀️ Solar Power Prediction — Random Forest & Dashboard interactif

**Démo :** [wael787.github.io/solar_prediction](https://wael787.github.io/solar_prediction/)
**Auteur :** Wael WADIH — Étudiant ingénieur, CY Tech

---

## 📌 Le projet

Prédire la production d'une centrale photovoltaïque (en kW) à partir de variables
météorologiques et astronomiques, puis rendre le modèle explorable dans un dashboard web.

Le projet a été **repris et corrigé** après une relecture critique : la première version
annonçait un R² de 0,96 obtenu avec un protocole d'évaluation défaillant. Ce README
documente les résultats mesurés sous un protocole corrigé, et pourquoi ils sont plus bas.

---

## 📊 Données

*Solar Energy Power Generation Dataset* (Kaggle — `stucom`)

| | |
|---|---|
| Observations | 4 213 (horaires) |
| Variables | 21 |
| Cible | `generated_power_kw`, de 0,0006 à 3 056,79 kW (moyenne 1 134 kW) |
| Valeurs manquantes | **aucune** (vérifié : 4 213 non-null sur les 21 colonnes) |
| Heures à production nulle | **0 %** — le jeu ne contient que des heures de jour |

Onze variables retenues, choisies pour leur pertinence physique : rayonnement,
géométrie solaire (zénith, angle d'incidence, azimut), couverture nuageuse à quatre
niveaux, température, humidité, vent.

---

## 🎯 Résultats

Random Forest — 200 arbres, `max_depth=15`, `min_samples_leaf=2`.
Le modèle est entraîné **deux fois**, avec deux protocoles de découpage train/test.

| | Split aléatoire | **Split chronologique** |
|---|---|---|
| R² train | 0,945 | 0,948 |
| **R² test** | 0,802 | **0,517** |
| MAE test | 269 kW | **490 kW** |
| RMSE test | 426 kW | **631 kW** |

**Le chiffre retenu est 0,517.**

### Pourquoi le plus bas est le bon

Les observations sont **horaires**. Un `train_test_split` aléatoire envoie 14h et 15h
du même jour de part et d'autre de la frontière train/test — deux lignes quasi identiques.
Le modèle ne généralise pas : il retrouve un voisin déjà mémorisé. C'est une **fuite de
données temporelle**, et elle vaut ici **+0,28 de R²**.

Le split chronologique réserve les 843 dernières heures au test : le modèle doit prédire
des journées qu'il n'a jamais vues. C'est la tâche réelle.

### Ce que disent les autres chiffres

- **Surapprentissage marqué** : 0,948 en train contre 0,517 en test. Le modèle mémorise
  bien, généralise mal.
- **MAE de 490 kW** pour une production moyenne de 1 134 kW, soit environ 43 % d'erreur
  relative. Utilisable pour dégager une tendance, pas pour du pilotage fin.
- **Baseline** : prédire systématiquement la moyenne donne un R² de 0 par définition.
  Le modèle apprend donc quelque chose de réel, mais reste loin d'un niveau exploitable.

# Sections à ajouter au README

> À insérer **après** la section « 🎯 Résultats », avant « 📈 Importance des variables ».
> La section « 🚧 Limites et pistes » existante est à remplacer par celle donnée en fin de document.

---

## 🧪 À quoi comparer ce résultat ?

Un R² de 0,517 ne veut rien dire sans point de comparaison. Trois modèles ont été
évalués sur le **même** jeu de test chronologique (843 heures) :

| Modèle | R² test | MAE test |
|---|---|---|
| Moyenne constante (baseline naïve) | −0,028 | 840,2 kW |
| Régression linéaire | **0,517** | 531,3 kW |
| Random Forest (200 arbres, `max_depth=15`) | **0,517** | 489,9 kW |

Deux enseignements.

**La baseline obtient un R² négatif.** Prédire systématiquement la moyenne du jeu
d'entraînement fait *pire que rien* sur le test. La moyenne du train n'est pas celle
du test : les 843 dernières heures forment une période saisonnière distincte. Le jeu
de test est donc réellement hors distribution — ce qui rend l'exercice plus difficile,
mais plus honnête.

**Une régression linéaire obtient exactement le même R² que le Random Forest.**
La forêt ne conserve un avantage que sur le MAE (490 contre 531 kW, soit 8 %).
Autrement dit : 200 arbres et un réglage d'hyperparamètres expliquent la même part de
variance qu'une équation à onze coefficients.

Ce résultat suggère que la relation entre les variables et la production est
largement **additive** : peu d'interactions fortes, peu de courbures marquées.
Une explication mécanique le renforce — le rayonnement, le zénith et l'angle
d'incidence sont fortement corrélés entre eux (jusqu'à −0,80) et portent la même
information physique. Là où trois variables disent la même chose, une somme pondérée
résume aussi bien qu'un découpage en paliers.

---

## 🔬 Le score dépend-il du réglage du modèle ?

L'écart entre R² train (0,948) et R² test (0,517) évoque un surapprentissage.
Pour le vérifier, 24 combinaisons de `max_depth` et `min_samples_leaf` ont été
évaluées, de la plus contrainte à la totalement libre.

| `max_depth` | `min_samples_leaf` | R² train | R² test |
|---|---|---|---|
| 4 | 2 | 0,774 | 0,371 |
| 6 | 2 | 0,832 | 0,451 |
| 8 | 2 | 0,883 | 0,486 |
| 10 | 2 | 0,919 | 0,505 |
| **15** | **2** | **0,948** | **0,517** |
| 20 | 2 | 0,950 | 0,516 |
| 30 | 2 | 0,950 | 0,516 |
| `None` | 2 | 0,950 | 0,516 |
| 15 | 30 | 0,818 | 0,441 |

Le R² test **augmente** avec la complexité, puis se stabilise autour de 0,517
sans jamais redescendre. Augmenter `min_samples_leaf` dégrade systématiquement
le résultat.

Ce n'est donc pas le schéma d'un surapprentissage classique, où le score de test
culmine puis chute. Brider le modèle ne fait que le dégrader. L'écart train/test
ne mesure pas ici un excès de complexité à corriger, mais la **difficulté
intrinsèque** de la tâche.

> ⚠️ Ce tableau est une **analyse de sensibilité**, pas une sélection
> d'hyperparamètres. Choisir un réglage sur la base du R² test reviendrait à
> laisser le jeu de test influencer le modèle — une fuite de données plus subtile
> que celle du découpage, mais réelle. Une sélection propre exigerait un jeu de
> validation distinct, ou une validation croisée sur le train. Le modèle retenu
> reste celui d'origine.


## 📈 Importance des variables

Mesurée par `feature_importances_` (réduction de variance — et non impureté de Gini,
qui est un critère de classification).

| Variable | Importance |
|---|---|
| `angle_of_incidence` | ≈ 48 % |
| `total_cloud_cover_sfc` | ≈ 14 % |
| `azimuth` | ≈ 11 % |
| `zenith` | ≈ 10 % |
| `shortwave_radiation_backwards_sfc` | ≈ 6 % |
| autres (humidité, vent, température, nuages par couche) | ≈ 11 % |

**Le rayonnement solaire n'arrive que cinquième**, alors qu'il est physiquement le
facteur premier. L'explication est dans la heatmap de corrélation : il est corrélé
à −0,80 avec le zénith et à −0,58 avec l'angle d'incidence. Ces variables portent la
même information — la position du soleil. Face à des variables redondantes, un arbre
en choisit une arbitrairement, et l'importance se **dilue**.

Cette métrique ne dit donc pas que le rayonnement est inutile, mais qu'il est
**remplaçable**. Une permutation importance donnerait une lecture plus fiable.

---

## 🔍 Test de cohérence physique

Deux scénarios fictifs soumis au modèle :

| Conditions | Prédiction |
|---|---|
| Ciel dégagé, rayonnement 350 W/m², 20 °C | 2 207 kW |
| Ciel couvert 90 %, rayonnement 30 W/m² | 1 804 kW |

Le sens est correct, **mais l'amplitude ne l'est pas** : diviser le rayonnement par
douze ne fait chuter la prédiction que de 18 %. Cohérent avec l'importance des
variables — le modèle s'appuie sur la géométrie solaire plutôt que sur le rayonnement
mesuré.

---

## 🛠️ Stack

**Analyse & ML** — Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn
**Web** — HTML5, CSS3 (CSS Grid, dark mode), JavaScript ES6, Chart.js, Canvas 2D
**Plateformes** — Kaggle (exécution du notebook), GitHub Pages (hébergement statique)

---

## 💻 Utilisation

1. **Analyse** — ouvrir `notebook_solar_prediction.ipynb` dans Jupyter ou Kaggle.
   Le dataset doit être attaché ; adapter le chemin du `read_csv` si besoin.
2. **Dashboard** — ouvrir `index.html`, ou consulter la
   [démo live](https://wael787.github.io/solar_prediction/).

> ⚠️ Le dashboard n'exécute pas le Random Forest. GitHub Pages étant un hébergement
> statique, il ne peut pas faire tourner Python. Le dashboard utilise une **fonction
> de score physique écrite en JavaScript**, qui approxime le comportement du modèle
> pour permettre l'exploration interactive. Les métriques affichées proviennent du
> notebook.

---

## 🚧 Pourquoi 0,517 — et ce qui manque

Trois approches indépendantes butent au même endroit :

- changer de famille d'algorithme (régression linéaire) → 0,517
- faire varier la complexité sur 24 réglages → plafond à 0,517
- baseline naïve → −0,028

La limite ne vient donc ni de l'algorithme ni de son réglage. **Elle vient des
données disponibles.** Ce qui manque, par ordre d'importance probable :

**1. La dimension temporelle.** Le jeu de données ne contient aucune colonne de
date ni d'heure. Impossible de construire des variables décalées
(`production_h-1`, moyenne glissante), alors que l'autocorrélation est
généralement l'information la plus prédictive d'une série temporelle : la
production à 15h dépend fortement de celle à 14h. La saisonnalité est également
inaccessible.

**2. Les caractéristiques de l'installation.** Puissance crête, orientation et
inclinaison des panneaux, technologie, âge, encrassement, ombrage, arrêts de
maintenance, et surtout **température de cellule** — distincte de la température
de l'air, et déterminante pour le rendement. Une chute de production due à un
onduleur défaillant est, pour ce modèle, du bruit pur.

**3. La décomposition du rayonnement.** Seul le rayonnement global est fourni.
Sa séparation en composantes directe et diffuse affecte différemment un panneau
selon son inclinaison.

### Pistes retenues

- Obtenir un jeu de données horodaté permettant des variables décalées et une
  validation par `TimeSeriesSplit` (plusieurs plis successifs plutôt qu'une
  seule coupure, ce qui séparerait l'effet de la fuite de celui du décalage
  saisonnier).
- Mettre en place un jeu de validation distinct pour sélectionner les
  hyperparamètres sans contaminer le test.
- Grouper les variables colinéaires (zénith, angle d'incidence, rayonnement)
  pour une permutation par blocs — ni la MDI ni la permutation simple ne
  gèrent la colinéarité.

### Ce qui n'est **pas** une piste

Tester un gradient boosting. Deux familles d'algorithmes et 24 réglages
convergent vers la même valeur : le gain attendu d'un modèle plus sophistiqué
est marginal au regard de ce qui manque en amont.



*Projet réalisé dans le cadre d'une montée en compétences en data science.
Les chiffres publiés ici sont ceux effectivement produits par le notebook,
reproductibles via `Restart & Run All`.*
