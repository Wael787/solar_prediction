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

---

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

## 🚧 Limites et pistes

- **Le split chronologique confond deux effets.** Les 843 dernières heures forment une
  saison entière, non un échantillon réparti sur l'année. Une partie de la chute de R²
  vient du décalage saisonnier, pas seulement de la fuite supprimée. `TimeSeriesSplit`
  découperait en plusieurs plis successifs et séparerait les deux.
- **Importance des variables non fiabilisée** — la permutation importance reste à faire.
- **Aucun feature engineering** — pas de variable d'heure ni de saison, alors que la
  production est fortement cyclique.
- **Un seul modèle testé** — aucune comparaison avec un gradient boosting ni avec une
  régression linéaire de référence.
- **Le dashboard n'affiche pas encore les prédictions réelles du jeu de test.**

---

*Projet réalisé dans le cadre d'une montée en compétences en data science.
Les chiffres publiés ici sont ceux effectivement produits par le notebook,
reproductibles via `Restart & Run All`.*
