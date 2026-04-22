# ☀️ Solar Power Prediction - IA & Dashboard Interactif

Démo Live : "https://github.com/Wael787/solar_prediction](https://wael787.github.io/solar_prediction/".

## 📌 Présentation du Projet
Ce projet vise à prédire la production d'énergie photovoltaïque en fonction des variables météorologiques et astronomiques. Il combine une phase d'analyse de données massives (Machine Learning) et une phase de déploiement (Web App).

**Auteur :** Wael WADIH (Étudiant Ingénieur à CY Tech)

## 🚀 Fonctionnalités
* **Modèle IA :** Entraînement d'un régresseur **Random Forest** atteignant un score **R² > 0.95**.
* **Analyse de Données :** Nettoyage, traitement des valeurs manquantes et étude de corrélation (Heatmap).
* **Dashboard Interactif :** Interface web permettant de simuler des scénarios météo en temps réel.
* **Visualisation :** Graphiques dynamiques (Chart.js) montrant les courbes de production sur 24h et l'importance des variables.

## 🛠️ Stack Technique
* **Analyse & ML :** Python (Pandas, Numpy, Scikit-Learn, Seaborn).
* **Web & Déploiement :** HTML5, CSS3 (Dark Mode), JavaScript (ES6+), Chart.js.
* **Plateformes :** Kaggle (Calcul) & GitHub Pages (Hébergement).

## 📊 Performance du Modèle
L'algorithme Random Forest a été optimisé pour minimiser l'erreur :
* **MAE (Mean Absolute Error) :** ~78 kW
* **R² (Coefficient de détermination) :** 0.96 (très haute précision)

## 💻 Installation et Utilisation
1. **Partie Analyse :** Ouvrir le fichier `.ipynb` dans Jupyter ou Kaggle pour voir l'entraînement du modèle.
2. **Partie Dashboard :** Ouvrir `index.html` dans un navigateur ou consulter la Démo Live : "https://github.com/Wael787/solar_prediction](https://wael787.github.io/solar_prediction/".

## 📈 Structure des Variables (Features Importance)
Le modèle identifie les facteurs clés par ordre d'importance :
1. Rayonnement solaire (Shortwave Radiation) - 38%
2. Angle Zénithal (Zenith) - 18%
3. Couverture nuageuse (Cloud Cover) - 8%
4. Température - 5%
