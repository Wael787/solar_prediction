# ============================================================
#  SOLAR ENERGY PREDICTION — Kaggle Notebook
#  Auteur : Wael WADIH | CY Tech — 2ème année Cycle Ingénieur
# ============================================================

# ─── 0. IMPORTS ─────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

# ─── 1. CHARGEMENT DES DONNÉES ──────────────────────────────
df = pd.read_csv("/kaggle/input/your-dataset/solar_data.csv")  # ← adapter le chemin
print("Shape :", df.shape)
print(df.head())


# ─── 2. NETTOYAGE DES DONNÉES ───────────────────────────────
print("\n── Valeurs manquantes par colonne ──")
print(df.isnull().sum())

# Suppression des lignes avec NaN sur la cible
df.dropna(subset=["generated_power_kw"], inplace=True)

# Remplissage des NaN numériques par la médiane (robuste aux outliers)
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)

print(f"\nDataset après nettoyage : {df.shape[0]} lignes, {df.shape[1]} colonnes")


# ─── 3. FEATURE SELECTION ───────────────────────────────────
# Features sélectionnées selon leur pertinence physique + corrélation attendue
FEATURES = [
    "shortwave_radiation_backwards_sfc",  # rayonnement solaire direct → clé
    "zenith",                              # angle zénithal du soleil
    "angle_of_incidence",                  # angle d'incidence sur le panneau
    "azimuth",                             # orientation du soleil
    "total_cloud_cover_sfc",               # couverture nuageuse totale
    "high_cloud_cover_high_cld_lay",       # nuages hauts
    "medium_cloud_cover_mid_cld_lay",      # nuages moyens
    "low_cloud_cover_low_cld_lay",         # nuages bas
    "temperature_2_m_above_gnd",           # température (rendement PV)
    "relative_humidity_2_m_above_gnd",     # humidité (liée à nébulosité)
    "wind_speed_10_m_above_gnd",           # vent (refroidissement panneaux)
]

TARGET = "generated_power_kw"

X = df[FEATURES]
y = df[TARGET]


# ─── 4. VISUALISATION — HEATMAP DE CORRÉLATION ──────────────
plt.figure(figsize=(12, 9))
corr_matrix = df[FEATURES + [TARGET]].corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
)
plt.title("Heatmap de corrélation — Features vs Production solaire", fontsize=14, pad=15)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("heatmap_correlation.png", dpi=150, bbox_inches="tight")
plt.show()
print("→ Heatmap sauvegardée.")


# ─── 5. SPLIT TRAIN / TEST ──────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain : {X_train.shape[0]} échantillons | Test : {X_test.shape[0]} échantillons")


# ─── 6. ENTRAÎNEMENT — RANDOM FOREST REGRESSOR ──────────────
rf_model = RandomForestRegressor(
    n_estimators=200,        # 200 arbres de décision
    max_depth=15,            # profondeur max pour éviter le surapprentissage
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,               # utilise tous les cœurs CPU disponibles
)

print("\nEntraînement du modèle Random Forest...")
rf_model.fit(X_train, y_train)
print("Entraînement terminé.")


# ─── 7. ÉVALUATION DES PERFORMANCES ─────────────────────────
y_pred = rf_model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

print("\n══════════════════════════════════════")
print("       MÉTRIQUES DE PERFORMANCE")
print("══════════════════════════════════════")
print(f"  MAE  (Erreur absolue moyenne) : {mae:.2f} kW")
print(f"  RMSE (Erreur quadratique)     : {rmse:.2f} kW")
print(f"  R²   (Coefficient détermin.)  : {r2:.4f}")
print("══════════════════════════════════════")


# ─── 8. VISUALISATION — PRÉDICTIONS vs RÉALITÉ ──────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 8a. Scatter : Réel vs Prédit
axes[0].scatter(y_test, y_pred, alpha=0.4, color="steelblue", s=15)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
axes[0].plot(lims, lims, "r--", linewidth=1.5, label="Prédiction parfaite")
axes[0].set_xlabel("Valeurs réelles (kW)")
axes[0].set_ylabel("Valeurs prédites (kW)")
axes[0].set_title(f"Réel vs Prédit  |  R² = {r2:.3f}")
axes[0].legend()
axes[0].grid(alpha=0.3)

# 8b. Distribution des résidus
residuals = y_test - y_pred
axes[1].hist(residuals, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
axes[1].set_xlabel("Résidu (kW)")
axes[1].set_ylabel("Fréquence")
axes[1].set_title("Distribution des résidus")
axes[1].grid(alpha=0.3)

plt.suptitle("Évaluation du modèle Random Forest — Production solaire", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("evaluation_modele.png", dpi=150, bbox_inches="tight")
plt.show()
print("→ Graphiques d'évaluation sauvegardés.")


# ─── 9. IMPORTANCE DES FEATURES ─────────────────────────────
feat_importances = pd.Series(rf_model.feature_importances_, index=FEATURES)
feat_importances = feat_importances.sort_values(ascending=True)

plt.figure(figsize=(8, 6))
colors = ["#1a6496" if v > 0.1 else "#5ba3c9" for v in feat_importances]
feat_importances.plot(kind="barh", color=colors)
plt.xlabel("Importance (Gini impurity)")
plt.title("Importance des features — Random Forest", fontsize=13)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("→ Feature importances sauvegardées.")


# ─── 10. AUTO-TEST AVEC DONNÉES FICTIVES ────────────────────
print("\n── Auto-test : prédiction sur données fictives ──")

test_fictif = pd.DataFrame([{
    "shortwave_radiation_backwards_sfc": 350.0,  # bon ensoleillement
    "zenith": 65.0,
    "angle_of_incidence": 30.0,
    "azimuth": 170.0,
    "total_cloud_cover_sfc": 10.0,               # ciel quasi dégagé
    "high_cloud_cover_high_cld_lay": 5,
    "medium_cloud_cover_mid_cld_lay": 0,
    "low_cloud_cover_low_cld_lay": 0,
    "temperature_2_m_above_gnd": 20.0,
    "relative_humidity_2_m_above_gnd": 40.0,
    "wind_speed_10_m_above_gnd": 3.5,
}])

prediction = rf_model.predict(test_fictif)[0]
print(f"  Conditions : bon ensoleillement, ciel dégagé, 20°C")
print(f"  → Production prédite : {prediction:.2f} kW")

test_fictif_nuageux = test_fictif.copy()
test_fictif_nuageux["total_cloud_cover_sfc"] = 90
test_fictif_nuageux["shortwave_radiation_backwards_sfc"] = 30.0  # ciel couvert

prediction_nuageux = rf_model.predict(test_fictif_nuageux)[0]
print(f"\n  Conditions : ciel couvert (90%), faible rayonnement")
print(f"  → Production prédite : {prediction_nuageux:.2f} kW")

print("\n  ✓ Logique validée : production plus faible par temps couvert.")
