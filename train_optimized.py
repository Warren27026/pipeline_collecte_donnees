import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import TimeSeriesSplit

# --- CONFIGURATION ---
DATA_FOLDER = "data"
FILE_PATH = os.path.join(DATA_FOLDER, "ALL_YFINANCE_features.csv")
TARGET_SYMBOL = "MSFT"  # On se concentre sur Microsoft pour commencer (plus propre)

# On garde les mêmes features
FEATURES = [
    "RSI_14", 
    "MACD", "MACD_Signal", "MACD_Diff", 
    "Bollinger_Width", "Bollinger_%B",
    "Return", "Volume"
]

def train_optimized():
    print(f"\n DÉMARRAGE V2 (Optimisé) sur {TARGET_SYMBOL}...")
    
    # 1. Chargement & Filtrage
    if not os.path.exists(FILE_PATH):
        print(f" Fichier introuvable : {FILE_PATH}")
        return
    
    df_all = pd.read_csv(FILE_PATH)
    
    # FILTRE SUR UN SEUL SYMBOLE (Crucial pour débuter)
    df = df_all[df_all["symbol"] == TARGET_SYMBOL].copy()
    
    # Nettoyage
    df = df.dropna().reset_index(drop=True)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
    
    print(f"Données filtrées pour {TARGET_SYMBOL} : {len(df)} jours de trading.")

    # 2. Split Temporel (80% Train / 20% Test)
    split_idx = int(len(df) * 0.8)
    
    # On prépare les données
    X = df[FEATURES]
    y_return = df["Return_next"]  # CIBLE = % de variation (et non le prix)
    y_class = (df["Return_next"] > 0).astype(int) # CIBLE = Hausse/Baisse
    
    # Prix réels (pour la reconstruction à la fin)
    prices = df["Close"]
    prices_test = prices.iloc[split_idx:]
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_ret_train, y_ret_test = y_return.iloc[:split_idx], y_return.iloc[split_idx:]
    y_class_train, y_class_test = y_class.iloc[:split_idx], y_class.iloc[split_idx:]
    
    # ====================================================
    # 3. RÉGRESSION OPTIMISÉE (Prédire le Rendement)
    # ====================================================
    print("\n Entraînement Régression (Objectif : Return_next)...")
    
    # On augmente un peu la puissance du modèle
    rf_reg = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train, y_ret_train)
    
    # Prédiction du % de variation
    pred_returns = rf_reg.predict(X_test)
    
    # --- RECONSTRUCTION DU PRIX ---
    # Prix_Predit = Prix_Hier * (1 + Return_Predit)
    # Attention : Pour prédire J+1, on utilise le prix de J (qui est dans X_test, mais il faut le récupérer)
    # Ici, prices_test correspond aux prix de J ("Close"). On veut prédire "Close_next".
    
    predicted_prices = prices_test * (1 + pred_returns)
    true_prices_next = df["Close_next"].iloc[split_idx:] # La vraie cible J+1
    
    # Baseline Naïve : On prédit que Prix_J+1 = Prix_J
    naive_prices = prices_test
    
    # Calcul des erreurs
    rmse_model = np.sqrt(mean_squared_error(true_prices_next, predicted_prices))
    rmse_naive = np.sqrt(mean_squared_error(true_prices_next, naive_prices))
    
    print(f"--- RÉSULTATS RÉGRESSION ({TARGET_SYMBOL}) ---")
    print(f"RMSE Modèle (Via Return) : {rmse_model:.4f} $")
    print(f"RMSE Naïf (Baseline)     : {rmse_naive:.4f} $")
    
    if rmse_model < rmse_naive:
        print(" VICTOIRE : Le modèle bat la baseline naïve !")
    elif rmse_model < rmse_naive * 1.05:
        print(" ÉGALITÉ : Le modèle est très proche de la baseline (C'est bon signe).")
    else:
        print(" DÉFAITE : Encore un peu de travail (Feature Engineering nécessaire).")

    # ====================================================
    # 4. CLASSIFICATION (Direction)
    # ====================================================
    print("\n🎲 Entraînement Classification...")
    rf_class = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf_class.fit(X_train, y_class_train)
    
    pred_class = rf_class.predict(X_test)
    acc_model = accuracy_score(y_class_test, pred_class)
    
    # Baseline
    acc_baseline = accuracy_score(y_class_test, [y_class_train.mode()[0]] * len(y_class_test))
    
    print(f"--- RÉSULTATS CLASSIFICATION ({TARGET_SYMBOL}) ---")
    print(f"Précision IA     : {acc_model*100:.2f}%")
    print(f"Précision Hasard : {acc_baseline*100:.2f}%")
    
    if acc_model > acc_baseline:
        print(f" VICTOIRE : +{acc_model*100 - acc_baseline*100:.2f} points au-dessus du hasard.")
    else:
        print(" DÉFAITE : Difficile de battre le marché.")

    # ====================================================
    # 5. VISUALISATION (Pour comprendre)
    # ====================================================
    print("\n Génération du graphique de prédiction...")
    plt.figure(figsize=(12, 6))
    # On affiche juste les 100 derniers jours pour y voir clair
    subset_true = true_prices_next.iloc[-100:]
    subset_pred = predicted_prices.iloc[-100:]
    subset_naive = naive_prices.iloc[-100:]
    
    plt.plot(subset_true.index, subset_true.values, label="Prix Réel", color="black", linewidth=2)
    plt.plot(subset_pred.index, subset_pred.values, label="Prédiction IA (Via Returns)", color="green", linestyle="--")
    plt.plot(subset_naive.index, subset_naive.values, label="Naïf (J-1)", color="red", alpha=0.3)
    
    plt.title(f"Prédiction {TARGET_SYMBOL} : IA vs Réalité")
    plt.legend()
    plt.savefig(os.path.join(DATA_FOLDER, "prediction_v2.png"))
    print("Graphique sauvegardé : data/prediction_v2.png")

if __name__ == "__main__":
    train_optimized()
