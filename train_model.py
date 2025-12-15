import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# Configuration
DATA_FOLDER = "data"
FILE_PATH = os.path.join(DATA_FOLDER, "ALL_YFINANCE_features.csv")

# Liste des features
FEATURES = [
    "RSI_14", 
    "MACD", "MACD_Signal", "MACD_Diff", 
    "Bollinger_Width", "Bollinger_%B",
    "Return", "Volume"
]

def train_global():
    print("\n DÉMARRAGE : ENTRAÎNEMENT GLOBAL (TOUT LE PORTEFEUILLE)...")
    
    if not os.path.exists(FILE_PATH):
        print(f" Fichier introuvable : {FILE_PATH}")
        return
    
    df_all = pd.read_csv(FILE_PATH)
    
    # Liste des symboles disponibles
    symbols = df_all["symbol"].unique()
    results = []

    print(f"Actions détectées : {symbols}\n")

    for sym in symbols:
        # 1. Préparation des données pour CETTE action
        df = df_all[df_all["symbol"] == sym].copy()
        df = df.dropna().reset_index(drop=True)
        
        if len(df) < 500: # On ignore s'il y a trop peu de données
            print(f" {sym} ignoré (pas assez de données : {len(df)} jours)")
            continue

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")

        # Split Temporel (80% Train / 20% Test)
        split_idx = int(len(df) * 0.8)
        
        X = df[FEATURES]
        y_class = (df["Return_next"] > 0).astype(int) # 1 = Hausse, 0 = Baisse
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y_class.iloc[:split_idx], y_class.iloc[split_idx:]
        
        # 2. Entraînement Classification (Random Forest)
        # On augmente les arbres à 200 pour la stabilité
        rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # 3. Évaluation
        preds = rf.predict(X_test)
        acc_model = accuracy_score(y_test, preds)
        
        # Baseline (Classe majoritaire)
        majority_class = y_train.mode()[0]
        acc_baseline = accuracy_score(y_test, [majority_class] * len(y_test))
        
        diff = (acc_model - acc_baseline) * 100
        result_str = "VICTOIRE " if diff > 0 else "DÉFAITE "
        
        print(f"📊 {sym:<8} | IA: {acc_model*100:.2f}% | Hasard: {acc_baseline*100:.2f}% | Diff: {diff:+.2f} pts | {result_str}")
        
        results.append({
            "Symbol": sym,
            "Accuracy_IA": acc_model,
            "Accuracy_Baseline": acc_baseline,
            "Diff_Points": diff
        })

    # === BILAN GÉNÉRAL ===
    if results:
        df_res = pd.DataFrame(results)
        avg_ia = df_res["Accuracy_IA"].mean() * 100
        avg_base = df_res["Accuracy_Baseline"].mean() * 100
        
        print("\n" + "="*50)
        print(f" MOYENNE GLOBALE DU PORTEFEUILLE")
        print("="*50)
        print(f"Précision Moyenne IA     : {avg_ia:.2f}%")
        print(f"Précision Moyenne Hasard : {avg_base:.2f}%")
        print(f"Gain Moyen (Edge)        : {avg_ia - avg_base:+.2f} points")
        
        # Sauvegarde pour le rapport
        df_res.to_csv(os.path.join(DATA_FOLDER, "ml_results_global.csv"), index=False)
        print(f"\nDétails sauvegardés dans : data/ml_results_global.csv")
    else:
        print("Aucun résultat généré.")

if __name__ == "__main__":
    train_global()
