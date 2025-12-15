import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score
from sklearn.model_selection import TimeSeriesSplit

# Configuration
DATA_FOLDER = "data"
FILE_PATH = os.path.join(DATA_FOLDER, "ALL_YFINANCE_features.csv")

# Liste des features (colonnes) qu'on donne au modèle pour apprendre
FEATURES = [
    "RSI_14", 
    "MACD", "MACD_Signal", "MACD_Diff", 
    "Bollinger_Width", "Bollinger_%B",
    "Return", "Volume"
]

def train_and_evaluate():
    print("\n DÉMARRAGE DE L'ENTRAÎNEMENT (RANDOM FOREST)...")
    
    # 1. Chargement
    if not os.path.exists(FILE_PATH):
        print(f" Fichier introuvable : {FILE_PATH}")
        return
    
    df = pd.read_csv(FILE_PATH)
    
    # Nettoyage des NaN (le ML déteste les trous)
    df = df.dropna().reset_index(drop=True)
    
    # On s'assure que les dates sont triées (CRUCIAL pour les séries temporelles)
    # Sinon le modèle apprend le futur pour prédire le passé !
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
    
    # Préparation des cibles
    # Cible 1 : Prix de demain (Régression)
    y_reg = df["Close_next"]
    
    # Cible 2 : Hausse (1) ou Baisse (0) (Classification)
    y_class = (df["Return_next"] > 0).astype(int)
    
    X = df[FEATURES]
    
    print(f"Données chargées : {len(df)} lignes, {len(FEATURES)} features.")

    # 2. Séparation Train / Test (Méthode Temporelle)
    # On ne fait surtout pas de random_split classique !
    # On prend les 80% plus vieux pour entraîner, et les 20% plus récents pour tester.
    split_idx = int(len(df) * 0.8)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_reg_train, y_reg_test = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]
    y_class_train, y_class_test = y_class.iloc[:split_idx], y_class.iloc[split_idx:]
    
    print(f"Split Temporel : {len(X_train)} train / {len(X_test)} test")
    
    # ====================================================
    # 3. MODÈLE 1 : RÉGRESSION (Prédire le prix exact)
    # ====================================================
    print("\n Entraînement du modèle de RÉGRESSION (Prix)...")
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train, y_reg_train)
    
    # Prédictions
    preds_reg = rf_reg.predict(X_test)
    
    # Évaluation vs Baseline Naïve
    # On compare l'erreur du modèle vs l'erreur si on avait juste dit prix(J+1) = prix(J)
    # Attention : Pour la baseline, on doit prendre le 'Close' du jour J correspondant aux données de test
    baseline_reg_preds = df.iloc[split_idx:]["Close"]
    
    rmse_model = np.sqrt(mean_squared_error(y_reg_test, preds_reg))
    rmse_naive = np.sqrt(mean_squared_error(y_reg_test, baseline_reg_preds))
    
    print(f"--- RÉSULTATS RÉGRESSION ---")
    print(f"RMSE Modèle IA   : {rmse_model:.4f} $")
    print(f"RMSE Naïf (Ref)  : {rmse_naive:.4f} $")
    
    if rmse_model < rmse_naive:
        print(" VICTOIRE : L'IA bat la méthode naïve !")
    else:
        print(" DÉFAITE : L'IA est moins bonne que la méthode naïve (Normal au début).")

    # ====================================================
    # 4. MODÈLE 2 : CLASSIFICATION (Hausse / Baisse)
    # ====================================================
    print("\n Entraînement du modèle de CLASSIFICATION (Signal)...")
    rf_class = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    rf_class.fit(X_train, y_class_train)
    
    # Prédictions
    preds_class = rf_class.predict(X_test)
    
    # Évaluation
    acc_model = accuracy_score(y_class_test, preds_class)
    # Baseline : Toujours prédire la classe majoritaire du train set
    majority_class = y_class_train.mode()[0]
    baseline_preds = [majority_class] * len(y_class_test)
    acc_baseline = accuracy_score(y_class_test, baseline_preds)
    
    print(f"--- RÉSULTATS CLASSIFICATION ---")
    print(f"Précision IA     : {acc_model*100:.2f}%")
    print(f"Précision Hasard : {acc_baseline*100:.2f}%")
    
    if acc_model > acc_baseline:
        print(" VICTOIRE : L'IA détecte des patterns !")
    else:
        print("DÉFAITE : L'IA ne bat pas le hasard pour l'instant.")
        
    # ====================================================
    # 5. IMPORTANCE DES FEATURES (Qu'est-ce qui compte ?)
    # ====================================================
    print("\n Analyse : Quels indicateurs sont importants ?")
    importances = rf_class.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for f in range(len(FEATURES)):
        print(f"{f+1}. {FEATURES[indices[f]]} ({importances[indices[f]]:.4f})")

if __name__ == "__main__":
    train_and_evaluate()
