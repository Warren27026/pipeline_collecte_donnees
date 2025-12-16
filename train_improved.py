import os
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit

# Configuration
DATA_FOLDER = "data"
FILE_PATH = os.path.join(DATA_FOLDER, "ALL_YFINANCE_features.csv")

# Liste étendue des features (Niveau 2)
FEATURES = [
    # --- Indicateurs Techniques (Existants) ---
    "RSI_14", "MACD", "MACD_Signal", "MACD_Diff", 
    "Bollinger_Width", "Bollinger_%B",
    "Return", "Volume",
    
    # --- NOUVEAUX : Mémoire (Lags) ---
    "Return_D-1", "Return_D-2",  # Ce qui s'est passé hier et avant-hier
    "RSI_D-1",                   # Le RSI d'hier
    
    # --- NOUVEAUX : Tendance & Volatilité ---
    "Dist_SMA_50",               # Distance au prix moyen des 50 derniers jours
    "ATR_14",                    # Volatilité (Average True Range)
    
    # --- NOUVEAU : Saisonnalité ---
    "Day_Of_Week"                # Lundi=0, ..., Vendredi=4
]

def add_advanced_features(df):
    """ Ajoute les indicateurs avancés pour le ML """
    df = df.copy()
    
    # 1. Calcul des Lags (Mémoire)
    df["Return_D-1"] = df["Return"].shift(1)
    df["Return_D-2"] = df["Return"].shift(2)
    df["RSI_D-1"] = df["RSI_14"].shift(1)
    
    # 2. Tendance (SMA 50)
    # On utilise la librairie 'ta' comme dans ton pipeline
    sma_50 = ta.trend.SMAIndicator(df["Close"], window=50).sma_indicator()
    df["Dist_SMA_50"] = (df["Close"] - sma_50) / sma_50
    
    # 3. Volatilité (ATR)
    atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14)
    df["ATR_14"] = atr.average_true_range()
    
    # 4. Date (Jour de la semaine)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["Day_Of_Week"] = df["date"].dt.dayofweek
    else:
        # Fallback si pas de date, on met 0 (ne devrait pas arriver)
        df["Day_Of_Week"] = 0
        
    return df

def train_improved():
    print("\n DÉMARRAGE : MODÈLE AMÉLIORÉ (FEATURE ENGINEERING)...")
    
    if not os.path.exists(FILE_PATH):
        print(f" Fichier introuvable : {FILE_PATH}")
        return
    
    df_all = pd.read_csv(FILE_PATH)
    
    # On filtre les actions qui ont peu d'historique (pour éviter les erreurs de calcul SMA50)
    symbols = df_all["symbol"].unique()
    results = []

    print(f"Features utilisées : {len(FEATURES)} indicateurs")

    for sym in symbols:
        # Préparation
        df = df_all[df_all["symbol"] == sym].copy()
        
        # Tri par date CRUCIAL avant les shifts/lags
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            
        # --- ÉTAPE CLÉ : On ajoute les nouveaux indicateurs ---
        try:
            df = add_advanced_features(df)
        except Exception as e:
            print(f" Erreur features sur {sym}: {e}")
            continue
            
        # Nettoyage des NaN créés par les Lags et SMA50 (les 50 premiers jours seront perdus)
        df = df.dropna().reset_index(drop=True)
        
        if len(df) < 500:
            continue

        # Split Temporel
        split_idx = int(len(df) * 0.8)
        
        X = df[FEATURES]
        y = (df["Return_next"] > 0).astype(int)
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Entraînement (Random Forest un peu plus robuste)
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_depth=15, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # Score
        preds = rf.predict(X_test)
        acc_model = accuracy_score(y_test, preds)
        
        # Baseline
        majority_class = y_train.mode()[0]
        acc_baseline = accuracy_score(y_test, [majority_class] * len(y_test))
        
        diff = (acc_model - acc_baseline) * 100
        
        # On sauvegarde aussi l'importance des features pour voir si nos ajouts servent
        top_feature = FEATURES[np.argmax(rf.feature_importances_)]
        
        print(f" {sym:<8} | IA: {acc_model*100:.2f}% | Hasard: {acc_baseline*100:.2f}% | Diff: {diff:+.2f} pts | Top: {top_feature}")
        
        results.append({
            "Symbol": sym,
            "Accuracy_IA": acc_model,
            "Diff": diff
        })

    # Bilan
    if results:
        df_res = pd.DataFrame(results)
        print("\n" + "="*50)
        print(f" RÉSULTATS MODÈLE AMÉLIORÉ")
        print("="*50)
        print(f"Précision Moyenne IA : {df_res['Accuracy_IA'].mean()*100:.2f}%")
        print(f"Amélioration Moyenne : {df_res['Diff'].mean():+.2f} points")
        
        # Analyse des features globales (sur le dernier modèle)
        print("\n TOP 5 INDICATEURS LES PLUS UTILES :")
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        for f in range(5):
            print(f"   {f+1}. {FEATURES[indices[f]]} ({importances[indices[f]]:.4f})")

if __name__ == "__main__":
    train_improved()
