import os
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import TimeSeriesSplit

# --- CONFIGURATION ---
DATA_FOLDER = "data"
FILE_PATH = os.path.join(DATA_FOLDER, "ALL_YFINANCE_features.csv")

FEATURES = [
    "RSI_14", "MACD", "MACD_Signal", "MACD_Diff", 
    "Bollinger_Width", "Bollinger_%B",
    "Return", "Volume",
    "Return_D-1", "Return_D-2", "RSI_D-1",
    "Dist_SMA_50", "ATR_14", "Day_Of_Week"
]

def add_advanced_features(df):
    """ (Même feature engineering que le niveau précédent) """
    df = df.copy()
    df["Return_D-1"] = df["Return"].shift(1)
    df["Return_D-2"] = df["Return"].shift(2)
    df["RSI_D-1"] = df["RSI_14"].shift(1)
    sma_50 = ta.trend.SMAIndicator(df["Close"], window=50).sma_indicator()
    df["Dist_SMA_50"] = (df["Close"] - sma_50) / sma_50
    atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14)
    df["ATR_14"] = atr.average_true_range()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["Day_Of_Week"] = df["date"].dt.dayofweek
    else:
        df["Day_Of_Week"] = 0
    return df

def train_sniper():
    print("\n DÉMARRAGE : MODE SNIPER (Optimisation du Seuil)...")
    
    if not os.path.exists(FILE_PATH):
        print(f" Fichier introuvable : {FILE_PATH}")
        return
    
    df_all = pd.read_csv(FILE_PATH)
    symbols = df_all["symbol"].unique()
    
    # On va tester 3 niveaux de confiance
    thresholds = [0.50, 0.55, 0.60]
    
    results = []

    print(f"{'SYMBOLE':<8} | {'STD (50%)':<10} | {'SNIPER (55%)':<12} | {'ELITE (60%)':<12} | {'TRADES (60%)':<10}")
    print("-" * 70)

    for sym in symbols:
        df = df_all[df_all["symbol"] == sym].copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            
        try:
            df = add_advanced_features(df)
        except:
            continue
            
        df = df.dropna().reset_index(drop=True)
        if len(df) < 500: continue

        # Split 80/20
        split_idx = int(len(df) * 0.8)
        
        X = df[FEATURES]
        y = (df["Return_next"] > 0).astype(int)
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Entraînement (Random Forest robuste)
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_depth=15, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # --- C'EST ICI QUE LA MAGIE OPÈRE ---
        # On récupère les probabilités (ex: 0.52, 0.60, 0.45...)
        probs = rf.predict_proba(X_test)[:, 1] # Proba de la classe 1 (Hausse)
        
        row_res = {"Symbol": sym}
        
        for thr in thresholds:
            # On ne trade que si Proba > Seuil
            # Masque des jours où on trade
            mask_trade = probs > thr
            
            if np.sum(mask_trade) > 10: # Il faut au moins 10 trades pour que ce soit significatif
                # Précision sur ces trades là uniquement
                precision = precision_score(y_test[mask_trade], (probs[mask_trade] > 0.5).astype(int), zero_division=0)
                row_res[f"P_{thr}"] = precision * 100
            else:
                row_res[f"P_{thr}"] = np.nan # Pas assez de trades

        # Nombre de trades pris en mode Elite
        nb_trades_elite = np.sum(probs > 0.60)
        
        # Affichage propre
        p50 = f"{row_res.get('P_0.5', 0):.1f}%"
        p55 = f"{row_res.get('P_0.55', 0):.1f}%"
        p60 = f"{row_res.get('P_0.6', 0):.1f}%" if not np.isnan(row_res.get('P_0.6', 0)) else "N/A"
        
        print(f"{sym:<8} | {p50:<10} | {p55:<12} | {p60:<12} | {nb_trades_elite:<10}")
        results.append(row_res)

    print("-" * 70)
    print("LÉGENDE :")
    print(" - STD (50%)    : Précision en tradant tout le temps.")
    print(" - SNIPER (55%) : Précision en tradant seulement quand sûr à 55%.")
    print(" - ELITE (60%)  : Précision en tradant seulement quand sûr à 60%.")

if __name__ == "__main__":
    train_sniper()
