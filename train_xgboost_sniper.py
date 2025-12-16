import os
import pandas as pd
import numpy as np
import ta
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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
    """ Feature Engineering habituel """
    df = df.copy()
    if len(df) < 50: return df
    
    close = df["Close"]
    df["Return"] = close.pct_change()
    
    # BB
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["Bollinger_Upper"] = bb.bollinger_hband()
    df["Bollinger_Lower"] = bb.bollinger_lband()
    df["Bollinger_%B"] = (close - df["Bollinger_Lower"]) / (df["Bollinger_Upper"] - df["Bollinger_Lower"])
    df["Bollinger_Width"] = df["Bollinger_Upper"] - df["Bollinger_Lower"]
    
    # RSI / MACD
    df["RSI_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd = ta.trend.MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Diff"] = macd.macd_diff()
    
    # Advanced
    df["Return_D-1"] = df["Return"].shift(1)
    df["Return_D-2"] = df["Return"].shift(2)
    df["RSI_D-1"] = df["RSI_14"].shift(1)
    sma_50 = ta.trend.SMAIndicator(close, window=50).sma_indicator()
    df["Dist_SMA_50"] = (close - sma_50) / sma_50
    atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], close, window=14)
    df["ATR_14"] = atr.average_true_range()
    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["Day_Of_Week"] = df["date"].dt.dayofweek
    else:
        df["Day_Of_Week"] = 0
        
    return df

def train_xgboost_sniper():
    print("\n DUEL COMPLET : RANDOM FOREST vs XGBOOST (Mode Sniper > 60%)...")
    
    if not os.path.exists(FILE_PATH):
        print(f" Fichier introuvable : {FILE_PATH}")
        return
    
    df_all = pd.read_csv(FILE_PATH)
    
    # ON PREND TOUS LES SYMBOLES DISPONIBLES
    symbols = df_all["symbol"].unique()
    print(f"Analyse en cours sur {len(symbols)} actifs...")

    print(f"\n{'SYMBOLE':<8} | {'RF (Précision)':<15} | {'XGB (Précision)':<15} | {'VAINQUEUR'}")
    print("-" * 75)
    
    rf_wins = 0
    xgb_wins = 0

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
        
        # --- 1. RANDOM FOREST ---
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_depth=15, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        probs_rf = rf.predict_proba(X_test)[:, 1]
        
        # Score Sniper RF
        mask_rf = probs_rf > 0.60
        score_rf = 0.0
        if np.sum(mask_rf) > 5:
            preds = (probs_rf[mask_rf] > 0.5).astype(int)
            real = y_test[mask_rf]
            score_rf = accuracy_score(real, preds)
            
        # --- 2. XGBOOST ---
        model_xgb = xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5, 
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, eval_metric="logloss"
        )
        model_xgb.fit(X_train, y_train)
        probs_xgb = model_xgb.predict_proba(X_test)[:, 1]
        
        # Score Sniper XGB
        mask_xgb = probs_xgb > 0.60
        score_xgb = 0.0
        if np.sum(mask_xgb) > 5:
            preds = (probs_xgb[mask_xgb] > 0.5).astype(int)
            real = y_test[mask_xgb]
            score_xgb = accuracy_score(real, preds)
        
        # Résultat
        winner = "-"
        if score_xgb > score_rf: 
            winner = "🚀 XGB"
            xgb_wins += 1
        elif score_rf > score_xgb: 
            winner = " RF"
            rf_wins += 1
        
        # Affichage
        str_rf = f"{score_rf*100:.1f}%" if score_rf > 0 else "N/A"
        str_xgb = f"{score_xgb*100:.1f}%" if score_xgb > 0 else "N/A"
        
        print(f"{sym:<8} | {str_rf:<15} | {str_xgb:<15} | {winner}")

    print("-" * 75)
    print(f"BILAN FINAL : RF {rf_wins} victoires vs XGB {xgb_wins} victoires")

if __name__ == "__main__":
    train_xgboost_sniper(
