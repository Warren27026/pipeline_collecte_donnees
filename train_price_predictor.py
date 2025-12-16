import os
import pandas as pd
import numpy as np
import ta
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# --- CONFIGURATION ---
DATA_FOLDER = "data"
FILE_PATH = os.path.join(DATA_FOLDER, "ALL_YFINANCE_features.csv")

FEATURES = [
    "Close", "RSI_14", "MACD", "MACD_Signal", 
    "Bollinger_Width", "Bollinger_%B",
    "Return", "Volume",
    "Return_D-1", "Return_D-2", 
    "Dist_SMA_50", "ATR_14"
]

def add_advanced_features(df):
    """ Feature Engineering identique """
    df = df.copy()
    if len(df) < 50: return df
    
    close = df["Close"]
    df["Return"] = close.pct_change()
    
    # BB
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["Bollinger_%B"] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
    df["Bollinger_Width"] = bb.bollinger_wband()
    
    # RSI / MACD
    df["RSI_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd = ta.trend.MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    
    # Advanced
    df["Return_D-1"] = df["Return"].shift(1)
    df["Return_D-2"] = df["Return"].shift(2)
    sma_50 = ta.trend.SMAIndicator(close, window=50).sma_indicator()
    df["Dist_SMA_50"] = (close - sma_50) / sma_50
    atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], close, window=14)
    df["ATR_14"] = atr.average_true_range()
    
    return df

def train_price_predictor():
    print("\n🔮 DÉMARRAGE : PRÉDICTION DE PRIX (XGBOOST REGRESSOR)...")
    
    if not os.path.exists(FILE_PATH):
        print(f"❌ Fichier introuvable : {FILE_PATH}")
        return
    
    df_all = pd.read_csv(FILE_PATH)
    symbols = df_all["symbol"].unique()
    
    print(f"{'SYMBOLE':<8} | {'RMSE NAÏF':<12} | {'RMSE AI':<12} | {'GAIN ($)':<10} | {'STATUS'}")
    print("-" * 70)
    
    predictions_log = []

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
            
        # CIBLE : Le prix de DEMAIN ("Close_next")
        # On doit créer cette colonne manuellement car on n'utilise pas le fichier pre-calculé pour tout
        df["Target_Price"] = df["Close"].shift(-1)
        df = df.dropna() # On perd la dernière ligne
        
        # Split 80/20
        split_idx = int(len(df) * 0.8)
        
        X = df[FEATURES]
        y = df["Target_Price"]
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # --- XGBOOST REGRESSOR ---
        # Objective 'reg:squarederror' est optimisé pour réduire le RMSE
        model = xgb.XGBRegressor(
            n_estimators=500,       # Beaucoup d'arbres
            learning_rate=0.01,     # Apprentissage lent et précis
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Évaluation
        rmse_ai = np.sqrt(mean_squared_error(y_test, predictions))
        
        # Baseline Naïve : On prédit que Prix Demain = Prix Aujourd'hui (Close)
        # Attention : X_test["Close"] est le prix d'aujourd'hui
        rmse_naive = np.sqrt(mean_squared_error(y_test, X_test["Close"]))
        
        gain = rmse_naive - rmse_ai
        
        status = "✅ WIN" if gain > 0 else "❌ LOSE"
        if gain > 0.5: status = "🔥 BIG WIN"
        
        print(f"{sym:<8} | {rmse_naive:<12.4f} | {rmse_ai:<12.4f} | {gain:<+10.4f} | {status}")
        
        # Si c'est une victoire notable, on sauvegarde le graph
        if gain > 0:
            plt.figure(figsize=(10,5))
            # On affiche juste les 100 derniers jours pour y voir clair
            subset_real = y_test.iloc[-100:].values
            subset_pred = predictions[-100:]
            
            plt.plot(subset_real, label="Prix Réel", color="black", alpha=0.7)
            plt.plot(subset_pred, label="Prédiction IA", color="#00cc66", linestyle="--")
            plt.title(f"Prédiction Prix {sym} (XGBoost)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(DATA_FOLDER, f"price_pred_{sym}.png"))
            plt.close()
            
            # Pour la production : on prédit VRAIMENT demain
            last_known_data = df.iloc[[-1]][FEATURES]
            future_price = model.predict(last_known_data)[0]
            predictions_log.append({
                "Symbol": sym,
                "Predicted_Price": future_price,
                "Current_Price": df.iloc[-1]["Close"]
            })

    # Sauvegarde des prix futurs
    if predictions_log:
        pd.DataFrame(predictions_log).to_csv(os.path.join(DATA_FOLDER, "final_price_predictions.csv"), index=False)
        print(f"\n✅ Prédictions de prix sauvegardées dans final_price_predictions.csv")

if __name__ == "__main__":
    train_price_predictor()
