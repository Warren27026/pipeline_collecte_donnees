import os
import pandas as pd
import numpy as np
import ta
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# --- CONFIGURATION ---
DATA_FOLDER = "data"
FILE_PATH = os.path.join(DATA_FOLDER, "ALL_YFINANCE_features.csv")

FEATURES = [
    "RSI_14", "MACD", "MACD_Signal", "MACD_Diff", 
    "Bollinger_Width", "Bollinger_%B",
    "Return", "Volume",
    "Return_D-1", "Return_D-2", 
    "Dist_SMA_50", "ATR_14"
]

def add_advanced_features(df):
    """ Feature Engineering """
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
    df["MACD_Diff"] = macd.macd_diff()
    
    # Advanced
    df["Return_D-1"] = df["Return"].shift(1)
    df["Return_D-2"] = df["Return"].shift(2)
    sma_50 = ta.trend.SMAIndicator(close, window=50).sma_indicator()
    df["Dist_SMA_50"] = (close - sma_50) / sma_50
    atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], close, window=14)
    df["ATR_14"] = atr.average_true_range()
    
    return df

def train_price_predictor_v2():
    print("\n🔮 DÉMARRAGE V2 : PRÉDICTION VIA RENDEMENTS (XGBOOST)...")
    
    if not os.path.exists(FILE_PATH):
        print(f" Fichier introuvable : {FILE_PATH}")
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
            
        # --- CHANGEMENT MAJEUR ICI ---
        # On prédit Return_next (le %) au lieu du Prix
        # Cible = Return de demain
        df["Target_Return"] = df["Return"].shift(-1)
        
        # On garde le prix pour la reconstruction
        prices = df["Close"]
        
        df = df.dropna()
        
        # Split 80/20
        split_idx = int(len(df) * 0.8)
        
        X = df[FEATURES]
        y = df["Target_Return"] # On apprend le %
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Prix correspondants au Test Set (pour reconstruire)
        # Attention : Pour prédire le prix de J+1, on part du prix de J
        current_prices_test = prices.iloc[split_idx:len(df)]
        true_future_prices = prices.iloc[split_idx+1:len(df)+1].values # Les vrais prix de demain
        
        # Si décalage d'index, on coupe le dernier
        if len(true_future_prices) < len(current_prices_test):
            current_prices_test = current_prices_test.iloc[:-1]
            X_test = X_test.iloc[:-1]
            y_test = y_test.iloc[:-1]
        
        # --- XGBOOST REGRESSOR (Sur le %) ---
        model = xgb.XGBRegressor(
            n_estimators=300, 
            learning_rate=0.05, 
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        pred_returns = model.predict(X_test)
        
        # --- RECONSTRUCTION DU PRIX ---
        # Prix_Predit = Prix_Aujourd'hui * (1 + %_Predit)
        predicted_prices = current_prices_test.values * (1 + pred_returns)
        
        # Baseline Naïve : On dit que Prix_Demain = Prix_Aujourd'hui
        # RMSE Naïf = écart entre Prix_Reel_Demain et Prix_Aujourd'hui
        rmse_naive = np.sqrt(mean_squared_error(true_future_prices, current_prices_test.values))
        
        # RMSE AI
        rmse_ai = np.sqrt(mean_squared_error(true_future_prices, predicted_prices))
        
        gain = rmse_naive - rmse_ai
        
        status = " WIN" if gain > 0 else " LOSE"
        if gain > 1.0: status = "🔥 BIG WIN"
        
        print(f"{sym:<8} | {rmse_naive:<12.4f} | {rmse_ai:<12.4f} | {gain:<+10.4f} | {status}")
        
        # Sauvegarde Graphique (Seulement si ça marche ou pour analyse)
        if gain > -2.0: # On affiche même les légères défaites pour voir la courbe
            plt.figure(figsize=(10,5))
            # Zoom sur les 100 derniers jours
            plt.plot(true_future_prices[-100:], label="Prix Réel", color="black", alpha=0.6)
            plt.plot(predicted_prices[-100:], label="Prédiction IA", color="#00cc66", linestyle="--")
            plt.title(f"Prédiction Prix {sym} (Via Rendements)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(DATA_FOLDER, f"price_pred_{sym}.png"))
            plt.close()

            # Prédiction pour demain (Production)
            last_row = df.iloc[[-1]][FEATURES]
            next_return = model.predict(last_row)[0]
            last_price = df.iloc[-1]["Close"]
            next_price = last_price * (1 + next_return)
            
            predictions_log.append({
                "Symbol": sym,
                "Current_Price": last_price,
                "Predicted_Return_Pct": round(next_return * 100, 2),
                "Predicted_Price": round(next_price, 2)
            })

    if predictions_log:
        pd.DataFrame(predictions_log).to_csv(os.path.join(DATA_FOLDER, "final_price_predictions.csv"), index=False)
        print(f"\n✅ Prédictions corrigées sauvegardées dans final_price_predictions.csv")

if __name__ == "__main__":
    train_price_predictor_v2()
