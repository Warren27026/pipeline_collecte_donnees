import pandas as pd
import numpy as np
import optuna
import os
import ta
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.dummy import DummyClassifier

# --- CONFIGURATION ---
DATA_FILE = "data/ALL_YFINANCE_features.csv"
TARGET_SYMBOL = "TSLA"  # On optimise sur une action volatile
N_TRIALS = 30           # 30 essais suffisent pour un Random Forest

# Liste EXACTE des features utilisées dans ton predict_final_hybrid.py
FEATURES = [
    "RSI_14", "MACD", "MACD_Signal", "MACD_Diff", 
    "Bollinger_Width", "Bollinger_%B",
    "Return", "Volume",
    "Return_D-1", "Return_D-2", "RSI_D-1",
    "Dist_SMA_50", "ATR_14", "Day_Of_Week"
]

def add_advanced_features(df):
    """ Feature Engineering identique à la production """
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
    
    # Features Avancées
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

def load_data(symbol):
    if not os.path.exists(DATA_FILE):
        print(f"❌ Données introuvables : {DATA_FILE}")
        return None
    
    df = pd.read_csv(DATA_FILE)
    df = df[df['symbol'] == symbol].copy()
    
    try:
        df = add_advanced_features(df)
    except Exception as e:
        print(f"Erreur Feature Engineering: {e}")
        return None

    # Cible : Est-ce que le rendement de DEMAIN est positif ?
    df["Target"] = (df["Return"].shift(-1) > 0).astype(int)
    
    # On nettoie les NaN (liés aux indicateurs et au shift)
    df = df.dropna()
    return df

def objective(trial):
    df = load_data(TARGET_SYMBOL)
    if df is None or len(df) < 200:
        return 0.5 
        
    X = df[FEATURES]
    y = df["Target"]
    
    # --- ESPACE DE RECHERCHE RANDOM FOREST ---
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800), # Nombre d'arbres
        'max_depth': trial.suggest_int('max_depth', 5, 30),          # Profondeur (RF aime la profondeur)
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10), # Protection contre overfitting
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'bootstrap': True,
        'n_jobs': -1,
        'random_state': 42
    }
    
    model = RandomForestClassifier(**param)
    
    # Validation Croisée Temporelle (3 blocs)
    tscv = TimeSeriesSplit(n_splits=3)
    scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
    
    return scores.mean()

def run_optimization():
    print(f"🌲 Optimisation Random Forest pour {TARGET_SYMBOL} ({N_TRIALS} essais)...")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS)
    
    best_acc = study.best_value
    print("\n🏆 RÉSULTATS OPTUNA (RF) :")
    print(f"Meilleure Accuracy : {best_acc:.2%}")
    print("Meilleurs Paramètres :")
    print(study.best_params)
    
    # --- REALITY CHECK ---
    print("\n⚖️ REALITY CHECK (Vs Dummy)...")
    df = load_data(TARGET_SYMBOL)
    if df is not None:
        X = df[FEATURES]
        y = df["Target"]
        
        dummy = DummyClassifier(strategy="most_frequent")
        tscv = TimeSeriesSplit(n_splits=3)
        dummy_score = cross_val_score(dummy, X, y, cv=tscv, scoring='accuracy').mean()
        
        print(f"Score DUMMY : {dummy_score:.2%}")
        print(f"Gain RF     : {best_acc - dummy_score:+.2%}")

    # Sauvegarde
    with open("data/best_rf_params.txt", "w") as f:
        f.write(str(study.best_params))

if __name__ == "__main__":
    run_optimization()
