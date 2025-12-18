import pandas as pd
import numpy as np
import optuna
import os
import ta
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier  # <--- NOUVEL IMPORT

# --- CONFIGURATION ---
DATA_FILE = "data/ALL_YFINANCE_features.csv"
TARGET_SYMBOL = "TSLA"  # On optimise sur une action volatile représentative
N_TRIALS = 20           # Nombre d'essais

def load_data(symbol):
    if not os.path.exists(DATA_FILE):
        print(f" Données introuvables : {DATA_FILE}")
        return None
    
    df = pd.read_csv(DATA_FILE)
    df = df[df['symbol'] == symbol].copy()
    
    # Feature Engineering (Doit être identique à ton modèle de prod)
    close = df["Close"]
    df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["SMA_50"] = ta.trend.SMAIndicator(close, window=50).sma_indicator()
    df["Dist_SMA"] = (close - df["SMA_50"]) / df["SMA_50"]
    df["Return"] = close.pct_change()
    df["Return_D1"] = df["Return"].shift(1)
    df["Vol_20"] = df["Return"].rolling(20).std()
    
    # Cible : Hausse demain ?
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    
    df = df.dropna()
    return df

def objective(trial):
    df = load_data(TARGET_SYMBOL)
    if df is None or len(df) < 100:
        return 0.5 # Score neutre si échec
        
    features = ["RSI", "Dist_SMA", "Return", "Return_D1", "Vol_20"]
    X = df[features]
    y = df["Target"]
    
    # L'espace de recherche des hyperparamètres
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'n_jobs': -1,
        'random_state': 42
    }
    
    model = XGBClassifier(**param)
    
    # Validation croisée temporelle
    tscv = TimeSeriesSplit(n_splits=3)
    scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
    
    return scores.mean()

def run_optimization():
    print(f"🚀 Démarrage de l'optimisation Optuna ({N_TRIALS} essais)...")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS)
    
    best_acc = study.best_value
    print("\n RÉSULTATS OPTUNA :")
    print(f"Meilleure Accuracy IA : {best_acc:.2%}")
    print("Meilleurs Paramètres :")
    print(study.best_params)
    
    # --- SECTION AJOUTÉE : COMPARAISON DIRECTE ---
    print("\n REALITY CHECK (Vs Dummy)...")
    df = load_data(TARGET_SYMBOL)
    if df is not None:
        X = df[["RSI", "Dist_SMA", "Return", "Return_D1", "Vol_20"]]
        y = df["Target"]
        
        # On calcule le score du "Hasard intelligent" (Toujours prédire la majorité)
        # On utilise la même méthode de validation croisée pour être 100% fair-play
        dummy = DummyClassifier(strategy="most_frequent")
        tscv = TimeSeriesSplit(n_splits=3)
        dummy_scores = cross_val_score(dummy, X, y, cv=tscv, scoring='accuracy')
        dummy_acc = dummy_scores.mean()
        
        print(f"Score DUMMY (Baseline) : {dummy_acc:.2%}")
        
        gain = best_acc - dummy_acc
        if gain > 0:
            print(f" SUCCÈS : L'optimisation apporte +{gain*100:.2f} points de performance !")
        else:
            print(f" ALERTE : Même optimisé, le modèle ne bat pas le Dummy ({gain*100:.2f} pts).")
            print(" Cela signifie que les indicateurs actuels ne suffisent pas à prédire ce titre.")

    # Sauvegardes habituelles
    df_results = study.trials_dataframe()
    df_results.to_csv("data/optuna_results.csv")
    
    with open("data/best_hyperparameters.txt", "w") as f:
        f.write(str(study.best_params))

if __name__ == "__main__":
    try:
        import optuna
        run_optimization()
    except ImportError:
        print(" Optuna n'est pas installé. Ajoute 'optuna' à requirements.txt")
