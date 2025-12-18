import pandas as pd
import numpy as np
import optuna
import os
import ta
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score

# --- CONFIGURATION ---
DATA_FILE = "data/ALL_YFINANCE_features.csv"
TARGET_SYMBOL = "TSLA"  # On optimise pour une action volatile (ex: Tesla)
N_TRIALS = 50           # Nombre de tentatives (plus c'est haut, mieux c'est)

def load_data(symbol):
    """ Charge et prépare les données pour un symbole spécifique """
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError("Données introuvables.")
    
    df = pd.read_csv(DATA_FILE)
    df = df[df['symbol'] == symbol].copy()
    
    # Feature Engineering Rapide (Même logique que le script Hybride)
    close = df["Close"]
    df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["SMA_50"] = ta.trend.SMAIndicator(close, window=50).sma_indicator()
    df["Dist_SMA"] = (close - df["SMA_50"]) / df["SMA_50"]
    df["Return"] = close.pct_change()
    df["Return_D1"] = df["Return"].shift(1)
    df["Vol_20"] = df["Return"].rolling(20).std()
    
    # Cible : Est-ce que le prix monte demain ?
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    
    df = df.dropna()
    return df

def objective(trial):
    """ Fonction que Optuna va essayer d'optimiser """
    # 1. Chargement Data
    df = load_data(TARGET_SYMBOL)
    features = ["RSI", "Dist_SMA", "Return", "Return_D1", "Vol_20"]
    X = df[features]
    y = df["Target"]
    
    # 2. La "Recette" que Optuna va tester (L'espace de recherche)
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),  # Régularisation L1
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10), # Régularisation L2
        'n_jobs': -1,
        'random_state': 42
    }
    
    # 3. Création du Modèle avec ces paramètres
    model = XGBClassifier(**param)
    
    # 4. Validation Croisée (TimeSeriesSplit pour respecter le temps)
    # On coupe en 5 morceaux chronologiques pour valider la robustesse
    tscv = TimeSeriesSplit(n_splits=5)
    
    # On cherche à maximiser la PRÉCISION (Éviter les faux positifs)
    # score = cross_val_score(model, X, y, cv=tscv, scoring='precision').mean()
    
    # Alternative : Accuracy si tu veux juste avoir raison souvent
    scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
    return scores.mean()

def run_optimization():
    print(f" Démarrage de l'optimisation pour {TARGET_SYMBOL}...")
    
    # On crée l'étude
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS)
    
    print("\n MEILLEURS PARAMÈTRES TROUVÉS :")
    print("------------------------------------------------")
    print(study.best_params)
    print("------------------------------------------------")
    print(f"Meilleur Score (Accuracy Moyenne) : {study.best_value:.2%}")
    
    # Sauvegarde dans un fichier texte pour que tu puisses les copier
    with open("data/best_hyperparameters.txt", "w") as f:
        f.write(str(study.best_params))

if __name__ == "__main__":
    run_optimization()
