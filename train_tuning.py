import os
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
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
    """ Feature Engineering (Mémoire + Volatilité) """
    df = df.copy()
    
    # On évite les erreurs de calcul sur des séries trop courtes
    if len(df) < 50: return df

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

def train_tuning():
    print("\n🔧 DÉMARRAGE : OPTIMISATION COMPLÈTE (TOUTES LES ENTREPRISES)...")
    
    if not os.path.exists(FILE_PATH):
        print(f" Fichier introuvable : {FILE_PATH}")
        return
    
    df_all = pd.read_csv(FILE_PATH)
    
    # ON PREND TOUT LE MONDE !
    symbols = df_all["symbol"].unique()
    print(f"Analyse en cours sur {len(symbols)} actifs... (Prends un café ☕)")
    
    # Grille de recherche
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4, 8],
        'bootstrap': [True, False]
    }
    
    # Pour stocker les meilleurs paramètres trouvés
    best_configs = []

    print(f"\n{'SYMBOLE':<8} | {'BASE':<8} | {'OPTIMISÉ':<8} | {'GAIN':<8} | {'MEILLEURS PARAMS'}")
    print("-" * 110)

    for sym in symbols:
        df = df_all[df_all["symbol"] == sym].copy()
        
        # Tri et Prépa
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            
        try:
            df = add_advanced_features(df)
        except Exception:
            continue
            
        df = df.dropna().reset_index(drop=True)
        
        # On ignore les actions avec trop peu d'historique (< 2 ans env.)
        if len(df) < 500: 
            continue

        # Split 80/20
        split_idx = int(len(df) * 0.8)
        
        X = df[FEATURES]
        y = (df["Return_next"] > 0).astype(int)
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # 1. Modèle de Base (Celui qu'on utilise depuis le début)
        base_rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_depth=15, random_state=42, n_jobs=-1)
        base_rf.fit(X_train, y_train)
        base_acc = accuracy_score(y_test, base_rf.predict(X_test))
        
        # 2. Recherche (Tuning)
        tscv = TimeSeriesSplit(n_splits=3)
        rf_tune = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        random_search = RandomizedSearchCV(
            estimator=rf_tune,
            param_distributions=param_dist,
            n_iter=20,  # 20 tests aléatoires par action
            cv=tscv,
            verbose=0,
            random_state=42,
            n_jobs=-1
        )
        
        random_search.fit(X_train, y_train)
        
        best_acc = accuracy_score(y_test, random_search.best_estimator_.predict(X_test))
        gain = (best_acc - base_acc) * 100
        
        # Formatage pour affichage
        params = random_search.best_params_
        params_str = str(params).replace("min_samples_", "min_").replace("n_estimators", "n_est").replace("'", "")
        
        print(f"{sym:<8} | {base_acc*100:.1f}%    | {best_acc*100:.1f}%     | {gain:+.1f} pts  | {params_str[:60]}...")
        
        best_configs.append({
            "Symbol": sym,
            "Best_Acc": best_acc,
            "Gain": gain,
            "Params": params
        })

    # Petit bilan à la fin
    if best_configs:
        avg_gain = np.mean([x["Gain"] for x in best_configs])
        print("-" * 110)
        print(f"GAIN MOYEN GLOBAL GRÂCE AU TUNING : {avg_gain:+.2f} points de précision")
        
        # Sauvegarde des meilleurs paramètres dans un CSV pour pouvoir les réutiliser plus tard
        pd.DataFrame(best_configs).to_csv(os.path.join(DATA_FOLDER, "best_hyperparameters.csv"), index=False)
        print(f"Configuration optimale sauvegardée dans : data/best_hyperparameters.csv")

if __name__ == "__main__":
    train_tuning()
