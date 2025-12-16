import os
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
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
    
    # Sécurité pour éviter les erreurs sur les séries trop courtes
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

def train_ensemble():
    print("\n DÉMARRAGE : MODÈLE ENSEMBLE COMPLET (TOUT LE PORTEFEUILLE)...")
    
    if not os.path.exists(FILE_PATH):
        print(f" Fichier introuvable : {FILE_PATH}")
        return
    
    df_all = pd.read_csv(FILE_PATH)
    
    # ON RÉCUPÈRE TOUTES LES ENTREPRISES
    symbols = df_all["symbol"].unique()
    print(f"Analyse en cours sur {len(symbols)} actifs...")
    
    print(f"\n{'SYMBOLE':<8} | {'RF SEUL':<10} | {'GB SEUL':<10} | {'ENSEMBLE':<10} | {'GAIN':<8}")
    print("-" * 70)
    
    results = []
    wins = 0

    for sym in symbols:
        df = df_all[df_all["symbol"] == sym].copy()
        
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            
        try:
            df = add_advanced_features(df)
        except Exception:
            continue
            
        df = df.dropna().reset_index(drop=True)
        # On ignore les historiques trop courts pour que le test soit fiable
        if len(df) < 500: continue

        # Split 80/20
        split_idx = int(len(df) * 0.8)
        
        X = df[FEATURES]
        y = (df["Return_next"] > 0).astype(int)
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # --- 1. RANDOM FOREST (Base solide) ---
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_depth=15, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        acc_rf = accuracy_score(y_test, rf.predict(X_test))
        
        # --- 2. GRADIENT BOOSTING (Le spécialiste) ---
        gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
        gb.fit(X_train, y_train)
        acc_gb = accuracy_score(y_test, gb.predict(X_test))
        
        # --- 3. VOTING CLASSIFIER (Le Consensus) ---
        # 'soft' voting = moyenne des probabilités (plus nuancé que 'hard' voting)
        ensemble = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')
        ensemble.fit(X_train, y_train)
        acc_ens = accuracy_score(y_test, ensemble.predict(X_test))
        
        # Calcul du gain par rapport au MEILLEUR des deux modèles seuls
        best_single = max(acc_rf, acc_gb)
        gain = (acc_ens - best_single) * 100
        
        if acc_ens > best_single:
            wins += 1
        
        print(f"{sym:<8} | {acc_rf*100:.1f}%     | {acc_gb*100:.1f}%     | {acc_ens*100:.1f}%     | {gain:+.1f} pts")
        
        results.append(acc_ens)

    if results:
        print("-" * 70)
        avg_ens = np.mean(results) * 100
        print(f"MOYENNE GLOBALE ENSEMBLE : {avg_ens:.2f}%")
        print(f"L'Ensemble a battu les modèles solos sur {wins}/{len(results)} actifs.")
        
        # Sauvegarde des résultats
        pd.DataFrame({'Symbol': symbols[:len(results)], 'Ensemble_Acc': results}).to_csv(os.path.join(DATA_FOLDER, "ensemble_results.csv"), index=False)
        print("Résultats sauvegardés dans data/ensemble_results.csv")

if __name__ == "__main__":
    train_ensemble()
