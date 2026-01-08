import os
import pandas as pd
import numpy as np
import ta
import xgboost as xgb
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
DATA_FOLDER = "data"
FILE_PATH = os.path.join(DATA_FOLDER, "ALL_YFINANCE_features.csv")

# Seuil de sécurité : Si le meilleur modèle fait moins de 55% de réussite, on ne trade pas l'actif.
MIN_ACCURACY_THRESHOLD = 0.55 
# Seuil de confiance Sniper : On n'achète que si la proba est > 60%
SNIPER_THRESHOLD = 0.60

FEATURES = [
    "RSI_14", "MACD", "MACD_Signal", "MACD_Diff", 
    "Bollinger_Width", "Bollinger_%B",
    "Return", "Volume",
    "Return_D-1", "Return_D-2", "RSI_D-1",
    "Dist_SMA_50", "ATR_14", "Day_Of_Week"
]

def add_advanced_features(df):
    """ Feature Engineering complet """
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
        df["Day_Of_Week"] = datetime.now().weekday()
        
    return df

def get_best_model(X_train, y_train, X_test, y_test):
    """ Entraîne RF et XGB et retourne le meilleur des deux """
    
    # 1. Random Forest
    rf = RandomForestClassifier(n_estimators=367, min_samples_leaf=4, max_depth=12,min_samples_split=9, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    """# 1. Random Forest (Optimisé V3 : 55.75% Accuracy / +4.44% Gain vs Hasard)
    rf = RandomForestClassifier(
        # Le nombre d'arbres de décision dans la forêt.
        # 367 arbres permettent de "lisser" les erreurs individuelles et de stabiliser la prédiction.
        n_estimators=367,
        
        # La profondeur maximale de chaque arbre.
        # 12 est un juste milieu : assez profond pour capter des motifs complexes, 
        # mais limité pour éviter d'apprendre le bruit par cœur (Overfitting).
        max_depth=12,
        
        # Le nombre minimum d'échantillons requis pour diviser un nœud.
        # À 9, on force l'arbre à ne prendre des décisions que sur des groupes de données 
        # significatifs, évitant de créer des branches pour des cas trop isolés.
        min_samples_split=9,
        
        # Le nombre minimum d'échantillons requis pour être une feuille (décision finale).
        # À 4, on interdit au modèle de conclure sur la base de moins de 4 exemples historiques.
        # C'est un filtre de sécurité puissant contre la volatilité des marchés.
        min_samples_leaf=4,
        
        # La méthode pour choisir le nombre d'indicateurs à tester à chaque embranchement.
        # 'log2' force chaque arbre à regarder un sous-ensemble différent d'indicateurs,
        # ce qui rend la forêt plus diversifiée et robuste.
        max_features='log2',
        
        # Assure que les résultats sont reproductibles (toujours les mêmes à chaque lancement).
        random_state=42,
        
        # Utilise tous les cœurs du processeur pour l'entraînement (Vitesse maximale).
        n_jobs=-1
    )"""
    
    # On évalue en mode "Sniper" (sur les trades > 60% de confiance seulement)
    probs_rf = rf.predict_proba(X_test)[:, 1]
    mask_rf = probs_rf > SNIPER_THRESHOLD
    score_rf = 0.0
    if np.sum(mask_rf) > 2: # Il faut un minimum de trades pour juger
        score_rf = accuracy_score(y_test[mask_rf], (probs_rf[mask_rf] > 0.5).astype(int))
    
    # 2. XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=159, learning_rate=0.033,max_depth=7,subsample=0.53,colsample_bytree=0.64,gamma=2.08,random_state=42,n_jobs=-1,eval_metric="logloss")
    xgb_model.fit(X_train, y_train)
    
    probs_xgb = xgb_model.predict_proba(X_test)[:, 1]
    mask_xgb = probs_xgb > SNIPER_THRESHOLD
    score_xgb = 0.0
    if np.sum(mask_xgb) > 2:
        score_xgb = accuracy_score(y_test[mask_xgb], (probs_xgb[mask_xgb] > 0.5).astype(int))
        
    # Choix du vainqueur
    if score_xgb > score_rf:
        return xgb_model, "XGBoost", score_xgb
    else:
        return rf, "RandomForest", score_rf

def predict_final_hybrid():
    print("\n" + "="*80)
    print(f" SYSTÈME HYBRIDE : PRÉDICTION OPTIMISÉE (Tous les actifs)")
    print("="*80)
    
    if not os.path.exists(FILE_PATH):
        print("❌ Données manquantes.")
        return

    df_all = pd.read_csv(FILE_PATH)
    symbols = df_all["symbol"].unique()
    
    predictions = []
    
    print(f"{'ACTIF':<8} | {'PRIX':<8} | {'MODÈLE':<12} | {'FIABILITÉ':<10} | {'SIGNAL':<10} | {'ANALYSE'}")
    print("-" * 80)

    for sym in symbols:
        # --- PRÉPARATION ---
        df = df_all[df_all["symbol"] == sym].copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            
        try:
            df = add_advanced_features(df)
        except:
            continue
            
        df = df.dropna().reset_index(drop=True)
        if len(df) < 500: continue # Pas assez de données
            
        # Split Train (Passé lointain) / Test (Passé récent pour validation)
        # On garde la dernière ligne pour la VRAIE prédiction de demain
        real_prediction_row = df.iloc[[-1]][FEATURES]
        current_price = df.iloc[-1]["Close"]
        
        # Données pour choisir le meilleur modèle (Tout sauf la ligne d'aujourd'hui)
        work_df = df.iloc[:-1]
        
        split_idx = int(len(work_df) * 0.8)
        X = work_df[FEATURES]
        y = (work_df["Return_next"] > 0).astype(int)
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # --- SÉLECTION DU CHAMPION ---
        best_model, model_name, reliability = get_best_model(X_train, y_train, X_test, y_test)
        
        # --- PRÉDICTION POUR DEMAIN ---
        # On ré-entraîne le champion sur TOUT l'historique disponible pour max de fraîcheur
        best_model.fit(X, y)
        proba_up = best_model.predict_proba(real_prediction_row)[0, 1]
        
        # --- DÉCISION ---
        decision = "NEUTRE"
        comment = "Attente"
        
        # 1. Filtre de Sécurité : Si le modèle n'est pas fiable historiquement, on rejette
        if reliability < MIN_ACCURACY_THRESHOLD:
            decision = "IGNORER"
            comment = f"Modèle peu fiable ({reliability*100:.0f}%)"
        else:
            # 2. Filtre Sniper : Si le modèle est fiable, est-il sûr de lui aujourd'hui ?
            if proba_up >= SNIPER_THRESHOLD:
                decision = "ACHAT "
                comment = f"Signal Fort ({proba_up*100:.1f}%)"
            elif proba_up <= (1.0 - SNIPER_THRESHOLD):
                decision = "VENTE ❌"
                comment = f"Baisse probable ({(1-proba_up)*100:.1f}%)"
            else:
                decision = "NEUTRE"
                comment = "Incertitude"

        # Affichage conditionnel (on met en évidence les opportunités)
        if decision == "ACHAT ":
            prefix = "🔥 "
        elif decision == "IGNORER":
            prefix = "⚠️ "
        else:
            prefix = ""
            
        print(f"{prefix}{sym:<6} | {current_price:<8.2f} | {model_name:<12} | {reliability*100:.1f}%       | {decision:<10} | {comment}")
        
        predictions.append({
            "Date": datetime.now().strftime('%Y-%m-%d'),
            "Symbol": sym,
            "Best_Model": model_name,
            "Reliability": round(reliability, 2),
            "Proba_Hausse": round(proba_up, 4),
            "Signal": decision
        })

    # Sauvegarde
    out_path = os.path.join(DATA_FOLDER, "final_hybrid_predictions.csv")
    pd.DataFrame(predictions).to_csv(out_path, index=False)
    print("\n Analyse terminée. Résultats exportés.")

if __name__ == "__main__":
    predict_final_hybrid()
