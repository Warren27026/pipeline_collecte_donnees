import os
import pandas as pd
import ta
from datetime import datetime

# --- CONFIGURATION ---
DATA_FOLDER = "data"
# Fichiers générés par tes modèles IA
HYBRID_PATH = os.path.join(DATA_FOLDER, "final_hybrid_predictions.csv")
PRICE_PATH = os.path.join(DATA_FOLDER, "final_price_predictions.csv")
FEATURES_PATH = os.path.join(DATA_FOLDER, "ALL_YFINANCE_features.csv")

def add_display_indicators(df):
    """
    On garde cette fonction pour l'affichage (l'interface aime bien afficher le RSI actuel).
    Mais ce n'est plus utilisé pour la décision.
    """
    df = df.copy()
    close = df["Close"]
    
    # RSI & BB pour le contexte visuel
    df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["upper_bb"] = bb.bollinger_hband()
    df["lower_bb"] = bb.bollinger_lband()
    
    # Position BB
    df["bb_position"] = "INSIDE"
    df.loc[close > df["upper_bb"], "bb_position"] = "ABOVE"
    df.loc[close < df["lower_bb"], "bb_position"] = "BELOW"
    
    # MACD Hist
    macd = ta.trend.MACD(close)
    df["macd_hist"] = macd.macd_diff()
    
    return df

def generate_signals():
    print("🔄 FUSION : Intégration des modèles IA dans le flux Interface...")

    # 1. Charger les données de marché (pour le prix actuel et les indicateurs visuels)
    if not os.path.exists(FEATURES_PATH):
        print(" Erreur : Données de marché introuvables.")
        return
    
    df_all = pd.read_csv(FEATURES_PATH)
    if "date" in df_all.columns:
        df_all["date"] = pd.to_datetime(df_all["date"])

    # 2. Charger les cerveaux (IA Hybride + IA Prix)
    if os.path.exists(HYBRID_PATH):
        df_hybrid = pd.read_csv(HYBRID_PATH)
    else:
        print(" Pas de prédictions Hybrides trouvées. Lance 'predict_final_hybrid.py'.")
        df_hybrid = pd.DataFrame(columns=["Symbol", "Signal", "Reliability"])

    if os.path.exists(PRICE_PATH):
        df_price = pd.read_csv(PRICE_PATH)
    else:
        print(" Pas de prédictions de Prix trouvées. Lance 'train_price_predictor_v2.py'.")
        df_price = pd.DataFrame(columns=["Symbol", "Predicted_Price"])

    final_signals = []

    # 3. Boucle de fusion par symbole
    for symbol in df_all["symbol"].unique():
        # Données techniques récentes
        df_sym = df_all[df_all["symbol"] == symbol].copy().sort_values("date")
        if len(df_sym) < 60: continue
        
        # On calcule les indicateurs juste pour l'affichage (Joli pour l'interface)
        df_sym = add_display_indicators(df_sym)
        last_row = df_sym.iloc[-1]
        
        # --- RÉCUPÉRATION DE L'INTELLIGENCE (IA) ---
        
        # A. Signal Achat/Vente (Hybride)
        ia_signal = "NEUTRE"
        ia_confidence = 0.0
        
        row_hybrid = df_hybrid[df_hybrid["Symbol"] == symbol]
        if not row_hybrid.empty:
            raw_signal = row_hybrid.iloc[0]["Signal"] # Ex: "ACHAT "
            ia_confidence = row_hybrid.iloc[0]["Reliability"]
            
            # Traduction pour l'Interface (Standardisation)
            if "ACHAT" in raw_signal: ia_signal = "BUY"
            elif "VENTE" in raw_signal: ia_signal = "SELL"
            elif "IGNORER" in raw_signal: ia_signal = "WAIT" # Trop risqué
            else: ia_signal = "HOLD"
        
        # B. Objectif de Prix (XGBoost)
        target_price = 0.0
        row_price = df_price[df_price["Symbol"] == symbol]
        if not row_price.empty:
            target_price = row_price.iloc[0]["Predicted_Price"]
            
        # 4. Construction de la ligne finale pour l'interface
        final_signals.append({
            "symbol": symbol,
            "date": last_row["date"].strftime("%Y-%m-%d"),
            "close": round(last_row["Close"], 2),
            "rsi": round(last_row["rsi"], 2),
            "bb_position": last_row["bb_position"],
            "macd_hist": round(last_row["macd_hist"], 4),
            "recommendation": ia_signal,   # <--- C'est ici que l'IA prend le pouvoir
            "confidence": f"{ia_confidence*100:.0f}%", # Nouvelle info pour l'interface
            "target_price": round(target_price, 2)      # Nouvelle info pour l'interface
        })

    # 5. Sauvegarde
    signals_df = pd.DataFrame(final_signals)
    
    # Fichier "Live" pour l'interface
    live_path = os.path.join(DATA_FOLDER, "latest_signals.csv")
    signals_df.to_csv(live_path, index=False)
    print(f" Fichier interface généré : {live_path}")
    
    # Historique (Optionnel)
    history_path = os.path.join(DATA_FOLDER, "signals_history.csv")
    if os.path.exists(history_path):
        pd.concat([pd.read_csv(history_path), signals_df], ignore_index=True).to_csv(history_path, index=False)
    else:
        signals_df.to_csv(history_path, index=False)

    # Petit aperçu
    print("\n=== APERÇU DES SIGNAUX INTELLIGENTS ===")
    print(signals_df[["symbol", "close", "recommendation", "confidence", "target_price"]].head(10))

# --- LA FONCTION MANQUANTE ---
def main():
    generate_signals()

if __name__ == "__main__":
    main()
