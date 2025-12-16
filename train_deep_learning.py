import os
import numpy as np
import pandas as pd
import ta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# --- CONFIGURATION ---
DATA_FOLDER = "data"
FILE_PATH = os.path.join(DATA_FOLDER, "ALL_YFINANCE_features.csv")
SEQ_LEN = 60  # Le modèle regarde les 60 derniers jours pour prédire le suivant

# On garde les indicateurs techniques comme features d'entrée
FEATURES = [
    "Close", "RSI_14", "MACD", "Bollinger_%B", 
    "Return", "Volume", "Dist_SMA_50", "ATR_14"
]

def add_advanced_features(df):
    """ Feature Engineering identique aux étapes précédentes """
    df = df.copy()
    if len(df) < 50: return df
    
    close = df["Close"]
    df["Return"] = close.pct_change()
    
    # BB
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["Bollinger_%B"] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
    
    # RSI / MACD
    df["RSI_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd = ta.trend.MACD(close)
    df["MACD"] = macd.macd()
    
    # Advanced
    sma_50 = ta.trend.SMAIndicator(close, window=50).sma_indicator()
    df["Dist_SMA_50"] = (close - sma_50) / sma_50
    atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], close, window=14)
    df["ATR_14"] = atr.average_true_range()
    
    return df

def create_sequences(data, seq_len):
    """ Transforme les données en séquences 3D pour le LSTM """
    xs, ys = [], []
    for i in range(len(data) - seq_len - 1):
        x = data[i:(i + seq_len)]
        # On prédit la 1ère colonne (qui correspondra au 'Close' scalé)
        y = data[i + seq_len, 0] 
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_deep_learning():
    print("\n DÉMARRAGE : DEEP LEARNING (LSTM) SUR TOUS LES SYMBOLES...")
    print(f"Objectif : Prédire la VALEUR exacte (Prix) en analysant des séquences de {SEQ_LEN} jours.\n")
    
    if not os.path.exists(FILE_PATH):
        print(f"❌ Fichier introuvable : {FILE_PATH}")
        return
    
    df_all = pd.read_csv(FILE_PATH)
    symbols = df_all["symbol"].unique()
    
    print(f"{'SYMBOLE':<8} | {'RMSE NAÏF ($)':<15} | {'RMSE LSTM ($)':<15} | {'PERFORMANCE'}")
    print("-" * 70)

    for sym in symbols:
        # 1. Préparation Data
        df = df_all[df_all["symbol"] == sym].copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            
        try:
            df = add_advanced_features(df)
        except:
            continue
            
        df = df.dropna().reset_index(drop=True)
        # Le Deep Learning a besoin de beaucoup de données
        if len(df) < 700: continue 
            
        # On prépare le dataset
        # On met le 'Close' en premier pour faciliter la création de la cible
        data_cols = ["Close"] + [col for col in FEATURES if col != "Close"]
        dataset_raw = df[data_cols].values
        
        # Scaling (0 à 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset_raw)
        
        # Création des séquences
        X, y = create_sequences(scaled_data, SEQ_LEN)
        
        if len(X) < 100: continue

        # Split Train/Test (80/20) sans mélange (séries temporelles)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # 2. Architecture du Modèle LSTM
        model = Sequential([
            # Couche 1 : Retourne une séquence pour la couche suivante
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2), # Évite le par cœur
            # Couche 2 : Retourne juste le vecteur final
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1) # Sortie : Le prix scalé prédit
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        
        # Entraînement (Rapide : 5 epochs pour la démo, mettre 20+ pour la prod)
        model.fit(X_train, y_train, batch_size=32, epochs=5, verbose=0)
        
        # 3. Prédictions & Reconstruction
        predictions_scaled = model.predict(X_test, verbose=0)
        
        # On doit inverser le scaling pour avoir des vrais Dollars
        # On crée une matrice vide avec la même forme que 'scaled_data' pour utiliser inverse_transform
        dummy_matrix = np.zeros((len(predictions_scaled), scaled_data.shape[1]))
        dummy_matrix[:, 0] = predictions_scaled.flatten()
        predictions = scaler.inverse_transform(dummy_matrix)[:, 0]
        
        # Idem pour les vraies valeurs (y_test)
        dummy_matrix_y = np.zeros((len(y_test), scaled_data.shape[1]))
        dummy_matrix_y[:, 0] = y_test.flatten()
        real_values = scaler.inverse_transform(dummy_matrix_y)[:, 0]
        
        # 4. Évaluation (RMSE)
        rmse_lstm = np.sqrt(mean_squared_error(real_values, predictions))
        
        # Baseline Naïve : Le prix prédit est juste le prix de la veille
        # Attention aux indices : on compare Real[t] avec Real[t-1]
        rmse_naive = np.sqrt(mean_squared_error(real_values[1:], real_values[:-1]))
        
        improvement = rmse_naive - rmse_lstm
        status = "SUCCÈS" if improvement > 0 else " ÉCHEC"
        
        print(f"{sym:<8} | {rmse_naive:<15.2f} | {rmse_lstm:<15.2f} | {status}")
        
        # Sauvegarde graphique pour le rapport (Si succès)
        if status == "SUCCÈS" and improvement > (rmse_naive * 0.05): # Si 5% mieux
            plt.figure(figsize=(12,6))
            plt.plot(real_values, label="Prix Réel", color="black")
            plt.plot(predictions, label="Prédiction LSTM", color="#00cc66")
            plt.title(f"Deep Learning {sym} : Réalité vs Prédiction")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(DATA_FOLDER, f"lstm_{sym}.png"))
            plt.close()

if __name__ == "__main__":
    train_deep_learning()
