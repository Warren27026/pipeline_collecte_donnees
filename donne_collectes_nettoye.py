# -*- coding: utf-8 -*-
"""
PIPELINE PRIX avec CSV + INDICATEURS TECHNIQUES (Bollinger, RSI, MACD…)
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
from tiingo import TiingoClient
import ta   # <--- LIBRAIRIE D'INDICATEURS
# pip install ta
from signals import main as signals_main

# ====================== CONFIG ======================

TIINGO_API_KEY = os.getenv('TIINGO_API_KEY')
if not TIINGO_API_KEY:
    raise ValueError("TIINGO_API_KEY manquante !")

DATA_FOLDER = "data"
TIINGO_FOLDER = os.path.join(DATA_FOLDER, "tiingo")

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(TIINGO_FOLDER, exist_ok=True)

# ====================== CLEAN DATA ======================

def clean_data(df):
    df = df.copy()

    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df['date'].dt.normalize()

    df = df.dropna()
    df = df[df['High'] >= df['Low']]
    df = df[df['Volume'] >= 0]
    df = df.sort_values('date').drop_duplicates('date')

    Q1, Q3 = df['Close'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df['Close'] = np.clip(df['Close'], Q1 - 3 * IQR, Q3 + 3 * IQR)

    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    return df


# ====================== ADD TECHNIC INDICATORS ======================

def add_indicators(df):
    df = df.copy()
    close = df["Close"]

    # --- BOLLINGER ---
    boll = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["Bollinger_Middle"] = boll.bollinger_mavg()
    df["Bollinger_Upper"] = boll.bollinger_hband()
    df["Bollinger_Lower"] = boll.bollinger_lband()
    df["Bollinger_%B"] = (close - df["Bollinger_Lower"]) / (
        df["Bollinger_Upper"] - df["Bollinger_Lower"]
    )
    df["Bollinger_Width"] = df["Bollinger_Upper"] - df["Bollinger_Lower"]

    # --- RSI ---
    df["RSI_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    # --- MACD ---
    macd = ta.trend.MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Diff"] = macd.macd_diff()

    # --- Returns ---
    df["Return"] = df["Close"].pct_change()
    df["Return_next"] = df["Return"].shift(-1)
    df["Close_next"] = df["Close"].shift(-1)

    df = df.dropna().reset_index(drop=True)
    return df


# ====================== YFINANCE ======================

def collect_yfinance():
    symbols = ["AAPL", "TSLA", "MSFT", "BTC-USD", "GOOGL"]
    all_data = []
    print("Collecte yfinance...")

    for s in symbols:
        df = yf.Ticker(s).history(period="1y").reset_index()
        df['symbol'] = s
        df['date'] = df['Date']
        df = df[['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'symbol']]

        df = clean_data(df)
        df_features = add_indicators(df)

        df.to_csv(os.path.join(DATA_FOLDER, f"{s}.csv"), index=False)
        df_features.to_csv(os.path.join(DATA_FOLDER, f"{s}_features.csv"), index=False)

        all_data.append(df_features)

    pd.concat(all_data).to_csv(os.path.join(DATA_FOLDER, "ALL_YFINANCE_features.csv"), index=False)
    print("yfinance OK")

# ====================== TIINGO ======================

def collect_tiingo():
    client = TiingoClient({'api_key': TIINGO_API_KEY, 'session': True})
    symbols = ["AAPL", "TSLA", "MSFT", "GOOGL"]
    all_data = []
    print("Collecte Tiingo...")

    start_date = datetime.now().replace(year=datetime.now().year - 1)

    for s in symbols:
        df = client.get_dataframe(s, frequency='daily', startDate=start_date)
        df = df.reset_index()
        df["symbol"] = s
        df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'symbol']]
        df.columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'symbol']

        df = clean_data(df)
        df_features = add_indicators(df)

        df_features.to_csv(os.path.join(TIINGO_FOLDER, f"{s}_features.csv"), index=False)
        all_data.append(df_features)

    pd.concat(all_data).to_csv(os.path.join(TIINGO_FOLDER, "ALL_TIINGO_features.csv"), index=False)
    print("Tiingo OK")

# ====================== MAIN ======================

def main():
    print("DÉBUT PIPELINE -", datetime.now().strftime("%Y-%m-%d %H:%M"))
    collect_yfinance()
    collect_tiingo()
    print("Données collectées et indicateurs calculés.")

    print("\nCalcul des signaux BUY/SELL...")
    signals_main()
    print("TERMINÉ – PRIX + SIGNAUX GÉNÉRÉS")

if __name__ == "__main__":
    main()
