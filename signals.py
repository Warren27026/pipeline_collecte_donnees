# signals.py
import os
import pandas as pd

DATA_FOLDER = "data"

def add_signals(df):
    """Ajoute une colonne 'signal' (BUY / SELL / HOLD) à partir des indicateurs."""
    df = df.copy()
    df["signal"] = "HOLD"

    # Règle BUY
    buy_cond = (
        (df["Close"] < df["Bollinger_Lower"]) &
        (df["RSI_14"] < 30) &
        (df["MACD_Diff"] > 0)
    )

    # Règle SELL
    sell_cond = (
        (df["Close"] > df["Bollinger_Upper"]) &
        (df["RSI_14"] > 70) &
        (df["MACD_Diff"] < 0)
    )

    df.loc[buy_cond, "signal"] = "BUY"
    df.loc[sell_cond, "signal"] = "SELL"

    return df


def get_last_signal_for_symbol(symbol: str):
    """
    Lit le fichier {symbol}_features.csv, calcule le signal pour chaque ligne,
    et renvoie le signal du DERNIER JOUR disponible.
    """
    path = os.path.join(DATA_FOLDER, f"{symbol}_features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    df = pd.read_csv(path)

    # On ajoute les signaux
    df = add_signals(df)

    # On prend la dernière ligne (la plus récente)
    last_row = df.iloc[-1]

    date = last_row.get("date", "N/A")
    price = float(last_row["Close"])
    signal = last_row["signal"]

    return {
        "symbol": symbol,
        "date": date,
        "close": price,
        "signal": signal
    }


def main():
    symbols = ["AAPL", "TSLA", "MSFT", "BTC-USD", "GOOGL"]
    print("=== SIGNAUX DU JOUR ===")
    for s in symbols:
        try:
            info = get_last_signal_for_symbol(s)
            print(f"{info['date']} | {info['symbol']} | Close={info['close']:.2f} | Signal={info['signal']}")
        except Exception as e:
            print(f"Erreur pour {s}: {e}")


if __name__ == "__main__":
    main()
