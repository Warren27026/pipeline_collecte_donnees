# signals.py
import os
import pandas as pd

DATA_FOLDER = "data"

def add_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute une colonne 'signal' (BUY / SELL / HOLD) à partir des indicateurs :
    - Close
    - Bollinger_Lower
    - Bollinger_Upper
    - RSI_14
    - MACD_Diff
    """
    df = df.copy()
    df["signal"] = "HOLD"  # par défaut

    # Condition d'achat (BUY)
    buy_cond = (
        (df["Close"] < df["Bollinger_Lower"]) &
        (df["RSI_14"] < 30) &
        (df["MACD_Diff"] > 0)
    )

    # Condition de vente (SELL)
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
    Lis data/{symbol}_features.csv,
    calcule les signaux, et renvoie le signal du DERNIER jour.
    """
    path = os.path.join(DATA_FOLDER, f"{symbol}_features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    df = pd.read_csv(path)

    # Ajout des signaux
    df = add_signals(df)

    # Dernière ligne (jour le plus récent)
    last = df.iloc[-1]

    date = last.get("date", "N/A")
    close = float(last["Close"])
    signal = last["signal"]

    return {
        "symbol": symbol,
        "date": date,
        "close": close,
        "signal": signal,
    }


def main():
    symbols = ["AAPL", "TSLA", "MSFT", "BTC-USD", "GOOGL"]

    print("=== SIGNAUX DU JOUR ===")
    for s in symbols:
        try:
            info = get_last_signal_for_symbol(s)
            print(f"{info['date']} | {info['symbol']} | Close={info['close']:.2f} | Signal={info['signal']}")
        except Exception as e:
            print(f"[ERREUR] {s} : {e}")


if __name__ == "__main__":
    main()
