# backtest_global.py

import os
import pandas as pd

DATA_FOLDER = "data"

# 5 entreprises de ton pipeline
SYMBOLS = [
    "AAPL",
    "TSLA",
    "MSFT",
    "GOOGL",
    "BTC-USD"   # Retire-le si tu veux rester 100% actions
]

CAPITAL_INITIAL = 25000.0   # Capital global unique


def load_data(symbol):
    path = os.path.join(DATA_FOLDER, f"{symbol}_features.csv")
    if not os.path.exists(path):
        print(f"[WARN] {symbol}_features.csv introuvable -> ignoré")
        return None

    df = pd.read_csv(path)

    if "signal" not in df.columns:
        print(f"[WARN] Pas de colonne 'signal' dans {symbol} -> ignoré")
        return None

    df["date"] = pd.to_datetime(df["date"])
    return df


def backtest_global():
    # Charger les données
    datasets = {}
    for symbol in SYMBOLS:
        df = load_data(symbol)
        if df is not None:
            datasets[symbol] = df

    if len(datasets) == 0:
        raise ValueError("Aucune donnée disponible pour le backtest global.")

    # Dates communes à toutes les entreprises
    common_dates = None
    for df in datasets.values():
        if common_dates is None:
            common_dates = set(df["date"])
        else:
            common_dates = common_dates.intersection(set(df["date"]))

    common_dates = sorted(list(common_dates))

    # Portefeuille global
    cash = CAPITAL_INITIAL
    positions = {sym: 0.0 for sym in SYMBOLS}
    portfolio_values = []

    for date in common_dates:

        # SELL en premier
        for sym, df in datasets.items():
            row = df[df["date"] == date].iloc[0]
            signal = row["signal"]
            price = row["Close"]

            if signal == "SELL" and positions[sym] > 0:
                cash += positions[sym] * price
                positions[sym] = 0.0

        # BUY ensuite
        buy_symbols = []
        for sym, df in datasets.items():
            row = df[df["date"] == date].iloc[0]
            if row["signal"] == "BUY":
                buy_symbols.append(sym)

        if buy_symbols:
            amount_per_symbol = cash / len(buy_symbols)

            for sym in buy_symbols:
                row = datasets[sym][datasets[sym]["date"] == date].iloc[0]
                price = row["Close"]
                shares = amount_per_symbol / price
                positions[sym] += shares
                cash -= shares * price

        # Valeur totale du portefeuille aujourd’hui
        total_value = cash
        for sym, df in datasets.items():
            row = df[df["date"] == date].iloc[0]
            price = row["Close"]
            total_value += positions[sym] * price

        portfolio_values.append({
            "date": date,
            "total_value": total_value,
            "cash": cash
        })

    result_df = pd.DataFrame(portfolio_values)
    final_value = result_df.iloc[-1]["total_value"]
    perf_pct = (final_value / CAPITAL_INITIAL - 1) * 100

    print("\n=== RÉSULTATS BACKTEST GLOBAL ===")
    print(f"Capital initial : {CAPITAL_INITIAL} €")
    print(f"Valeur finale : {final_value:.2f} €")
    print(f"Performance : {perf_pct:.2f} %")

    result_df.to_csv(os.path.join(DATA_FOLDER, "backtest_global.csv"), index=False)
    print("\nDonnées sauvegardées dans : data/backtest_global.csv")

    return result_df


if __name__ == "__main__":
    backtest_global()
