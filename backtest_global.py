# backtest_global.py

import os
import pandas as pd

DATA_FOLDER = "data"

# 5 actifs du pipeline
SYMBOLS = [
    "AAPL",
    "TSLA",
    "MSFT",
    "GOOGL",
    "BTC-USD", 
]

CAPITAL_INITIAL = 25000.0  # capital global unique


def load_symbol_history(symbol: str, history_path: str):
    """Charge l'historique des signaux pour un symbole à partir de signals_history.csv"""
    df = pd.read_csv(history_path)

    # On garde uniquement ce symbole
    df = df[df["symbol"] == symbol].copy()
    if df.empty:
        print(f"[WARN] Aucun signal historique pour {symbol}, ignoré.")
        return None

    # Date propre
    df["date"] = pd.to_datetime(df["date"])

    # On enlève les doublons éventuels (même jour, même symbole)
    df = df.sort_values("date").drop_duplicates(subset=["symbol", "date"], keep="last")

    # On renomme pour rester cohérent avec le reste du code
    df = df.rename(columns={
        "recommendation": "signal",
        "close": "Close"
    })

    return df[["date", "Close", "signal"]]


def backtest_global():
    history_path = os.path.join(DATA_FOLDER, "signals_history.csv")
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"Fichier introuvable : {history_path} (lance d'abord signals.py plusieurs jours).")

    datasets = {}
    for sym in SYMBOLS:
        df_sym = load_symbol_history(sym, history_path)
        if df_sym is not None:
            datasets[sym] = df_sym

    if not datasets:
        raise ValueError("Aucune donnée de signaux disponible pour le backtest global.")

    # Dates communes à tous les symboles retenus
    common_dates = None
    for df in datasets.values():
        if common_dates is None:
            common_dates = set(df["date"])
        else:
            common_dates = common_dates.intersection(set(df["date"]))

    common_dates = sorted(list(common_dates))
    if not common_dates:
        raise ValueError("Aucune date commune entre les symboles (trop peu d'historique).")

    # Portefeuille global
    cash = CAPITAL_INITIAL
    positions = {sym: 0.0 for sym in datasets.keys()}  # nb d'actions détenues par symbole
    portfolio_values = []

    for date in common_dates:
        # 1) SELL d'abord
        for sym, df in datasets.items():
            row = df[df["date"] == date].iloc[0]
            signal = row["signal"]
            price = float(row["Close"])

            if signal == "SELL" and positions[sym] > 0:
                cash += positions[sym] * price
                positions[sym] = 0.0

        # 2) BUY : on répartit le cash entre tous les symboles en BUY
        buy_symbols = []
        for sym, df in datasets.items():
            row = df[df["date"] == date].iloc[0]
            if row["signal"] == "BUY":
                buy_symbols.append(sym)

        if buy_symbols:
            amount_per_symbol = cash / len(buy_symbols)
            for sym in buy_symbols:
                row = datasets[sym][datasets[sym]["date"] == date].iloc[0]
                price = float(row["Close"])
                shares = amount_per_symbol / price
                positions[sym] += shares
                cash -= shares * price

        # 3) Valeur totale du portefeuille à cette date
        total_value = cash
        for sym, df in datasets.items():
            row = df[df["date"] == date].iloc[0]
            price = float(row["Close"])
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

    out_path = os.path.join(DATA_FOLDER, "backtest_global.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nDétails sauvegardés dans : {out_path}")

    return result_df


if __name__ == "__main__":
    backtest_global()
