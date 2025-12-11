# backtest_global.py

import os
import pandas as pd

DATA_FOLDER = "data"

SYMBOLS = [
    "AAPL",
    "TSLA",
    "MSFT",
    "GOOGL",
    "BTC-USD",  # tu peux l'enlever si tu veux seulement les actions
]

CAPITAL_INITIAL = 25000.0  # capital global unique


def load_symbol_history(symbol: str, history_path: str):
    """Charge l'historique des signaux pour un symbole à partir de signals_history.csv"""
    df = pd.read_csv(history_path)

    df = df[df["symbol"] == symbol].copy()
    if df.empty:
        print(f"[WARN] Aucun signal historique pour {symbol}, ignoré.")
        return None

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["symbol", "date"], keep="last")

    df = df.rename(columns={
        "recommendation": "signal",
        "close": "Close"
    })

    return df[["date", "Close", "signal"]]


def backtest_global():
    history_path = os.path.join(DATA_FOLDER, "signals_history.csv")
    if not os.path.exists(history_path):
        raise FileNotFoundError(
            f"Fichier introuvable : {history_path} (lance d'abord donne_collectes_nettoye.py / signals.py)."
        )

    datasets = {}
    for sym in SYMBOLS:
        df_sym = load_symbol_history(sym, history_path)
        if df_sym is not None:
            datasets[sym] = df_sym

    if not datasets:
        raise ValueError("Aucune donnée de signaux disponible pour le backtest global.")

    # ⚠️ AU LIEU D'UNE INTERSECTION STRICTE, ON PREND LA UNION DES DATES
    all_dates = set()
    for df in datasets.values():
        all_dates |= set(df["date"])
    all_dates = sorted(list(all_dates))

    if not all_dates:
        raise ValueError("Aucune date disponible dans l'historique des signaux.")

    # Portefeuille global
    cash = CAPITAL_INITIAL
    positions = {sym: 0.0 for sym in datasets.keys()}  # nb d'actions
    last_price = {sym: None for sym in datasets.keys()}  # dernier prix connu
    portfolio_rows = []

    for date in all_dates:
        # 1) SELL d'abord
        for sym, df in datasets.items():
            rows = df[df["date"] == date]
            if not rows.empty:
                row = rows.iloc[0]
                price = float(row["Close"])
                last_price[sym] = price
                signal = row["signal"]
            else:
                # Pas de nouvelle donnée ce jour-là pour ce symbole
                signal = "HOLD"
                price = last_price[sym]

            # Si signal SELL et on a des actions → on vend
            if signal == "SELL" and positions[sym] > 0 and price is not None:
                cash += positions[sym] * price
                positions[sym] = 0.0

        # 2) BUY : on répartit le cash entre tous les symboles qui ont BUY ce jour-là
        buy_symbols = []
        for sym, df in datasets.items():
            rows = df[df["date"] == date]
            if not rows.empty:
                row = rows.iloc[0]
                signal = row["signal"]
                price = float(row["Close"])
                last_price[sym] = price
            else:
                signal = "HOLD"
                price = last_price[sym]

            if signal == "BUY" and price is not None:
                buy_symbols.append(sym)

        if buy_symbols and cash > 0:
            amount_per_symbol = cash / len(buy_symbols)
            for sym in buy_symbols:
                price = last_price[sym]
                if price is None:
                    continue
                shares = amount_per_symbol / price
                positions[sym] += shares
                cash -= shares * price

        # 3) Calcule la valeur totale du portefeuille à cette date
        total_value = cash
        for sym in datasets.keys():
            price = last_price[sym]
            if price is not None and positions[sym] > 0:
                total_value += positions[sym] * price

        portfolio_rows.append({
            "date": date,
            "total_value": total_value,
            "cash": cash,
            **{f"pos_{sym}": positions[sym] for sym in datasets.keys()}
        })

    result_df = pd.DataFrame(portfolio_rows)
    final_value = result_df.iloc[-1]["total_value"]
    perf_pct = (final_value / CAPITAL_INITIAL - 1) * 100

    print("\n=== RÉSULTATS BACKTEST GLOBAL ===")
    print(f"Capital initial : {CAPITAL_INITIAL:.2f} €")
    print(f"Valeur finale : {final_value:.2f} €")
    print(f"Performance : {perf_pct:.2f} %")

    out_path = os.path.join(DATA_FOLDER, "backtest_global.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nDétails sauvegardés dans : {out_path}")

    return result_df


if __name__ == "__main__":
    backtest_global()
