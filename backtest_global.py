# backtest_global.py
import os
import pandas as pd

DATA_FOLDER = "data"

# 10 à 15 entreprises
SYMBOLS = [
    "AAPL", "TSLA", "MSFT", "AMZN", "META",
    "NVDA", "GOOGL", "NFLX", "IBM", "INTC",
    "ORCL", "AMD", "PYPL", "CSCO"
]

CAPITAL_INITIAL = 25000.0   # Capital global unique


def load_data(symbol):
    path = os.path.join(DATA_FOLDER, f"{symbol}_features.csv")
    if not os.path.exists(path):
        print(f"[WARN] {symbol}_features.csv introuvable")
        return None

    df = pd.read_csv(path)

    if "signal" not in df.columns:
        print(f"[WARN] Pas de colonne 'signal' dans {symbol}")
        return None

    df["date"] = pd.to_datetime(df["date"])
    return df


def backtest_global():
    # Charger les données de toutes les entreprises
    data_dict = {}
    for symbol in SYMBOLS:
        df = load_data(symbol)
        if df is not None:
            data_dict[symbol] = df

    # Synchroniser les dates communes
    common_dates = None
    for df in data_dict.values():
        if common_dates is None:
            common_dates = set(df["date"])
        else:
            common_dates = common_dates.intersection(set(df["date"]))

    common_dates = sorted(list(common_dates))

    # Portefeuille global
    cash = CAPITAL_INITIAL
    positions = {symbol: 0.0 for symbol in SYMBOLS}   # nombre d'actions par entreprise
    portfolio_values = []

    for date in common_dates:
        invested_companies = []

        # Première boucle : SELL et HOLD
        for symbol, df in data_dict.items():
            row = df[df["date"] == date].iloc[0]
            price = row["Close"]
            signal = row["signal"]

            # SELL → vendre tout
            if signal == "SELL" and positions[symbol] > 0:
                cash += positions[symbol] * price
                positions[symbol] = 0.0

            # HOLD → rien ne change

        # Deuxième boucle : BUY (après avoir vendu)
        for symbol, df in data_dict.items():
            row = df[df["date"] == date].iloc[0]
            price = row["Close"]
            signal = row["signal"]

            if signal == "BUY":
                invested_companies.append(symbol)

        # Répartition équitable du cash entre les BUY
        if invested_companies:
            amount_per_company = cash / len(invested_companies)

            for symbol in invested_companies:
                row = data_dict[symbol][data_dict[symbol]["date"] == date].iloc[0]
                price = row["Close"]

                shares = amount_per_company / price
                positions[symbol] += shares
                cash -= shares * price

        # Calcul de la valeur totale du portefeuille aujourd’hui
        total_value = cash
        for symbol, df in data_dict.items():
            row = df[df["date"] == date].iloc[0]
            price = row["Close"]
            total_value += positions[symbol] * price

        portfolio_values.append({
            "date": date,
            "total_value": total_value,
            "cash": cash
        })

    # Résultat final
    result_df = pd.DataFrame(portfolio_values)
    final_value = result_df.iloc[-1]["total_value"]
    perf_pct = (final_value / CAPITAL_INITIAL - 1) * 100

    print("\n=== RÉSULTATS DU BACKTEST GLOBAL ===")
    print(f"Capital initial : {CAPITAL_INITIAL} €")
    print(f"Valeur finale : {final_value:.2f} €")
    print(f"Performance : {perf_pct:.2f} %")

    out_path = os.path.join(DATA_FOLDER, "backtest_global.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nDétails enregistrés dans {out_path}")

    return result_df


if __name__ == "__main__":
    backtest_global()
