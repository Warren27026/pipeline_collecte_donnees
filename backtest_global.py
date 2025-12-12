# backtest_global.py

import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_FOLDER = "data"

SYMBOLS = [
    "AAPL",
    "TSLA",
    "MSFT",
    "GOOGL",
    "BTC-USD",
]

CAPITAL_INITIAL = 25000.0


def load_symbol_history(symbol: str, history_path: str):
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

    # Union des dates
    all_dates = set()
    for df in datasets.values():
        all_dates |= set(df["date"])
    all_dates = sorted(list(all_dates))

    if not all_dates:
        raise ValueError("Aucune date disponible dans l'historique des signaux.")

    cash = CAPITAL_INITIAL
    positions = {sym: 0.0 for sym in datasets.keys()}
    last_price = {sym: None for sym in datasets.keys()}
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
                signal = "HOLD"
                price = last_price[sym]

            if signal == "SELL" and positions[sym] > 0 and price is not None:
                cash += positions[sym] * price
                positions[sym] = 0.0

        # 2) BUY ensuite
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

        # 3) Valeur totale du portefeuille
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

    result_df = pd.DataFrame(portfolio_rows).sort_values("date").reset_index(drop=True)

    final_value = result_df.iloc[-1]["total_value"]
    perf_pct = (final_value / CAPITAL_INITIAL - 1) * 100

    # =========================
    # MÉTRIQUES DE RISQUE
    # =========================
    result_df["daily_return"] = result_df["total_value"].pct_change()

    # Drawdown
    result_df["peak"] = result_df["total_value"].cummax()
    result_df["drawdown"] = (result_df["total_value"] - result_df["peak"]) / result_df["peak"]
    max_drawdown = result_df["drawdown"].min()  # valeur négative

    # Volatilité
    volatility = result_df["daily_return"].std()

    # Sharpe (taux sans risque = 0)
    mean_return = result_df["daily_return"].mean()
    sharpe_ratio = (mean_return / volatility) if (volatility and volatility != 0) else 0.0

    print("\n=== RÉSULTATS BACKTEST GLOBAL ===")
    print(f"Capital initial : {CAPITAL_INITIAL:.2f} €")
    print(f"Valeur finale : {final_value:.2f} €")
    print(f"Performance : {perf_pct:.2f} %")

    print("\n=== MÉTRIQUES DE RISQUE ===")
    print(f"Drawdown maximal : {max_drawdown * 100:.2f} %")
    print(f"Volatilité journalière : {volatility * 100:.2f} %")
    print(f"Sharpe ratio : {sharpe_ratio:.2f}")

    # Sauvegarde CSV backtest
    out_path = os.path.join(DATA_FOLDER, "backtest_global.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nDétails sauvegardés dans : {out_path}")

    # Sauvegarde CSV des métriques
    metrics_path = os.path.join(DATA_FOLDER, "backtest_metrics.csv")
    pd.DataFrame([{
        "capital_initial": CAPITAL_INITIAL,
        "final_value": final_value,
        "performance_pct": perf_pct,
        "max_drawdown_pct": max_drawdown * 100,
        "volatility_pct": volatility * 100,
        "sharpe_ratio": sharpe_ratio
    }]).to_csv(metrics_path, index=False)
    print(f"Métriques sauvegardées dans : {metrics_path}")

    # =========================
    # GRAPHIQUE
    # =========================
    plt.figure(figsize=(10, 5))
    plt.plot(result_df["date"], result_df["total_value"], label="Portefeuille")
    plt.axhline(CAPITAL_INITIAL, linestyle="--", label="Capital initial")
    plt.xlabel("Date")
    plt.ylabel("Valeur du portefeuille (€)")
    plt.title("Évolution du portefeuille global (25 000 €)")
    plt.legend()
    plt.tight_layout()

    img_path = os.path.join(DATA_FOLDER, "backtest_global.png")
    plt.savefig(img_path)
    plt.close()
    print(f"Graphique sauvegardé dans : {img_path}")

    return result_df


if __name__ == "__main__":
    backtest_global()
