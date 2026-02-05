# backtest_global.py

import os
import pandas as pd
import matplotlib.pyplot as plt  # <-- ajout
import numpy as np

DATA_FOLDER = "data"

SYMBOLS = [
    # =========================
    # Big Tech / Large Caps
    # =========================
    "AAPL",   # Apple – Technologie (hardware, services)
    "MSFT",   # Microsoft – Technologie (logiciels, cloud)
    "GOOGL",  # Alphabet – Technologie (internet, publicité, IA)
    "AMZN",   # Amazon – E-commerce / Cloud (AWS)
    "META",   # Meta Platforms – Réseaux sociaux / Métavers
    "NVDA",   # Nvidia – Semi-conducteurs / IA
    "TSLA",   # Tesla – Véhicules électriques / Énergie
    "INTC",   # Intel – Semi-conducteurs
    "AMD",    # AMD – Semi-conducteurs
    "IBM",    # IBM – Services informatiques / Cloud
    "ORCL",   # Oracle – Logiciels / Bases de données
    "NFLX",   # Netflix – Streaming / Médias

    # =========================
    # Startups / Growth Stocks
    # =========================
    "PLTR",   # Palantir – Data analytics / Intelligence artificielle
    "SNOW",   # Snowflake – Cloud data / Big Data
    "SHOP",   # Shopify – E-commerce / SaaS
    "COIN",   # Coinbase – Fintech / Crypto-exchange
    "ROKU",   # Roku – Streaming / Publicité
    "U",      # Unity Software – 3D / Jeux vidéo / Métavers
    "CRWD",   # CrowdStrike – Cybersécurité
    "ZS",     # Zscaler – Cybersécurité cloud
    "RIVN",   # Rivian – Véhicules électriques (startup)
    "LCID",   # Lucid Motors – Véhicules électriques (startup)

    # =========================
    # Crypto-actif
    # =========================
    "BTC-USD",  # Bitcoin – Crypto-monnaie
]

CAPITAL_INITIAL = 1000.0
FRAIS_TRANSACTION = 0.001  # 0.1% par ordre (Achat ou Vente)

def load_symbol_history(symbol: str, history_path: str):
    """Charge et nettoie les données d'un symbole."""
    try:
        df = pd.read_csv(history_path)
    except FileNotFoundError:
        return None

    # Filtrer sur le symbole
    df = df[df["symbol"] == symbol].copy()
    if df.empty:
        return None

    df["date"] = pd.to_datetime(df["date"])
    # Garder la dernière info par jour
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    # Renommage pour standardiser
    col_map = {"recommendation": "signal", "close": "Close"}
    df = df.rename(columns=col_map)

    # Vérification des colonnes nécessaires
    required_cols = ["date", "Close", "signal"]
    if not all(col in df.columns for col in required_cols):
        return None

    return df[required_cols]

def calculate_metrics(df_result):
    """Calcule Sharpe Ratio et Max Drawdown."""
    # Rendements quotidiens
    df_result["daily_return"] = df_result["total_value"].pct_change().fillna(0)
    
    # 1. Sharpe Ratio (Annualisé, hypothèse 252 jours de bourse, sans taux sans risque pour simplifier)
    mean_ret = df_result["daily_return"].mean()
    std_ret = df_result["daily_return"].std()
    if std_ret == 0:
        sharpe = 0
    else:
        sharpe = (mean_ret / std_ret) * np.sqrt(252) # Annualisé

    # 2. Max Drawdown (Perte maximale depuis un sommet)
    df_result["cummax"] = df_result["total_value"].cummax()
    df_result["drawdown"] = (df_result["total_value"] - df_result["cummax"]) / df_result["cummax"]
    max_drawdown = df_result["drawdown"].min()

    return sharpe, max_drawdown

def backtest_global():
    # history_path = os.path.join(DATA_FOLDER, "signals_history.csv")
    history_path = os.path.join(DATA_FOLDER, "latest_signals.csv") # Assurez-vous que ce fichier existe
    
    if not os.path.exists(history_path):
        print(f"[ERREUR] Fichier introuvable : {history_path}")
        return

    # 1. Chargement des données
    datasets = {}
    for sym in SYMBOLS:
        df_sym = load_symbol_history(sym, history_path)
        if df_sym is not None:
            datasets[sym] = df_sym.set_index("date") # Indexer par date pour faciliter l'accès

    if not datasets:
        print("[ERREUR] Aucune donnée valide trouvée.")
        return

    # Union de toutes les dates (Timeline globale)
    all_dates = sorted(list(set().union(*[d.index for d in datasets.values()])))
    
    if not all_dates:
        print("[ERREUR] Timeline vide.")
        return

    # --- INITIALISATION ---
    cash = CAPITAL_INITIAL
    positions = {sym: 0.0 for sym in datasets.keys()}
    last_known_price = {sym: None for sym in datasets.keys()}
    
    # Pour le Benchmark (Buy & Hold équipondéré)
    # On imagine qu'on investit 1000€ divisés par le nombre d'actifs au début
    bench_positions = {sym: 0.0 for sym in datasets.keys()}
    bench_cash = CAPITAL_INITIAL
    first_day_invested = False
    
    portfolio_history = []

    print(f"Démarrage du backtest sur {len(all_dates)} jours avec {len(datasets)} actifs...")

    # --- BOUCLE TEMPORELLE ---
    for current_date in all_dates:
        
        # 1. Mise à jour des prix du jour
        daily_prices = {}
        daily_signals = {}
        
        for sym, df in datasets.items():
            if current_date in df.index:
                row = df.loc[current_date]
                # Gérer les doublons potentiels (si index non unique)
                if isinstance(row, pd.DataFrame): 
                    row = row.iloc[-1]
                
                price = float(row["Close"])
                signal = row["signal"]
                
                daily_prices[sym] = price
                daily_signals[sym] = signal
                last_known_price[sym] = price
            else:
                # Si pas de donnée ce jour-là, on garde le signal HOLD et le vieux prix
                daily_prices[sym] = last_known_price[sym]
                daily_signals[sym] = "HOLD"

        # --- LOGIQUE BENCHMARK (Investir tout le 1er jour) ---
        if not first_day_invested:
            valid_start_symbols = [s for s, p in daily_prices.items() if p is not None]
            if valid_start_symbols:
                amt_per_asset = bench_cash / len(valid_start_symbols)
                for sym in valid_start_symbols:
                    p = daily_prices[sym]
                    qty = (amt_per_asset / p) * (1 - FRAIS_TRANSACTION) # On paie les frais
                    bench_positions[sym] = qty
                bench_cash = 0 # Tout est investi
                first_day_invested = True
        
        # Calcul valeur Benchmark
        val_bench = bench_cash
        for sym, qty in bench_positions.items():
            if daily_prices[sym] is not None:
                val_bench += qty * daily_prices[sym]

        # --- LOGIQUE STRATEGIE ---
        
        # A. D'abord les VENTES (SELL) pour libérer du cash
        for sym, signal in daily_signals.items():
            price = daily_prices[sym]
            if price is None: continue
            
            if signal == "SELL" and positions[sym] > 0:
                # Vente totale
                revenue = positions[sym] * price
                frais = revenue * FRAIS_TRANSACTION
                cash += (revenue - frais)
                positions[sym] = 0.0

        # B. Ensuite les ACHATS (BUY)
        # On identifie qui on veut acheter
        buy_candidates = [s for s, sig in daily_signals.items() if sig == "BUY" and daily_prices[s] is not None]
        
        if buy_candidates and cash > 1.0: # S'il reste au moins 1€
            # Stratégie simple : on divise le cash équitablement entre les candidats
            amount_to_invest = cash / len(buy_candidates)
            
            for sym in buy_candidates:
                price = daily_prices[sym]
                # Calcul quantité achetable en prenant en compte les frais
                # Cash = Qty * Price * (1 + Frais)
                # Donc Qty = Cash / (Price * (1 + Frais))
                qty = amount_to_invest / (price * (1 + FRAIS_TRANSACTION))
                
                cost = qty * price
                fees = cost * FRAIS_TRANSACTION
                
                if cash >= (cost + fees):
                    positions[sym] += qty
                    cash -= (cost + fees)

        # C. Calcul Valeur Portefeuille Stratégie
        val_strat = cash
        for sym, qty in positions.items():
            p = daily_prices[sym]
            if p is not None:
                val_strat += qty * p

        # Enregistrement
        portfolio_history.append({
            "date": current_date,
            "total_value": val_strat,
            "benchmark_value": val_bench,
            "cash": cash
        })

    # --- ANALYSE FINALE ---
    df_res = pd.DataFrame(portfolio_history)
    
    # Calculs finaux
    final_val_strat = df_res.iloc[-1]["total_value"]
    final_val_bench = df_res.iloc[-1]["benchmark_value"]
    
    perf_strat = (final_val_strat - CAPITAL_INITIAL) / CAPITAL_INITIAL * 100
    perf_bench = (final_val_bench - CAPITAL_INITIAL) / CAPITAL_INITIAL * 100
    
    sharpe, max_dd = calculate_metrics(df_res)

    print("\n" + "="*40)
    print(" RÉSULTATS BACKTEST AVANCÉ")
    print("="*40)
    print(f"Capital Initial   : {CAPITAL_INITIAL:.2f} €")
    print(f"Solde Final IA    : {final_val_strat:.2f} € ({perf_strat:+.2f}%)")
    print(f"Solde Benchmark   : {final_val_bench:.2f} € ({perf_bench:+.2f}%)")
    print("-" * 20)
    print(f"Surperformance IA : {perf_strat - perf_bench:+.2f} points")
    print(f"Sharpe Ratio      : {sharpe:.2f} ( > 1 est bon, > 2 excellent)")
    print(f"Max Drawdown      : {max_dd:.2%} (Chute maximale)")
    print("="*40)

    # Sauvegarde
    out_csv = os.path.join(DATA_FOLDER, "backtest_results.csv")
    df_res.to_csv(out_csv, index=False)

    # --- VISUALISATION ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot 1: Courbes de valeur
    ax1.plot(df_res["date"], df_res["total_value"], label="Stratégie IA", color="#00e676", linewidth=2)
    ax1.plot(df_res["date"], df_res["benchmark_value"], label="Benchmark (Buy & Hold)", color="gray", linestyle="--", alpha=0.7)
    ax1.set_title("Comparaison Performance : IA vs Marché", fontsize=14, color="white")
    ax1.set_ylabel("Valeur Portefeuille (€)", color="white")
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    # Style sombre pour faire "Tech"
    ax1.set_facecolor('#1e1e1e')
    fig.patch.set_facecolor('#121212')
    ax1.tick_params(colors='white')
    
    # Plot 2: Drawdown (Risque)
    ax2.fill_between(df_res["date"], df_res["drawdown"], 0, color="#ff5252", alpha=0.3, label="Drawdown")
    ax2.plot(df_res["date"], df_res["drawdown"], color="#ff5252", linewidth=1)
    ax2.set_ylabel("Drawdown", color="white")
    ax2.set_xlabel("Date", color="white")
    ax2.legend()
    ax2.grid(True, alpha=0.2)
    ax2.set_facecolor('#1e1e1e')
    ax2.tick_params(colors='white')
    
    # Sauvegarde Image
    out_img = os.path.join(DATA_FOLDER, "backtest_advanced.png")
    plt.tight_layout()
    plt.savefig(out_img, dpi=300)
    plt.close()
    print(f"Graphique généré : {out_img}")

if __name__ == "__main__":
    backtest_global()
