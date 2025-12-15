import os
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import mean_squared_error

# Configuration des chemins
DATA_FOLDER = "data"
FILE_PATH = os.path.join(DATA_FOLDER, "ALL_YFINANCE_features.csv")

def evaluate_baselines():
    """
    Calcule et affiche les performances des modèles de référence (Baselines).
    Ce sont les scores minimums que le futur modèle ML doit battre.
    """
    
    # 1. Chargement des données
    if not os.path.exists(FILE_PATH):
        print(f"[ERREUR] Le fichier de données est introuvable : {FILE_PATH}")
        print("Assurez-vous d'avoir lancé 'donne_collectes_nettoye.py' avant.")
        return

    print(f"\n--- CHARGEMENT DES DONNÉES : {FILE_PATH} ---")
    df = pd.read_csv(FILE_PATH)
    
    # Nettoyage : On a besoin de la cible (Close_next) et des features
    # On supprime les lignes où les valeurs futures (Target) ne sont pas encore connues
    df = df.dropna(subset=["Close", "Close_next", "Return_next"])
    
    if df.empty:
        print("[ERREUR] Pas assez de données pour l'évaluation (DataFrame vide après nettoyage).")
        return
        
    print(f"Nombre d'échantillons évalués : {len(df)}")

    print("\n" + "="*60)
    print("   1. BASELINE DE RÉGRESSION (Prédiction de Prix)")
    print("="*60)
    
    # --- MODÈLE NAÏF (Persistence Model) ---
    # Hypothèse : Le prix de demain sera identique au prix d'aujourd'hui.
    
    # Calcul Global (Attention : Biaisé par les gros prix comme BTC)
    rmse_global = np.sqrt(mean_squared_error(df["Close_next"], df["Close"]))
    print(f" Erreur Moyenne Globale (RMSE) : {rmse_global:.2f} $ (Attention: mélangé avec BTC!)")
    
    print("\n DÉTAIL PAR ACTIF (C'est ce tableau qui compte pour le rapport) :")
    print(f"{'Symbole':<10} | {'Prix Moyen':<12} | {'RMSE Naïf':<12} | {'Erreur %':<10}")
    print("-" * 55)
    
    symbols = df["symbol"].unique()
    for sym in symbols:
        df_sym = df[df["symbol"] == sym]
        
        # Calcul de l'erreur pour cet actif spécifique
        mse_sym = mean_squared_error(df_sym["Close_next"], df_sym["Close"])
        rmse_sym = np.sqrt(mse_sym)
        
        # Calcul du pourcentage d'erreur par rapport au prix moyen
        mean_price = df_sym["Close"].mean()
        error_pct = (rmse_sym / mean_price) * 100
        
        print(f"{sym:<10} | {mean_price:<12.2f} | {rmse_sym:<12.4f} | {error_pct:.2f}%")
        
    print("-" * 55)
    print(f" OBJECTIF : Ton modèle IA devra avoir une erreur % plus faible que ces chiffres.")


    print("\n" + "="*60)
    print("   2. BASELINE DE CLASSIFICATION (Achat / Vente)")
    print("="*60)
    
    # --- MODÈLE STATISTIQUE (Dummy Classifier) ---
    # Hypothèse : On prédit toujours la classe majoritaire.
    
    # Création de la cible binaire : 1 si le rendement demain est positif, 0 sinon
    df["Target_Binary"] = (df["Return_next"] > 0).astype(int)
    
    X = df[["Close"]] # Feature fictive
    y = df["Target_Binary"]
    
    # Initialisation du Dummy Classifier (Scikit-Learn)
    # strategy="most_frequent" -> Prédit toujours la classe la plus représentée
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X, y)
    
    # Score
    score_dummy = dummy_clf.score(X, y)
    class_majoritaire = "HAUSSE" if dummy_clf.predict([[0]])[0] == 1 else "BAISSE"
    
    print(f"Modèle : Dummy Classifier (Stratégie 'Most Frequent')")
    print(f"Tendance du marché détectée : {class_majoritaire}")
    print(f"--------------------------------------------------")
    print(f"PRÉCISION DE RÉFÉRENCE (Accuracy) : {score_dummy*100:.2f} %")
    print(f"--------------------------------------------------")
    print(f" OBJECTIF : Ton modèle IA devra avoir une précision > {score_dummy*100:.2f} %")
    print(f"   (C'est le seuil minimum pour battre le hasard statistique)")

if __name__ == "__main__":
    evaluate_baselines()
