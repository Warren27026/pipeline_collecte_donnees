import os
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

# Configuration des chemins
DATA_FOLDER = "data"
# On utilise le fichier contenant toutes les features calculées
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
    # On supprime les lignes où les valeurs futures (Target) ne sont pas encore connues (le dernier jour)
    df = df.dropna(subset=["Close", "Close_next", "Return_next"])
    
    if df.empty:
        print("[ERREUR] Pas assez de données pour l'évaluation (DataFrame vide après nettoyage).")
        return
        
    print(f"Nombre d'échantillons évalués : {len(df)}")

    print("\n" + "="*50)
    print("   1. BASELINE DE RÉGRESSION (Prédiction de Prix)")
    print("="*50)
    
    # --- MODÈLE NAÏF (Persistence Model) ---
    # Hypothèse : Le prix de demain sera identique au prix d'aujourd'hui.
    # C'est souvent difficile à battre en finance sur des horizons très courts.
    
    y_true_price = df["Close_next"]  # La réalité
    y_pred_naive = df["Close"]       # La prédiction naïve (J = J-1)
    
    # Calcul de l'erreur quadratique moyenne (RMSE)
    rmse_naive = np.sqrt(mean_squared_error(y_true_price, y_pred_naive))
    
    print(f"Modèle : Naive Prediction (Prix(J+1) = Prix(J))")
    print(f"Metric : RMSE (Root Mean Squared Error)")
    print(f"--------------------------------------------------")
    print(f" ERREUR MOYENNE (RMSE) : {rmse_naive:.4f} $")
    print(f"--------------------------------------------------")
    print(f"👉 OBJECTIF : Le modèle IA (Random Forest/LSTM) devra avoir une RMSE < {rmse_naive:.4f} $")
    print(f"   (Sinon, il est moins bon qu'une simple copie du prix de la veille)")


    print("\n" + "="*50)
    print("   2. BASELINE DE CLASSIFICATION (Achat / Vente)")
    print("="*50)
    
    # --- MODÈLE STATISTIQUE (Dummy Classifier) ---
    # Hypothèse : On prédit toujours la classe majoritaire.
    # Exemple : Si le marché monte 55% du temps, prédire "HAUSSE" tout le temps donne 55% de réussite.
    
    # Création de la cible binaire : 1 si le rendement demain est positif (Hausse), 0 sinon (Baisse)
    df["Target_Binary"] = (df["Return_next"] > 0).astype(int)
    
    X = df[["Close"]] # Feature fictive (le Dummy s'en fiche)
    y = df["Target_Binary"]
    
    # Initialisation du Dummy Classifier de Scikit-Learn
    # strategy="most_frequent" -> Prédit toujours la classe la plus représentée
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X, y)
    
    # Prédiction et Score
    score_dummy = dummy_clf.score(X, y)
    class_majoritaire = "HAUSSE" if dummy_clf.predict([0])[0] == 1 else "BAISSE"
    
    print(f"Modèle : Dummy Classifier (Stratégie 'Most Frequent')")
    print(f"Classe majoritaire détectée : {class_majoritaire}")
    print(f"--------------------------------------------------")
    print(f"PRÉCISION DE RÉFÉRENCE (Accuracy) : {score_dummy*100:.2f} %")
    print(f"--------------------------------------------------")
    print(f"OBJECTIF : Notre modèle IA devra avoir une précision > {score_dummy*100:.2f} %")
    print(f"   (Attention : 50% n'est pas la référence si le marché est haussier, c'est ce chiffre qu'il faut battre)")

if __name__ == "__main__":
    evaluate_baselines()
