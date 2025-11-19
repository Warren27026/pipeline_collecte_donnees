# pipeline_collecte_donnees
Pipeline pour mon projet de prédiction de valeur boursière


#  Trading Analytics Pipeline — Collecte, Nettoyage & Analyse des Données Boursières

Ce projet met en place un pipeline automatisé qui :

1. **Collecte** quotidiennement les données financières (yFinance + Tiingo)
2. **Nettoie** et normalise les données
3. **Génère des indicateurs techniques clés** (Bollinger, RSI, MACD, returns…)
4. **Construit une base exploitable pour un modèle AI** (prédiction de prix & signaux BUY/SELL)
5. **Sauvegarde** automatiquement les fichiers CSV dans le dossier `data/`

Le pipeline est conçu pour s’exécuter automatiquement via **GitHub Actions**, chaque jour à 20h.

---

# Objectifs du Projet

*  Automatiser la collecte des prix journaliers d’actions (AAPL, MSFT, TSLA…)
*  Nettoyer et structurer les données pour une utilisation en IA
*  Générer des indicateurs techniques utilisés en finance quantitative
*  Préparer le terrain pour un modèle prédictif/algorithmique
*  Créer une base pour des signaux d’achat/vente exploitables

---

#  Pipeline Complet

## 1. Collecte Automatisée

Le pipeline récupère chaque jour 1 an d’historique via :

* **yFinance** : AAPL, TSLA, MSFT, BTC-USD, GOOGL
* **Tiingo** : AAPL, TSLA, MSFT, GOOGL

Ces données sont stockées dans :

```
data/
data/tiingo/
```

---

## 2. Nettoyage des Données

Le script applique :

✔ Normalisation des dates
✔ Suppression des doublons
✔ Suppression/Interpolation des valeurs manquantes
✔ Correction des outliers (méthode IQR 3×)
✔ Filtrage des volumes négatifs
✔ Tri chronologique

Les données propres sont enregistrées sous forme :

```
AAPL.csv
MSFT.csv
...
```

---

## 3. Génération des Indicateurs Techniques

Chaque actif reçoit un fichier enrichi :

```
AAPL_features.csv
```

Avec :

###  Bandes de Bollinger (Volatilité)

* `Bollinger_Middle`
* `Bollinger_Upper`
* `Bollinger_Lower`
* `Bollinger_%B`
* `Bollinger_Width`

###  Momentum – RSI (14)

* `RSI_14`

###  Tendance – MACD

* `MACD`
* `MACD_Signal`
* `MACD_Diff`

###  Performance

* `Return`
* `Return_next`
* `Close_next`

---

#  Description des Variables Calculées

## 🟦 Prix bruts

| Variable | Description       |
| -------- | ----------------- |
| Open     | Prix d’ouverture  |
| High     | Plus haut du jour |
| Low      | Plus bas du jour  |
| Close    | Prix de clôture   |
| Volume   | Activité du jour  |

## 🟧 Bandes de Bollinger

| Variable         | Signification           |
| ---------------- | ----------------------- |
| Bollinger_Middle | Moyenne mobile 20 jours |
| Bollinger_Upper  | SMA20 + 2σ              |
| Bollinger_Lower  | SMA20 – 2σ              |
| Bollinger_%B     | Position dans le canal  |
| Bollinger_Width  | Volatilité du marché    |

## 🟩 RSI – Momentum

| Variable | Description       |
| -------- | ----------------- |
| RSI_14   | Surachat/survente |

## 🟨 MACD – Tendance

| Variable    | Description     |
| ----------- | --------------- |
| MACD        | EMA12 – EMA26   |
| MACD_Signal | EMA9 du MACD    |
| MACD_Diff   | Signal BUY/SELL |

## 🟥 Returns

| Variable    | Description                   |
| ----------- | ----------------------------- |
| Return      | Rendement du jour             |
| Return_next | Rendement du lendemain        |
| Close_next  | Prix du lendemain (target ML) |

---

#  Modèle de Base BUY / SELL

Le pipeline permet de créer facilement une première stratégie :

###  **Signal BUY** si :

* `Close < Bollinger_Lower`
* `RSI_14 < 30`
* `MACD_Diff > 0`

###  **Signal SELL** si :

* `Close > Bollinger_Upper`
* `RSI_14 > 70`
* `MACD_Diff < 0`

Ce modèle simple sert de baseline pour les futurs modèles IA (RandomForest, LSTM…).

---

#  Installation

### 1. Cloner le repo

```bash
git clone https://github.com/Warren27026/pipeline_collecte_donnees
cd pipeline_collecte_donnees
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

Assurez-vous d’avoir :

```
yfinance
tiingo
pandas
numpy
ta
```

### 3. Ajouter vos clés API

Dans les **GitHub Secrets** :

* `TIINGO_API_KEY`
* `PUSH_TOKEN` (Personal Access Token pour push auto)

---

# ⚡ Exécution Manuelle

```bash
python pipeline.py
```

Cela génère :

```
data/AAPL.csv
data/AAPL_features.csv
data/ALL_YFINANCE_features.csv
...
```

---

# GitHub Actions

Le pipeline est exécuté automatiquement chaque soir à **20h** pour mettre les prix à jour.

---

#  Architecture du Projet

```
root/
├── donne_collectes_nettoye.py               # Script principal
├── data/
│   ├── AAPL.csv
│   ├── AAPL_features.csv
│   ├── ALL_YFINANCE.csv
│   └── ...
│
└── data/tiingo/
    ├── AAPL_features.csv
    └── ALL_TIINGO_features.csv
```

---

# Contact & Contributions

Les contributions sont les bienvenues !
N'hésitez pas à ouvrir une **issue** ou un **pull request**.

---



