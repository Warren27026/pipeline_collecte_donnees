## 📊 Explication des métriques de performance et de risque

Afin d’évaluer le comportement du portefeuille simulé, plusieurs métriques financières sont calculées à partir de l’évolution de la valeur totale du portefeuille (`total_value`).  
Ces métriques permettent d’analyser à la fois **la performance** et **le risque** de la stratégie.

---

### 💰 Valeur finale du portefeuille (`final_value`)

**Définition :**  
La valeur finale correspond à la **valeur totale du portefeuille à la dernière date du backtest**.

Elle inclut :
- l’argent liquide disponible (cash)
- la valeur des actions détenues (positions × prix)

**Interprétation :**
- Si la valeur finale est supérieure au capital initial, la stratégie est gagnante
- Sinon, la stratégie est perdante

---

### 📈 Performance (%) (`performance_pct`)

**Définition :**  
La performance mesure le **gain ou la perte en pourcentage** par rapport au capital initial.

**Idée simple :**  
> “Combien d’argent ai-je gagné ou perdu au total ?”

**Interprétation :**
- Performance positive → gain
- Performance négative → perte

Cette métrique permet de comparer facilement plusieurs stratégies.

---

### 📉 Drawdown maximal (`max_drawdown_pct`)

**Définition :**  
Le drawdown maximal représente la **plus forte baisse du portefeuille** entre un point haut et le point bas qui suit.

**Idée simple :**  
> “Jusqu’où le portefeuille est-il descendu avant de remonter ?”

**Pourquoi c’est important :**
- Une stratégie peut être rentable, mais subir de fortes pertes temporaires
- Le drawdown mesure le **risque réel** et la difficulté psychologique à suivre la stratégie

**Interprétation :**
- Drawdown faible → stratégie plus stable
- Drawdown élevé → stratégie plus risquée

---

### 📊 Volatilité journalière (`volatility_pct`)

**Définition :**  
La volatilité mesure à quel point la valeur du portefeuille **varie d’un jour à l’autre**.

**Idée simple :**  
> “Est-ce que la courbe est régulière ou très instable ?”

**Pourquoi c’est important :**
- Une stratégie très volatile est plus risquée
- Elle est aussi plus difficile à suivre sur le long terme

**Interprétation :**
- Volatilité faible → portefeuille stable
- Volatilité élevée → portefeuille instable

---

### ⚖️ Sharpe Ratio (`sharpe_ratio`)

**Définition :**  
Le Sharpe ratio met en relation :
- la performance moyenne
- le risque pris (volatilité)

**Idée simple :**  
> “Est-ce que le gain obtenu vaut le risque pris ?”

**Interprétation générale :**
- Sharpe < 0 → mauvaise stratégie
- Sharpe ≈ 0.5 → faible
- Sharpe ≈ 1 → correct
- Sharpe ≥ 2 → très bon

Un Sharpe élevé indique une stratégie plus efficace et mieux équilibrée.

---

### 📈 Courbe d’évolution du portefeuille

**Définition :**  
La courbe représente l’évolution de la valeur totale du portefeuille dans le temps.

**Ce qu’elle permet d’observer :**
- la tendance globale (hausse ou baisse)
- les périodes de pertes importantes
- la stabilité ou l’instabilité de la stratégie

C’est la visualisation la plus importante du backtest.

---

## 🧠 Pourquoi ces métriques sont adaptées à un modèle dummy

Le modèle utilisé étant un **modèle dummy basé sur des règles fixes**,  
il n’est pas évalué sur des métriques de prédiction (accuracy, précision, etc.),  
mais sur son **impact réel sur un portefeuille financier**.

Ces métriques permettent de :
- mesurer la rentabilité
- évaluer le risque
- comparer la stratégie à une approche passive (Buy & Hold)

Elles constituent une base de référence avant l’introduction de modèles plus avancés.
