import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =========================
# LOAD
# =========================
df = pd.read_csv("data/gold_features.csv")
model = joblib.load("data/random_forest_model.pkl")

features = [
    'RSI','EMA_20','MACD','MACD_SIGNAL',
    'ATR','Bullish','Bearish',
    'Close_lag1','RSI_lag1'
]

X = df[features]

balance = 1000
balances = []

cooldown = 5
last_trade = -cooldown

# =========================
# SIMULATION
# =========================
for i in range(len(X)-1):

    if i - last_trade < cooldown:
        balances.append(balance)
        continue

    row = X.iloc[i:i+1]
    prob = model.predict_proba(row)[0]

    buy_prob = prob[1]
    sell_prob = prob[0]

    price_change = df['Close'].iloc[i+1] - df['Close'].iloc[i]

    pnl = 0

    if buy_prob > 0.60 and buy_prob > sell_prob:

        last_trade = i
        pnl = price_change * 0.1 - 0.2  # cost

    elif sell_prob > 0.60 and sell_prob > buy_prob:

        last_trade = i
        pnl = -price_change * 0.1 - 0.2  # cost

    balance += pnl
    balances.append(balance)
    max_balance = max(balances)

drawdown = [
    (x - max_balance) / max_balance * 100
    for x in balances
]

max_dd = min(drawdown)

# =========================
# PLOT
# =========================
plt.figure()

plt.subplot(2,1,1)
plt.plot(balances)
plt.title("Equity Curve")
plt.ylabel("Balance")

plt.subplot(2,1,2)
plt.plot(drawdown)
plt.title(f"Drawdown (Max: {max_dd:.2f}%)")
plt.ylabel("%")

plt.tight_layout()
plt.show()