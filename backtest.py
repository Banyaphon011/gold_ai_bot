import pandas as pd
import joblib

# =========================
# LOAD DATA + MODEL
# =========================
df = pd.read_csv("data/gold_features.csv")
model = joblib.load("data/random_forest_model.pkl")

features = [
    'RSI','EMA_20','MACD','MACD_SIGNAL',
    'ATR','Bullish','Bearish',
    'Close_lag1','RSI_lag1'
]

X = df[features].reset_index(drop=True)
df = df.reset_index(drop=True)

# =========================
# SETTINGS
# =========================
balance = 1000
win = 0
loss = 0
trades = 0

cooldown = 5
last_trade_index = -cooldown

# =========================
# BACKTEST LOOP
# =========================
for i in range(len(X)-1):

    # cooldown
    if i - last_trade_index < cooldown:
        continue

    row = X.iloc[i:i+1]
    prob = model.predict_proba(row)[0]

    buy_prob = prob[1]
    sell_prob = prob[0]

    price_change = df['Close'].iloc[i+1] - df['Close'].iloc[i]

    # =========================
    # BUY SIGNAL
    # =========================
    if buy_prob > 0.60 and buy_prob > sell_prob:

        trades += 1
        last_trade_index = i

        pnl = price_change * 0.1

        # SL / TP
        tp = 2.0
        sl = -1.5

        if pnl >= tp:
            pnl = tp
        elif pnl <= sl:
            pnl = sl

        pnl -= 0.2  # cost

        balance += pnl

        if pnl > 0:
            win += 1
        else:
            loss += 1

    # =========================
    # SELL SIGNAL (FIXED elif)
    # =========================
    elif sell_prob > 0.60 and sell_prob > buy_prob:

        trades += 1
        last_trade_index = i

        pnl = -price_change * 0.1

        # SL / TP
        tp = 2.0
        sl = -1.5

        if pnl >= tp:
            pnl = tp
        elif pnl <= sl:
            pnl = sl

        pnl -= 0.2  # cost

        balance += pnl

        if pnl > 0:
            win += 1
        else:
            loss += 1

# =========================
# RESULT
# =========================
winrate = (win / trades * 100) if trades > 0 else 0

print("\n========== BACKTEST RESULT ==========")
print("Trades:", trades)
print("Win:", win)
print("Loss:", loss)
print("Winrate:", round(winrate, 2), "%")
print("Final Balance:", round(balance, 2))
print("=====================================")