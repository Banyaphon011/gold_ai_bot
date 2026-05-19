import pandas as pd
import joblib

df = pd.read_csv("data/gold_features.csv")
model = joblib.load("data/random_forest_model.pkl")

features = [
    'RSI','EMA_20','MACD','MACD_SIGNAL',
    'ATR','Bullish','Bearish',
    'Close_lag1','RSI_lag1'
]

balance = 1000
equity = []

for i in range(len(df)-1):

    row = df[features].iloc[i:i+1]

    prob = model.predict_proba(row)[0]

    buy_prob = prob[1]
    sell_prob = prob[0]

    price_diff = df['Close'].iloc[i+1] - df['Close'].iloc[i]

    pnl = 0

    if buy_prob > 0.7 and buy_prob > sell_prob:
        pnl = price_diff * 0.1

    elif sell_prob > 0.7 and sell_prob > buy_prob:
        pnl = -price_diff * 0.1

    balance += pnl
    equity.append(balance)

print("Final:", balance)