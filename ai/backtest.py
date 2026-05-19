import pandas as pd
import joblib

# โหลดโมเดล
model = joblib.load("data/random_forest_model.pkl")

# โหลดข้อมูล
df = pd.read_csv("data/gold_features.csv")

# Features
features = [
    'RSI',
    'EMA_20',
    'MACD',
    'MACD_SIGNAL',
    'ATR',
    'Bullish',
    'Bearish'
]

# Predict
df['Prediction'] = model.predict(df[features])

# คำนวณผลตอบแทน
df['Return'] = df['Close'].pct_change()

# Strategy Return
df['Strategy_Return'] = (
    df['Prediction'].shift(1) * df['Return']
)

# ลบ NaN
df.dropna(inplace=True)

# Winrate
wins = (
    df['Strategy_Return'] > 0
).sum()

total = len(df)

winrate = wins / total

# Equity Curve
df['Equity'] = (
    1 + df['Strategy_Return']
).cumprod()

# แสดงผล
print("Winrate:", round(winrate * 100, 2), "%")

print(
    "Final Equity:",
    round(df['Equity'].iloc[-1], 2)
)

print(df[['Close', 'Prediction', 'Equity']].tail())