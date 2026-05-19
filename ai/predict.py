import pandas as pd
import joblib

# โหลดโมเดล
model = joblib.load("data/random_forest_model.pkl")

# โหลดข้อมูล
df = pd.read_csv("data/gold_features.csv")

# เลือกข้อมูลล่าสุด
latest = df[
    [
        'RSI',
        'EMA_20',
        'MACD',
        'MACD_SIGNAL',
        'ATR',
        'Bullish',
        'Bearish'
    ]
].tail(1)

# Predict
prediction = model.predict(latest)[0]

# Probability
prob = model.predict_proba(latest)[0]

buy_prob = prob[1]
sell_prob = prob[0]

# แสดงผล
if prediction == 1:
    print("BUY SIGNAL")
else:
    print("SELL SIGNAL")

print(f"BUY Probability: {buy_prob:.2f}")
print(f"SELL Probability: {sell_prob:.2f}")