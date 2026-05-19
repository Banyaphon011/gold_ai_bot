import pandas as pd
import ta

# =========================
# โหลดข้อมูล
# =========================
df = pd.read_csv("data/gold_data.csv")

# =========================
# แปลงเป็นตัวเลข
# =========================
for col in ['Open', 'High', 'Low', 'Close']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# =========================
# เรียงข้อมูล
# =========================
df = df.sort_index()

# =========================
# INDICATORS
# =========================
df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

df['EMA_20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()

macd = ta.trend.MACD(df['Close'])
df['MACD'] = macd.macd()
df['MACD_SIGNAL'] = macd.macd_signal()

df['ATR'] = ta.volatility.AverageTrueRange(
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    window=14
).average_true_range()

# =========================
# CANDLESTICK
# =========================
df['Bullish'] = (df['Close'] > df['Open']).astype(int)
df['Bearish'] = (df['Close'] < df['Open']).astype(int)

# =========================
# 🔥 LAG FEATURES (เพิ่มแล้ว)
# =========================
df['Close_lag1'] = df['Close'].shift(1)
df['RSI_lag1'] = df['RSI'].shift(1)

# =========================
# TARGET
# =========================
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# =========================
# CLEAN DATA
# =========================
df.dropna(inplace=True)

# =========================
# FEATURES
# =========================
features = [
    'RSI',
    'EMA_20',
    'MACD',
    'MACD_SIGNAL',
    'ATR',
    'Bullish',
    'Bearish',
    'Close_lag1',
    'RSI_lag1'
]

X = df[features]
y = df['Target']

# =========================
# PREVIEW
# =========================
print(X.tail(1))

# =========================
# SAVE
# =========================
df.to_csv("data/gold_features.csv", index=False)

print("Saved gold_features.csv")