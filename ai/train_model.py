import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/gold_features.csv")

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
# SPLIT (FIXED)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False
)

# =========================
# MODEL (IMPROVED)
# =========================
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_split=4,
    class_weight="balanced",
    random_state=42
)

# =========================
# TRAIN
# =========================
model.fit(X_train, y_train)

# =========================
# ACCURACY
# =========================
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, "data/random_forest_model.pkl")

print("Model Saved")