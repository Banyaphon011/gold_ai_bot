import time
import requests
import joblib
import pandas as pd
import yfinance as yf
import ta
import MetaTrader5 as mt5
import csv

# =========================
# MODEL
# =========================
model = joblib.load("data/random_forest_model.pkl")

# =========================
# TELEGRAM
# =========================
TOKEN = "YOUR_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": msg})

# =========================
# LOGS
# =========================
def log_trade(side, price, prob):
    with open("trade_log.csv", "a", newline="") as f:
        csv.writer(f).writerow([time.time(), side, price, prob])

def log_equity(balance):
    with open("data/equity.csv", "a", newline="") as f:
        csv.writer(f).writerow([time.time(), balance])

# =========================
# MT5 INIT
# =========================
mt5.initialize()

ACCOUNT = 123456
PASSWORD = "your_password"
SERVER = "your_server"

mt5.login(ACCOUNT, password=PASSWORD, server=SERVER)

# =========================
# STATE
# =========================
equity = 1000
position = None
entry_price = 0

risk_per_trade = 0.01
last_trade_time = 0
cooldown_sec = 60

# =========================
# ORDER
# =========================
def place_order(symbol, lot, order_type):

    tick = mt5.symbol_info_tick(symbol)
    point = mt5.symbol_info(symbol).point

    if order_type == "BUY":
        price = tick.ask
        mt5_type = mt5.ORDER_TYPE_BUY
    else:
        price = tick.bid
        mt5_type = mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5_type,
        "price": price,
        "deviation": 20,
        "magic": 123456,
        "comment": "AI BOT",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    return mt5.order_send(request), price

# =========================
# LOOP
# =========================
while True:

    try:

        # =========================
        # DATA
        # =========================
        df = yf.download("GC=F", period="5d", interval="5m")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna()

        for c in ["Open","High","Low","Close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # =========================
        # INDICATORS
        # =========================
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"], 14).rsi()
        df["EMA_20"] = ta.trend.EMAIndicator(df["Close"], 20).ema_indicator()

        macd = ta.trend.MACD(df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_SIGNAL"] = macd.macd_signal()

        df["ATR"] = ta.volatility.AverageTrueRange(
            df["High"], df["Low"], df["Close"], 14
        ).average_true_range()

        df["Bullish"] = (df["Close"] > df["Open"]).astype(int)
        df["Bearish"] = (df["Close"] < df["Open"]).astype(int)

        df["Close_lag1"] = df["Close"].shift(1)
        df["RSI_lag1"] = df["RSI"].shift(1)

        df.dropna(inplace=True)

        features = [
            "RSI","EMA_20","MACD","MACD_SIGNAL",
            "ATR","Bullish","Bearish",
            "Close_lag1","RSI_lag1"
        ]

        latest = df[features].tail(1)

        # =========================
        # PREDICT
        # =========================
        prob = model.predict_proba(latest)[0]

        buy_prob = prob[1]
        sell_prob = prob[0]

        price = df["Close"].iloc[-1]

        # =========================
        # POSITION PNL (REAL)
        # =========================
        pnl = 0
        if position == "BUY":
            pnl = price - entry_price
        elif position == "SELL":
            pnl = entry_price - price

        equity = equity + pnl

        log_equity(equity)

        # =========================
        # KILL SWITCH
        # =========================
        if equity < 900:
            print("🚨 KILL SWITCH")
            break

        # =========================
        # COOLDOWN
        # =========================
        now = time.time()
        if now - last_trade_time < cooldown_sec:
            time.sleep(10)
            continue

        # =========================
        # LOT SIZE
        # =========================
        lot = max(round((equity * risk_per_trade) / 1000, 2), 0.01)

        # =========================
        # BUY
        # =========================
        if buy_prob > 0.70:

            order, entry_price = place_order("XAUUSD", lot, "BUY")

            if order:
                position = "BUY"
                log_trade("BUY", price, buy_prob)

                send_telegram(f"🟢 BUY\nPrice: {price}\nProb: {buy_prob:.2f}")

                last_trade_time = now

        # =========================
        # SELL
        # =========================
        elif sell_prob > 0.70:

            order, entry_price = place_order("XAUUSD", lot, "SELL")

            if order:
                position = "SELL"
                log_trade("SELL", price, sell_prob)

                send_telegram(f"🔴 SELL\nPrice: {price}\nProb: {sell_prob:.2f}")

                last_trade_time = now

        else:
            print("NO TRADE")

    except Exception as e:
        print("ERROR:", e)

    time.sleep(10)