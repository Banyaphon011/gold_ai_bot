import yfinance as yf

gold = yf.download(
    "GC=F",
    period="30d",
    interval="1h"
)

gold.to_csv("data/gold_data.csv")

print("Saved")