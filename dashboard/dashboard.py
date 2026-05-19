import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/equity.csv")

# ถ้ามีเวลาให้ใช้เวลา
if "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"])
    plt.plot(df["time"], df["balance"])
else:
    plt.plot(df["balance"])

plt.title("Equity Curve")
plt.xlabel("Time")
plt.ylabel("Balance")
plt.grid(True)

plt.show()