import yfinance as yf
import pandas as pd

# Download USDKES historical daily data (last 10 years for example)
ticker = "USDKES=X"
data = yf.download(ticker, start="2003-12-01", end="2025-12-31", interval="1d")

# Reset index for easy handling
data = data.reset_index()

# Create your final FX dataset for ML
data = data.rename(columns={"Date": "date", "Close": "fx_usd_kes"})
data = data[['date', 'fx_usd_kes']].dropna().sort_values('date')

# Save to CSV
data.to_csv("cbk_usd_kes_history.csv", index=False)
print(data.tail())
print("✅ Downloaded and cleaned USD/KES history saved to cbk_usd_kes_history.csv")
