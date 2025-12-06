import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

csv_file = "cbk_usd_kes_history.csv"
ticker = "USDKES=X"

# Step 1: Determine the download start date
if os.path.exists(csv_file):
    existing = pd.read_csv(csv_file, parse_dates=["date"])
    last_date = existing["date"].max()
    # Ensure the next day (so no overlap)
    start_date = (pd.to_datetime(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")
    # Don't fetch if already up to date
    if pd.to_datetime(start_date) > datetime.today():
        print("Data is already up to date.")
        exit()
else:
    existing = None
    start_date = "2015-01-01"

end_date = datetime.today().strftime("%Y-%m-%d")

print(f"Downloading USDKES from {start_date} to {end_date}...")

# Step 2: Download just the new data
data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
if data.empty:
    print("No new data to add.")
    exit()
data = data.reset_index()
data = data.rename(columns={"Date": "date", "Close": "fx_usd_kes"})
data = data[['date', 'fx_usd_kes']].dropna().sort_values('date')

# Step 3: Append to or create the CSV
if existing is not None and not existing.empty:
    # Only append if dates are new
    combined = pd.concat([existing, data]).drop_duplicates(subset="date").sort_values("date")
else:
    combined = data

combined.to_csv(csv_file, index=False)
print(combined.tail())
print(f"✅ Updated USD/KES history saved to {csv_file}")
