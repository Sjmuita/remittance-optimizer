"""
live_fx_fetcher.py (robust version)
Fetch and update USD→KES FX rates (historical batch + live).
Handles API errors and missing keys.

Author: Muita Shalyn J. (USIU-A)
Date: Nov 2025
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import time

FX_CSV = "fx_usd_kes_history.csv"
API_URL = "https://api.exchangerate.host/"
RETRIES = 3
RETRY_DELAY = 2  # Seconds

class LiveFXFetcher:
    def __init__(self, log_file=FX_CSV):
        self.log_file = log_file

    def download_historical_fx(self, start='2017-01-01', end=None):
        """Download full daily USD→KES history from exchangerate.host with error handling and retry."""
        if not end:
            end = datetime.today().strftime('%Y-%m-%d')
        url = f"{API_URL}timeseries?start_date={start}&end_date={end}&base=USD&symbols=KES"
        print(f"Fetching batch historic rates {start} to {end}...")

        for attempt in range(RETRIES):
            try:
                response = requests.get(url)
                if response.status_code != 200:
                    print(f"HTTP Error {response.status_code}: {response.text}")
                    time.sleep(RETRY_DELAY)
                    continue
                data_json = response.json()
                if not data_json.get("success", True) or "rates" not in data_json:
                    print("API error:", data_json)
                    time.sleep(RETRY_DELAY)
                    continue
                # success
                records = [
                    {"date": date, "fx_usd_kes": v["KES"]}
                    for date, v in data_json["rates"].items()
                ]
                df = pd.DataFrame(records)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                df.to_csv(self.log_file, index=False)
                print(f"Saved {len(df)} records to {self.log_file}")
                return df
            except Exception as e:
                print(f"Attempt {attempt+1}: Exception occurred: {e}")
                time.sleep(RETRY_DELAY)
        print("❌ Failed to fetch batch FX rates after multiple retries.")
        return None

    def fetch_live_fx(self):
        """Fetch today's USD→KES from public API, handles errors."""
        url = f"{API_URL}latest?base=USD&symbols=KES"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data_json = response.json()
            rate = data_json["rates"].get("KES")
            if rate is None:
                print("API response missing 'KES' key:", data_json)
                return None
            now = pd.Timestamp.utcnow()
            print(f"Fetched live USD→KES: {rate:.2f} at {now}")
            return {"date": now.normalize(), "fx_usd_kes": rate, "timestamp": now}
        except Exception as e:
            print(f"❌ Live FX fetch error: {e}")
            return None

    def append_today_if_new(self):
        """Fetch live, append today’s value if NOT already in log."""
        today = datetime.utcnow().date()
        df = self.load_fx_history()
        if df is not None and today in pd.to_datetime(df['date']).dt.date.values:
            print("Today's rate already in log.")
            return df
        live = self.fetch_live_fx()
        if live and live["fx_usd_kes"] is not None:
            new_row = pd.DataFrame([{"date": today, "fx_usd_kes": live["fx_usd_kes"]}])
            df = pd.concat([df, new_row], ignore_index=True) if df is not None else new_row
            df = df.sort_values('date').reset_index(drop=True)
            df.to_csv(self.log_file, index=False)
            print(f"Appended latest rate to {self.log_file}")
        else:
            print("❌ Could not fetch today's FX; no update made.")
        return df

    def load_fx_history(self):
        """Load local CSV log or try to batch-download if missing."""
        if not os.path.exists(self.log_file):
            print("FX history log not found. Downloading batch now...")
            return self.download_historical_fx()
        df = pd.read_csv(self.log_file, parse_dates=['date'])
        return df

    def get_recent_series(self, days=90):
        df = self.load_fx_history()
        cutoff = datetime.utcnow() - timedelta(days=days)
        return df[df['date'] >= cutoff]

if __name__ == '__main__':
    fx = LiveFXFetcher()
    # 1. Download batch if needed
    if not os.path.exists(FX_CSV):
        fx.download_historical_fx(start="2017-01-01")
    # 2. Append today’s latest rate if new
    df = fx.append_today_if_new()
    if df is not None:
        print(df.tail(5))
    else:
        print("❌ FX history not available right now.")
