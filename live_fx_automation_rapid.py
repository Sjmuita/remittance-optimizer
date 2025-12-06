import requests
import pandas as pd
import os
import time
from datetime import datetime

FX_CSV = "fx_usd_kes_history.csv"
RAPID_API_KEY = "88e7d0431fmshfd313d3b11f12cep1d2b0fjsn1e26a0efcbe2"
RAPID_API_HOST = "currency-conversion-and-exchange-rates.p.rapidapi.com"

def fetch_usd_kes_rapid():
    url = "https://currency-conversion-and-exchange-rates.p.rapidapi.com/convert"
    headers = {
        "X-RapidAPI-Key": RAPID_API_KEY,
        "X-RapidAPI-Host": RAPID_API_HOST
    }
    params = {
        "from": "USD",
        "to": "KES",
        "amount": 1
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        return float(data.get("result", 0))
    return None

def append_fx_to_csv(rate):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    new_row = pd.DataFrame([{"date": today, "fx_usd_kes": rate}])
    if os.path.exists(FX_CSV):
        df = pd.read_csv(FX_CSV)
        if (df['date'] == today).any():
            df.loc[df['date'] == today, 'fx_usd_kes'] = rate
        else:
            df = pd.concat([df, new_row], ignore_index=True)
        df = df.sort_values('date').reset_index(drop=True)
    else:
        df = new_row
    df.to_csv(FX_CSV, index=False)
    print(f"Appended today's rate {rate}")

if __name__ == "__main__":
    print("=== SmartRemit AI: Live FX Updater ===")
    while True:
        rate = fetch_usd_kes_rapid()
        if rate:
            append_fx_to_csv(rate)
        else:
            print("❌ FX fetch failed.")
        print("Waiting 60 seconds...")
        time.sleep(60)
