import requests

url = "https://currency-conversion-and-exchange-rates.p.rapidapi.com/convert"
headers = {
    "X-RapidAPI-Key": "88e7d0431fmshfd313d3b11f12cep1d2b0fjsn1e26a0efcbe2",
    "X-RapidAPI-Host": "currency-conversion-and-exchange-rates.p.rapidapi.com"
}
params = {
    "from": "USD",
    "to": "KES",
    "amount": 1
}

response = requests.get(url, headers=headers, params=params)
print("Status:", response.status_code)
try:
    data = response.json()
    print("Full response:", data)
    if "result" in data:
        print("Live USD/KES rate:", data["result"])
except Exception as e:
    print("Error parsing response:", e)
