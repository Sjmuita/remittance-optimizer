import streamlit as st
from datetime import datetime
import json
from pathlib import Path

SENTIMENT_LOG = "user_actions.json"

def get_market_sentiment():
    if Path(SENTIMENT_LOG).exists():
        data = [json.loads(line) for line in open(SENTIMENT_LOG,"r") if line.strip()]
        last24h = [d for d in data if (datetime.utcnow() - datetime.fromisoformat(d["timestamp"])).total_seconds() < 86400]
        send = sum(1 for d in last24h if "send" in d["decision"].lower())
        wait = sum(1 for d in last24h if "wait" in d["decision"].lower())
        total = len(last24h)
        return send, wait, total
    return 0, 0, 0

def render_market_sentiment():
    send, wait, total = get_market_sentiment()
    st.header("Market Sentiment (24h)")
    if total:
        st.metric("Send Now %", f"{(send/total*100):.1f}")
        st.metric("Wait %", f"{(wait/total*100):.1f}")
    else:
        st.write("No data logged yet.")
