import streamlit as st
import json
from datetime import datetime
from pathlib import Path

REMINDER_LOG = "reminders.json"

def render_reminder_setting():
    st.subheader("Set a rate alert")
    fx_alert = st.number_input("Alert me when FX reaches (KES)", value=160)
    if st.button("Set Alert"):
        data = []
        if Path(REMINDER_LOG).exists():
            try: data = json.loads(Path(REMINDER_LOG).read_text())
            except: data = []
        data.append({
            "timestamp": datetime.utcnow().isoformat(),
            "threshold": fx_alert,
            "user": "demo_user"
        })
        Path(REMINDER_LOG).write_text(json.dumps(data, indent=2))
        st.success(f"Alert set for {fx_alert}")
