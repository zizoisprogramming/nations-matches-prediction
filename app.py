"""
Match predictor UI.

    make app        (or: uv run streamlit run app.py)
"""
import datetime as dt
import json
from pathlib import Path

import streamlit as st

from src.predict import predict_match

TEAM_IDS_PATH = Path(__file__).parent / "data" / "cache" / "team_ids.json"


@st.cache_data
def team_names():
    return sorted(json.loads(TEAM_IDS_PATH.read_text()))


st.set_page_config(page_title="Match predictor", page_icon="⚽", layout="centered")
st.title("⚽ Match predictor")
st.caption("Pick two national teams and a date. The venue comes from ESPN; the form, weather "
           "and travel features are built live, then the current model predicts the result.")

teams = team_names()
col1, col2 = st.columns(2)
with col1:
    team_a = st.selectbox("Home team", teams, index=None, placeholder="Type to search…")
with col2:
    team_b = st.selectbox("Away team", teams, index=None, placeholder="Type to search…")

today = dt.date.today()
choice = st.radio("Date", ["Today", "Tomorrow", "Earlier"], horizontal=True)
if choice == "Today":
    date = today
elif choice == "Tomorrow":
    date = today + dt.timedelta(days=1)
else:
    date = st.date_input("Match date", value=today - dt.timedelta(days=1),
                         max_value=today - dt.timedelta(days=1))

ready = team_a and team_b and team_a != team_b
if team_a and team_a == team_b:
    st.warning("Pick two different teams.")

if st.button("Predict", type="primary", disabled=not ready):
    with st.status(f"Predicting {team_a} vs {team_b} on {date:%a %d %b %Y} …", expanded=True) as status:
        try:
            r = predict_match(team_a, team_b, date.isoformat(), log=st.write)
            status.update(label="Done", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Prediction failed", state="error")
            st.error(str(e))
            st.stop()

    st.subheader(f"{r['home_team']} vs {r['away_team']}")
    where = ", ".join(x for x in [r["city"], r["country"]] if x)
    st.caption(" · ".join(x for x in [r["tournament"], where, r["date"]] if x))

    outcomes = [(f"{r['home_team']} win", r["prob_home_win"]),
                ("Draw", r["prob_draw"]),
                (f"{r['away_team']} win", r["prob_away_win"])]
    best = max(outcomes, key=lambda o: o[1])[0]
    cols = st.columns(3)
    for c, (label, p) in zip(cols, outcomes):
        c.metric(label + (" ✓" if label == best else ""), f"{p:.0%}")
        c.progress(p)

    if r["actual"]:
        st.info(f"Actual result: **{r['actual']}**")
    for n in r["notes"]:
        st.caption(f"ℹ️ {n}")
