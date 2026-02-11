import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px

st.set_page_config(page_title="OSHA Severe Injury Dashboard", layout="wide")

st.title("OSHA Severe Injury Analytics Dashboard")
st.caption("Industry risk patterns and injury type intelligence")

DATA_PATH = Path("data/severeinjury.csv")

# ---------- Load ----------
@st.cache_data
def load_data():
    for enc in ["utf-8", "cp1252", "latin1"]:
        try:
            return pd.read_csv(DATA_PATH, encoding=enc, encoding_errors="replace", low_memory=False)
        except:
            continue
    st.error("Failed to load dataset")
    st.stop()

df = load_data()
df.columns = df.columns.str.strip()

# ---------- Detect columns ----------
industry_col = next((c for c in df.columns if "industry" in c.lower()), None)
injury_col = next((c for c in df.columns if "nature" in c.lower()), None)
state_col = next((c for c in df.columns if "state" in c.lower()), None)
date_col = next((c for c in df.columns if "date" in c.lower()), None)

# ---------- KPIs ----------
st.subheader("Key Metrics")

c1, c2, c3 = st.columns(3)
c1.metric("Total Severe Injuries", len(df))
if industry_col:
    c2.metric("Industries Covered", df[industry_col].nunique())
if state_col:
    c3.metric("States Covered", df[state_col].nunique())

# ---------- Industry Chart ----------
if industry_col:
    st.subheader("Top Industries by Severe Injuries")
    top_ind = df[industry_col].value_counts().head(15).reset_index()
    top_ind.columns = ["industry", "count"]
    fig = px.bar(top_ind, x="count", y="industry", orientation="h")
    st.plotly_chart(fig, use_container_width=True)

# ---------- Injury Types ----------
if injury_col:
    st.subheader("Top Injury Types")
    top_inj = df[injury_col].value_counts().head(15).reset_index()
    top_inj.columns = ["injury", "count"]
    fig2 = px.bar(top_inj, x="count", y="injury", orientation="h")
    st.plotly_chart(fig2, use_container_width=True)

# ---------- State ----------
if state_col:
    st.subheader("Top States")
    st.dataframe(df[state_col].value_counts().head(15))

# ---------- Time Trend ----------
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["year"] = df[date_col].dt.year
    trend = df["year"].value_counts().sort_index()
    st.subheader("Yearly Trend")
    st.line_chart(trend)

st.success("Dashboard ready.")