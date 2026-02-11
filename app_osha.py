import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="OSHA Severe Injury Dashboard", layout="wide")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "severeinjury.csv"
RISK_PATH = ROOT / "outputs" / "osha_with_risk_score.csv"

st.title("OSHA Severe Injury Analytics Dashboard")
st.caption("Interactive safety analytics + risk scoring from OSHA severe injury records.")

# -----------------------------
# Load data (robust)
# -----------------------------
@st.cache_data
def load_csv_safely(path: Path):
    # common encodings for Kaggle/OSHA-like files
    for enc in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    # last resort: ignore bad characters
    return pd.read_csv(path, encoding="utf-8", errors="replace")

if not DATA_PATH.exists():
    st.error(f"Dataset not found at: {DATA_PATH}\n\nPut severeinjury.csv inside: data/")
    st.stop()

df = load_csv_safely(DATA_PATH)

# -----------------------------
# Column guessing (works even if names differ)
# -----------------------------
def find_col(possible_names):
    lower_cols = {c.lower(): c for c in df.columns}
    for name in possible_names:
        if name.lower() in lower_cols:
            return lower_cols[name.lower()]
    # fuzzy contains
    for c in df.columns:
        cl = c.lower()
        for name in possible_names:
            if name.lower() in cl:
                return c
    return None

industry_col = find_col(["industry", "industry_description", "naics_title", "industry name"])
injury_col   = find_col(["injury", "injury_description", "nature of injury", "injury nature"])
state_col    = find_col(["state", "state_name"])
date_col     = find_col(["date", "incident_date", "event_date", "report_date"])

# -----------------------------
# Sidebar filters (optional)
# -----------------------------
st.sidebar.header("Filters")

df_filtered = df.copy()

if state_col:
    states = sorted(df_filtered[state_col].dropna().astype(str).unique().tolist())
    pick_states = st.sidebar.multiselect("State", states, default=states[:0])
    if pick_states:
        df_filtered = df_filtered[df_filtered[state_col].astype(str).isin(pick_states)]

if industry_col:
    inds = sorted(df_filtered[industry_col].dropna().astype(str).unique().tolist())
    pick_inds = st.sidebar.multiselect("Industry", inds, default=inds[:0])
    if pick_inds:
        df_filtered = df_filtered[df_filtered[industry_col].astype(str).isin(pick_inds)]

st.sidebar.divider()
st.sidebar.write("Rows:", len(df_filtered))

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Overview + KPIs", "Trends", "Risk Score Intelligence"])

# -----------------------------
# Tab 1: Overview + KPIs
# -----------------------------
with tab1:
    c1, c2, c3 = st.columns(3)

    c1.metric("Total Records", f"{len(df_filtered):,}")

    if state_col:
        c2.metric("States covered", f"{df_filtered[state_col].nunique():,}")
    else:
        c2.metric("States covered", "N/A")

    if industry_col:
        c3.metric("Industries covered", f"{df_filtered[industry_col].nunique():,}")
    else:
        c3.metric("Industries covered", "N/A")

    st.divider()

    # Top Industries
    if industry_col:
        st.subheader("Top Industries (by incident count)")
        top_ind = df_filtered[industry_col].value_counts().head(15).reset_index()
        top_ind.columns = ["industry", "count"]
        fig = px.bar(top_ind, x="count", y="industry", orientation="h")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Industry column not detected in this dataset.")

    # Top Injury Types
    if injury_col:
        st.subheader("Top Injury Types (by incident count)")
        top_inj = df_filtered[injury_col].value_counts().head(15).reset_index()
        top_inj.columns = ["injury", "count"]
        fig2 = px.bar(top_inj, x="count", y="injury", orientation="h")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Injury column not detected in this dataset.")

    # Top States
    if state_col:
        st.subheader("Top States (by incident count)")
        st.dataframe(df_filtered[state_col].value_counts().head(15))
    else:
        st.info("State column not detected in this dataset.")

# -----------------------------
# Tab 2: Trends
# -----------------------------
with tab2:
    st.subheader("Time trend")

    if date_col:
        dft = df_filtered.copy()
        dft[date_col] = pd.to_datetime(dft[date_col], errors="coerce")
        dft = dft.dropna(subset=[date_col])

        if len(dft) == 0:
            st.warning("Date column exists but couldn’t be parsed to datetime.")
        else:
            dft["year"] = dft[date_col].dt.year
            trend = dft["year"].value_counts().sort_index()

            st.line_chart(trend)
            st.caption("Yearly incident count (based on parsed date column).")
    else:
        st.info("Date column not detected in this dataset, so trend chart is unavailable.")

# -----------------------------
# Tab 3: Risk Score Intelligence (NEW)
# -----------------------------
with tab3:
    st.subheader("Risk Score Intelligence")
    st.caption("Uses outputs/osha_with_risk_score.csv generated by src/03_risk_score.py")

    if not RISK_PATH.exists():
        st.warning(
            "Risk score file not found.\n\n"
            "Run this first:\n"
            "python src/03_risk_score.py\n\n"
            f"Expected file at: {RISK_PATH}"
        )
    else:
        rdf = load_csv_safely(RISK_PATH)

        if "risk_score" not in rdf.columns:
            st.error("Found the file, but it has no column named 'risk_score'. Check src/03_risk_score.py output.")
        else:
            c1, c2 = st.columns([1, 1])

            with c1:
                st.markdown("### Risk Score Distribution")
                fig = px.histogram(rdf, x="risk_score", nbins=30)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.markdown("### Quick Stats")
                st.metric("Mean risk score", f"{rdf['risk_score'].mean():.2f}")
                st.metric("Max risk score", f"{rdf['risk_score'].max():.2f}")
                st.metric("Records with score", f"{rdf['risk_score'].notna().sum():,}")

            st.divider()

            # High-risk industries (mean risk)
            if "industry" in rdf.columns:
                st.markdown("### Top High-Risk Industries (mean risk score)")
                top_ind = (
                    rdf.groupby("industry")["risk_score"]
                    .mean()
                    .sort_values(ascending=False)
                    .head(10)
                    .reset_index()
                )
                fig2 = px.bar(top_ind, x="risk_score", y="industry", orientation="h")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No 'industry' column found in risk output (expected name: industry).")

            # High-risk injury types (mean risk)
            if "injury" in rdf.columns:
                st.markdown("### Top High-Risk Injury Types (mean risk score)")
                top_inj = (
                    rdf.groupby("injury")["risk_score"]
                    .mean()
                    .sort_values(ascending=False)
                    .head(10)
                    .reset_index()
                )
                fig3 = px.bar(top_inj, x="risk_score", y="injury", orientation="h")
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("No 'injury' column found in risk output (expected name: injury).")

st.success("Dashboard ready.")