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

# Existing Tier-1 risk score (optional)
RISK_PATH = ROOT / "outputs" / "osha_with_risk_score.csv"

# Tier-2 artifacts (created by your src/04 and src/05 scripts)
TIER2_PRED_PATH = ROOT / "outputs" / "tier2_baseline_predictions.csv"
TIER2_MODEL_PATH = ROOT / "outputs" / "tier2_text_model.joblib"  # optional (if you save it)

st.title("OSHA Severe Injury Analytics Dashboard")
st.caption("Interactive safety analytics + risk scoring + Tier-2 NLP baseline from OSHA severe injury records.")

# -----------------------------
# Load data (robust)
# -----------------------------
@st.cache_data
def load_csv_safely(path: Path):
    for enc in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", errors="replace")


if not DATA_PATH.exists():
    st.error(f"Dataset not found at: {DATA_PATH}\n\nPut severeinjury.csv inside: data/")
    st.stop()

df = load_csv_safely(DATA_PATH)

# -----------------------------
# Column guessing (works even if names differ)
# -----------------------------
# NOTE: The OSHA Severe Injury dataset commonly uses columns like:
# EventDate, State, Primary NAICS, NatureTitle, EventTitle, Final Narrative, etc.

def find_col(possible_names):
    lower_cols = {c.lower(): c for c in df.columns}
    for name in possible_names:
        if name.lower() in lower_cols:
            return lower_cols[name.lower()]
    for c in df.columns:
        cl = c.lower()
        for name in possible_names:
            if name.lower() in cl:
                return c
    return None


industry_col = find_col([
    "primary naics", "naics", "naics code", "primary_naics",
    "industry", "industry_description", "naics_title", "industry name"
])

injury_col = find_col([
    "naturetitle", "nature title", "nature", "injury", "injury_description",
    "eventtitle", "event title"
])

state_col = find_col(["state", "state_name"])
date_col  = find_col(["eventdate", "event date", "date", "incident_date", "event_date", "report_date"])

naics_title_col   = find_col(["naics title", "naics_title", "industry title"])
hospital_col      = find_col(["hospitalized", "hospitalised", "hospitalized?"])
amputation_col    = find_col(["amputation", "amputated"])
inspection_col    = find_col(["inspection", "inspected"])
narrative_col     = find_col(["final narrative", "narrative", "description", "incident description"])

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters")
df_filtered = df.copy()

if state_col:
    states = sorted(df_filtered[state_col].dropna().astype(str).unique().tolist())
    pick_states = st.sidebar.multiselect("State", states, default=[])
    if pick_states:
        df_filtered = df_filtered[df_filtered[state_col].astype(str).isin(pick_states)]

if industry_col:
    inds = sorted(df_filtered[industry_col].dropna().astype(str).unique().tolist())
    pick_inds = st.sidebar.multiselect("Industry / NAICS", inds, default=[])
    if pick_inds:
        df_filtered = df_filtered[df_filtered[industry_col].astype(str).isin(pick_inds)]

st.sidebar.divider()
st.sidebar.write("Rows:", len(df_filtered))

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview + KPIs", "Trends", "Risk Score Intelligence", "Tier-2 NLP Model"]
)

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
        c3.metric("Industries/NAICS covered", f"{df_filtered[industry_col].nunique():,}")
    else:
        c3.metric("Industries/NAICS covered", "N/A")

    st.divider()

    # Top Industries / NAICS
    if industry_col:
        label = "Top NAICS (by incident count)" if "naics" in industry_col.lower() else "Top Industries (by incident count)"
        st.subheader(label)

        top_ind = df_filtered[industry_col].astype(str).value_counts().head(15).reset_index()
        top_ind.columns = ["industry", "count"]
        fig = px.bar(top_ind, x="count", y="industry", orientation="h")
        st.plotly_chart(fig, use_container_width=True)

        # severity signals by NAICS (if columns exist)
        if hospital_col or amputation_col:
            st.caption("Quick severity signals by NAICS (based on available fields).")
            tmp = df_filtered.copy()

            def _to01(s):
                s = s.astype(str).str.strip().str.lower()
                return s.isin(["1", "true", "t", "yes", "y"]).astype(int)

            tmp["_hosp"] = _to01(tmp[hospital_col]) if hospital_col else 0
            tmp["_amp"]  = _to01(tmp[amputation_col]) if amputation_col else 0

            sev = (
                tmp.groupby(industry_col)
                .agg(incidents=(industry_col, "size"),
                     hospitalized_rate=("_hosp", "mean"),
                     amputation_rate=("_amp", "mean"))
                .sort_values("incidents", ascending=False)
                .head(10)
                .reset_index()
            )
            sev["hospitalized_rate"] = (sev["hospitalized_rate"] * 100).round(1)
            sev["amputation_rate"]   = (sev["amputation_rate"] * 100).round(1)
            sev = sev.rename(columns={
                industry_col: "naics_or_industry",
                "hospitalized_rate": "% hospitalized",
                "amputation_rate": "% amputations"
            })
            st.dataframe(sev, use_container_width=True)
    else:
        st.info("Industry/NAICS column not detected in this dataset.")

    # Top Injury Types
    if injury_col:
        st.subheader("Top Nature / Injury Types (by incident count)")
        top_inj = df_filtered[injury_col].astype(str).value_counts().head(15).reset_index()
        top_inj.columns = ["injury", "count"]
        fig2 = px.bar(top_inj, x="count", y="injury", orientation="h")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Injury/Nature column not detected in this dataset.")

    # Top States
    if state_col:
        st.subheader("Top States (by incident count)")
        top_states = df_filtered[state_col].astype(str).value_counts().head(15).reset_index()
        top_states.columns = ["state", "count"]
        st.dataframe(top_states, use_container_width=True)
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
# Tab 3: Risk Score Intelligence
# -----------------------------
with tab3:
    st.subheader("Risk Score Intelligence")
    st.caption("If outputs/osha_with_risk_score.csv is missing (common on Streamlit Cloud), the app computes an explainable risk score on the fly.")

    def _to01(series: pd.Series) -> pd.Series:
        s = series.fillna(0).astype(str).str.strip().str.lower()
        return s.isin(["1", "true", "t", "yes", "y"]).astype(int)

    def build_risk_df(source_df: pd.DataFrame) -> pd.DataFrame:
        rdf = source_df.copy()

        rdf["_amp"] = _to01(rdf[amputation_col]) if amputation_col else 0
        rdf["_hosp"] = _to01(rdf[hospital_col]) if hospital_col else 0
        rdf["_insp"] = _to01(rdf[inspection_col]) if inspection_col else 0

        if narrative_col:
            text = rdf[narrative_col].fillna("").astype(str).str.lower()
            kw = {
                "fatal": 5, "death": 5, "amput": 4, "hospital": 3,
                "fractur": 2, "burn": 2, "lacerat": 1, "crush": 2,
                "electr": 2, "fall": 1,
            }
            score_kw = 0
            for k, w in kw.items():
                score_kw = score_kw + text.str.contains(k, regex=False).astype(int) * w
            rdf["_kw"] = score_kw
        else:
            rdf["_kw"] = 0

        rdf["risk_score"] = (1 + 4 * rdf["_amp"] + 3 * rdf["_hosp"] + 1 * rdf["_insp"] + rdf["_kw"]).clip(1, 20)

        if industry_col:
            rdf["industry"] = rdf[industry_col].astype(str)
        if injury_col:
            rdf["injury"] = rdf[injury_col].astype(str)
        if state_col:
            rdf["state"] = rdf[state_col].astype(str)
        if date_col:
            dt = pd.to_datetime(rdf[date_col], errors="coerce")
            rdf["year"] = dt.dt.year

        return rdf

    if RISK_PATH.exists():
        rdf = load_csv_safely(RISK_PATH)
        if "risk_score" not in rdf.columns:
            st.warning("Risk file exists but has no 'risk_score'. Computing live score instead.")
            rdf = build_risk_df(df_filtered)
        else:
            if industry_col and "industry" not in rdf.columns:
                rdf["industry"] = rdf.get(industry_col, "").astype(str)
            if injury_col and "injury" not in rdf.columns:
                rdf["injury"] = rdf.get(injury_col, "").astype(str)
    else:
        st.info("Risk score file not found in repo (normal on Streamlit Cloud). Computing risk score live now.")
        rdf = build_risk_df(df_filtered)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("### Risk Score Distribution")
        fig = px.histogram(rdf, x="risk_score", nbins=20)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("### Quick Stats")
        st.metric("Mean risk score", f"{rdf['risk_score'].mean():.2f}")
        st.metric("Max risk score", f"{rdf['risk_score'].max():.0f}")
        st.metric("Records scored", f"{rdf['risk_score'].notna().sum():,}")

    st.divider()

    if "industry" in rdf.columns:
        st.markdown("### Top High-Risk NAICS / Industries (mean risk score)")
        top_ind = (
            rdf.groupby("industry")["risk_score"]
            .mean()
            .sort_values(ascending=False)
            .head(12)
            .reset_index()
        )
        fig2 = px.bar(top_ind, x="risk_score", y="industry", orientation="h")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No industry/NAICS column detected for risk grouping.")

    if "injury" in rdf.columns:
        st.markdown("### Top High-Risk Injury / Nature Types (mean risk score)")
        top_inj = (
            rdf.groupby("injury")["risk_score"]
            .mean()
            .sort_values(ascending=False)
            .head(12)
            .reset_index()
        )
        fig3 = px.bar(top_inj, x="risk_score", y="injury", orientation="h")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No injury/nature column detected for risk grouping.")

    st.divider()
    st.markdown("### Download")
    st.download_button(
        "Download scored dataset (CSV)",
        data=rdf.to_csv(index=False).encode("utf-8"),
        file_name="osha_scored.csv",
        mime="text/csv",
    )

# -----------------------------
# Tab 4: Tier-2 NLP Baseline (ML)
# -----------------------------
with tab4:
    st.subheader("Tier-2 NLP Model (Baseline)")
    st.caption("Baseline: TF-IDF + Logistic Regression trained in src/05_train_text_model.py")

    if not TIER2_PRED_PATH.exists():
        st.warning(
            "Tier-2 predictions file not found.\n\n"
            "Run:\n"
            "python src/04_make_target.py\n"
            "python src/05_train_text_model.py\n\n"
            f"Expected: {TIER2_PRED_PATH}"
        )
    else:
        pdf = load_csv_safely(TIER2_PRED_PATH)

        # normalize column names (in case you saved slightly different headers)
        ren = {}
        for col in pdf.columns:
            if col.lower() in ["true", "ytrue", "y_true"]:
                ren[col] = "y_true"
            if col.lower() in ["pred", "ypred", "y_pred"]:
                ren[col] = "y_pred"
        if ren:
            pdf = pdf.rename(columns=ren)

        if {"y_true", "y_pred"}.issubset(set(pdf.columns)):
            acc = (pdf["y_true"] == pdf["y_pred"]).mean()
            st.metric("Test accuracy (baseline)", f"{acc:.3f}")

            cm = pd.crosstab(pdf["y_true"], pdf["y_pred"], rownames=["true"], colnames=["pred"])
            st.markdown("### Confusion matrix (counts)")
            st.dataframe(cm, use_container_width=True)

        st.markdown("### Example predictions")
        st.dataframe(pdf.head(30), use_container_width=True)

        st.divider()
        st.markdown("### Try your own text (optional live prediction)")
        st.caption("If you save a joblib model to outputs/tier2_text_model.joblib, we can predict here.")
        user_text = st.text_area("Paste an incident narrative", height=140)

        if user_text.strip() and TIER2_MODEL_PATH.exists():
            try:
                from joblib import load
                model = load(TIER2_MODEL_PATH)
                pred = model.predict([user_text])[0]
                st.success(f"Prediction: {'HIGH severity' if int(pred) == 1 else 'NOT high severity'}")
            except Exception as e:
                st.error(f"Model loaded but prediction failed: {e}")
        elif user_text.strip() and not TIER2_MODEL_PATH.exists():
            st.info("No saved model found yet. (Optional) Save it from src/05_train_text_model.py as outputs/tier2_text_model.joblib.")

st.success("Dashboard ready.")