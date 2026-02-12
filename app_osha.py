import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from pathlib import Path

# Optional (only used if available)
try:
    from joblib import load as joblib_load
except Exception:
    joblib_load = None


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="OSHA Severe Injury Analytics Dashboard", layout="wide")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "severeinjury.csv"

# (Optional) precomputed files (if you committed them OR generated locally)
RISK_PATH = ROOT / "outputs" / "osha_with_risk_score.csv"
TIER2_PRED_PATH = ROOT / "outputs" / "tier2_baseline_predictions.csv"
TIER2_MODEL_PATH = ROOT / "outputs" / "tier2_text_model.joblib"

st.title("OSHA Severe Injury Analytics Dashboard")

st.caption(
    "A practical safety analytics demo: KPIs + simple risk scoring + a text classifier that flags potentially severe incidents."
)

# -----------------------------
# Recruiter-friendly “What is this?” (TOP)
# -----------------------------
with st.container():
    st.info(
        "**What you’re looking at :**\n\n"
        "This app turns OSHA severe injury records into **safety intelligence**.\n"
        "- **Overview & KPIs:** where incidents happen (state / NAICS / injury nature)\n"
        "- **Trends:** how incident counts change over time\n"
        "- **Risk Score Intelligence:** an explainable score (based on hospitalization/amputation + keywords)\n"
        "- **Tier-2 Text Model:** a simple model that reads an incident narrative and predicts whether it looks **high severity**\n\n"
        "**Why it matters:** In real EHS work, this supports **incident triage**, **prioritizing investigations**, "
        "and **early warning** from incoming reports."
    )


# -----------------------------
# Load data (robust)
# -----------------------------
@st.cache_data
def load_csv_safely(path: Path) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", errors="replace")


if not DATA_PATH.exists():
    st.error(f"Dataset not found at: {DATA_PATH}\n\nPut `severeinjury.csv` inside: `data/`")
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
    for c in df.columns:
        cl = c.lower()
        for name in possible_names:
            if name.lower() in cl:
                return c
    return None


# OSHA-ish common columns
industry_col = find_col([
    "primary naics", "naics", "naics code", "primary_naics",
    "industry", "industry_description", "naics_title", "industry name"
])

injury_col = find_col([
    "naturetitle", "nature title", "nature",
    "injury", "injury_description",
    "eventtitle", "event title"
])

state_col = find_col(["state", "state_name"])
date_col = find_col(["eventdate", "event date", "date", "incident_date", "event_date", "report_date"])

# Optional severity / signals
hospital_col = find_col(["hospitalized", "hospitalised", "hospitalized?"])
amputation_col = find_col(["amputation", "amputated"])
inspection_col = find_col(["inspection", "inspected"])
narrative_col = find_col(["final narrative", "narrative", "description", "incident description"])


# -----------------------------
# Helpers
# -----------------------------
def _to01(series: pd.Series) -> pd.Series:
    s = series.fillna(0).astype(str).str.strip().str.lower()
    return s.isin(["1", "true", "t", "yes", "y"]).astype(int)


def build_risk_df(source_df: pd.DataFrame) -> pd.DataFrame:
    """
    Explainable risk score:
    - +4 if amputation
    - +3 if hospitalized
    - +1 if inspection
    - +keyword weights from narrative (if present)
    """
    rdf = source_df.copy()

    rdf["_amp"] = _to01(rdf[amputation_col]) if amputation_col else 0
    rdf["_hosp"] = _to01(rdf[hospital_col]) if hospital_col else 0
    rdf["_insp"] = _to01(rdf[inspection_col]) if inspection_col else 0

    # Keywords from narrative (optional)
    if narrative_col:
        text = rdf[narrative_col].fillna("").astype(str).str.lower()
        kw = {
            "fatal": 5,
            "death": 5,
            "amput": 4,
            "hospital": 3,
            "fractur": 2,
            "burn": 2,
            "lacerat": 1,
            "crush": 2,
            "electr": 2,
            "fall": 1,
        }
        score_kw = 0
        for k, w in kw.items():
            score_kw = score_kw + text.str.contains(k, regex=False).astype(int) * w
        rdf["_kw"] = score_kw
    else:
        rdf["_kw"] = 0

    rdf["risk_score"] = (1 + 4 * rdf["_amp"] + 3 * rdf["_hosp"] + 1 * rdf["_insp"] + rdf["_kw"]).clip(1, 20)

    # Standard names for grouping
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


def safe_unique_count(s: pd.Series) -> int:
    try:
        return int(s.dropna().nunique())
    except Exception:
        return 0


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

with st.sidebar.expander("Glossary "):
    st.markdown(
        "- **NAICS**: an industry classification code\n"
        "- **Nature / Injury type**: what kind of injury happened\n"
        "- **Risk score**: a simple number (1–20) estimating severity using clear rules\n"
        "- **Text model**: a basic classifier that reads a narrative and predicts if it looks high severity\n"
        "- **Confusion matrix**: shows correct vs wrong predictions\n"
        "- **Accuracy**: percent of total predictions that were correct (can look high even if classes are imbalanced)"
    )


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview + KPIs", "Trends", "Risk Score Intelligence", "Tier-2 Text Model"]
)


# -----------------------------
# Tab 1: Overview + KPIs
# -----------------------------
with tab1:
    st.subheader("Overview + KPIs (What’s happening, where?)")
    st.caption("This section answers: **Where are incidents happening and what types are common?**")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Records", f"{len(df_filtered):,}")
    c2.metric("States covered", f"{safe_unique_count(df_filtered[state_col]) if state_col else 'N/A'}")
    c3.metric("Industries / NAICS covered", f"{safe_unique_count(df_filtered[industry_col]) if industry_col else 'N/A'}")

    st.divider()

    # Top industries/NAICS
    if industry_col:
        label = "Top NAICS (by incident count)" if "naics" in industry_col.lower() else "Top Industries (by incident count)"
        st.markdown(f"### {label}")
        top_ind = df_filtered[industry_col].astype(str).value_counts().head(15).reset_index()
        top_ind.columns = ["industry", "count"]
        fig = px.bar(top_ind, x="count", y="industry", orientation="h")
        st.plotly_chart(fig, use_container_width=True)

        # severity signals by NAICS (if columns exist)
        if hospital_col or amputation_col:
            st.caption("Extra: severity signals by NAICS (based on available fields).")
            tmp = df_filtered.copy()
            tmp["_hosp"] = _to01(tmp[hospital_col]) if hospital_col else 0
            tmp["_amp"] = _to01(tmp[amputation_col]) if amputation_col else 0
            sev = (
                tmp.groupby(industry_col)
                .agg(
                    incidents=(industry_col, "size"),
                    hospitalized_rate=("_hosp", "mean"),
                    amputation_rate=("_amp", "mean"),
                )
                .sort_values("incidents", ascending=False)
                .head(10)
                .reset_index()
            )
            sev["hospitalized_rate"] = (sev["hospitalized_rate"] * 100).round(1)
            sev["amputation_rate"] = (sev["amputation_rate"] * 100).round(1)
            sev = sev.rename(
                columns={
                    industry_col: "naics_or_industry",
                    "hospitalized_rate": "% hospitalized",
                    "amputation_rate": "% amputations",
                }
            )
            st.dataframe(sev, use_container_width=True)
    else:
        st.info("Industry/NAICS column not detected in this dataset.")

    # Top injury types
    if injury_col:
        st.markdown("### Top Injury / Nature Types (by incident count)")
        top_inj = df_filtered[injury_col].astype(str).value_counts().head(15).reset_index()
        top_inj.columns = ["injury", "count"]
        fig2 = px.bar(top_inj, x="count", y="injury", orientation="h")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Injury/Nature column not detected in this dataset.")

    # Top states
    if state_col:
        st.markdown("### Top States (by incident count)")
        top_states = df_filtered[state_col].astype(str).value_counts().head(15).reset_index()
        top_states.columns = ["state", "count"]
        st.dataframe(top_states, use_container_width=True)
    else:
        st.info("State column not detected in this dataset.")


# -----------------------------
# Tab 2: Trends
# -----------------------------
with tab2:
    st.subheader("Trends (How does it change over time?)")
    st.caption("This section answers: **Are incidents increasing or decreasing year-by-year?**")

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
        st.info("Date column not detected, so trend chart is unavailable.")


# -----------------------------
# Tab 3: Risk Score Intelligence
# -----------------------------
with tab3:
    st.subheader("Risk Score Intelligence (Explainable severity signals)")
    st.caption(
        "This is **not** a black-box model. It’s an **explainable score** using fields like hospitalization/amputation + keywords."
    )

    with st.expander("What is the risk score ?", expanded=True):
        st.markdown(
            "We compute a simple score from **1 to 20**:\n\n"
            "- **+4** if the record indicates **amputation**\n"
            "- **+3** if it indicates **hospitalization**\n"
            "- **+1** if there was an **inspection** (if available)\n"
            "- Plus a **keyword signal** from narrative text (if available)\n\n"
            "**Use case:** Helps EHS teams quickly prioritize which incidents may need faster escalation."
        )

    # Prefer precomputed CSV, otherwise compute live
    if RISK_PATH.exists():
        rdf = load_csv_safely(RISK_PATH)
        if "risk_score" not in rdf.columns:
            st.warning("Risk file exists but has no 'risk_score' column. Computing risk score live instead.")
            rdf = build_risk_df(df_filtered)
        else:
            # normalize columns for grouping
            if industry_col and "industry" not in rdf.columns and industry_col in rdf.columns:
                rdf["industry"] = rdf[industry_col].astype(str)
            if injury_col and "injury" not in rdf.columns and injury_col in rdf.columns:
                rdf["injury"] = rdf[injury_col].astype(str)
    else:
        st.info("No precomputed risk file found in repo (normal on Streamlit Cloud). Computing risk score live now.")
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
# Tab 4: Tier-2 Text Model
# -----------------------------
with tab4:
    st.subheader("Tier-2 Text Model (Severity flag from narrative text)")
    st.caption("This section demonstrates a practical workflow: **incident narrative → severity flag**.")

    with st.expander("What does this model do ?", expanded=True):
        st.markdown(
            "Many companies receive incident reports as **free text** (narratives). "
            "This model reads that text and predicts whether the incident *looks like high severity*.\n\n"
            "**Use case examples:**\n"
            "- Auto-flag severe cases for faster investigation\n"
            "- Help safety teams triage large volumes of reports\n"
            "- Create early-warning signals from narrative descriptions\n\n"
            "**Important:** This is a baseline demo model. In real deployment, it needs careful validation and governance."
        )

    # Show evaluation results from predictions file (works on Streamlit Cloud if you committed outputs/* OR created them)
    if not TIER2_PRED_PATH.exists():
        st.warning(
            "Tier-2 predictions file not found.\n\n"
            "**If you are running locally:** generate it using:\n"
            "`python src/04_make_target.py`\n"
            "`python src/05_train_text_model.py`\n\n"
            "**If you are on Streamlit Cloud:** commit `outputs/tier2_baseline_predictions.csv` to the repo "
            "(or keep the on-the-fly demo only)."
        )
    else:
        pred_df = load_csv_safely(TIER2_PRED_PATH)

        required_cols = {"text", "y_true", "y_pred"}
        if not required_cols.issubset(set(pred_df.columns)):
            st.error(
                f"Predictions file exists but is missing required columns: {required_cols}. "
                f"Found: {list(pred_df.columns)}"
            )
        else:
            # Compute metrics in-app (no ML knowledge required)
            y_true = pred_df["y_true"].astype(int).values
            y_pred = pred_df["y_pred"].astype(int).values

            acc = float((y_true == y_pred).mean())

            # Confusion matrix counts
            # rows=true, cols=pred
            tn = int(((y_true == 0) & (y_pred == 0)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())
            tp = int(((y_true == 1) & (y_pred == 1)).sum())

            st.markdown("### Model performance ")
            st.metric("Test accuracy (baseline)", f"{acc:.3f}")

            with st.expander("How to read these numbers "):
                st.markdown(
                    "- **Accuracy**: % of all predictions that were correct.\n"
                    "- **TP (true positives)**: model correctly flagged high severity.\n"
                    "- **FN (false negatives)**: model missed a high severity case (risky in real safety workflows).\n"
                    "- **FP (false positives)**: model flagged something as high severity but it wasn’t.\n\n"
                    "**EHS note:** In many safety systems, reducing **false negatives** is often more important than maximizing accuracy."
                )

            st.markdown("### Confusion matrix (counts)")
            cm_df = pd.DataFrame(
                [[tn, fp], [fn, tp]],
                index=["true: 0 (not high)", "true: 1 (high)"],
                columns=["pred: 0", "pred: 1"],
            )
            st.dataframe(cm_df, use_container_width=True)

            st.markdown("### Example predictions")
            show_n = min(15, len(pred_df))
            st.dataframe(pred_df.sample(show_n, random_state=42), use_container_width=True)

    st.divider()
    st.markdown("### Try your own text (live prediction)")

    st.caption(
        "Type a short incident narrative. The app will predict whether it looks **high severity**. "
        "Tip: include details like hospitalization/amputation/fall/crush/electrical to see how the model reacts."
    )

    example_col1, example_col2 = st.columns(2)
    with example_col1:
        st.markdown("**Example (high severity):**")
        st.code("Employee’s hand was caught in a hydraulic press causing finger amputation and emergency hospitalization.")
    with example_col2:
        st.markdown("**Example (not high severity):**")
        st.code("Employee slipped and had a minor bruise, returned to work the same day.")

    user_text = st.text_area("Paste an incident narrative", height=110)

    # If a saved model exists, use it. Otherwise, explain how to create it.
    if not TIER2_MODEL_PATH.exists():
        st.info(
            "No saved model found yet.\n\n"
            "To enable live prediction:\n"
            "- Run locally: `python src/05_train_text_model.py`\n"
            "- Ensure it saves: `outputs/tier2_text_model.joblib`\n"
            "- Commit/push that file if you want it on Streamlit Cloud (optional).\n\n"
            "**Good news:** Your dashboard still demonstrates the workflow using the evaluation outputs."
        )
    else:
        if joblib_load is None:
            st.error("joblib is not available in this environment. Add `joblib` to requirements.txt.")
        else:
            try:
                model = joblib_load(TIER2_MODEL_PATH)
                if user_text and user_text.strip():
                    pred = int(model.predict([user_text.strip()])[0])
                    proba = None
                    try:
                        if hasattr(model, "predict_proba"):
                            proba = float(model.predict_proba([user_text.strip()])[0][1])
                    except Exception:
                        proba = None

                    if pred == 1:
                        st.error("Prediction: HIGH severity (flag for fast review)")
                    else:
                        st.success("Prediction: NOT high severity")

                    if proba is not None:
                        st.caption(f"Model confidence (approx): {proba:.2f} probability of high severity")

                    st.caption(
                        "Interpretation: This is a triage signal — it helps prioritize review, "
                        "but final decisions should be made by safety professionals."
                    )
            except Exception as e:
                st.error(f"Could not load/use the model file: {e}")


st.success("Dashboard ready.")