# app_osha.py
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

# Tier-2 NLP artifacts (from src/04_make_target.py + src/05_train_text_model.py)
TIER2_PRED_PATH = ROOT / "outputs" / "tier2_baseline_predictions.csv"
TIER2_MODEL_PATH = ROOT / "outputs" / "tier2_text_model.joblib"

# A4: Structured ML artifacts (your structured model)
STRUCT_MODEL_PATH = ROOT / "outputs" / "tier2_structured_model.joblib"

# B: Anomaly artifacts (from src/07_anomaly_model.py)
ANOMALY_SCORES_PATH = ROOT / "outputs" / "anomaly_scores.csv"
ANOMALY_MODEL_PATH = ROOT / "outputs" / "anomaly_model.joblib"

# 8: Cost + impact artifacts (from src/08_cost_model.py)
COST_EST_PATH = ROOT / "outputs" / "cost_estimates.csv"

# 9: Workflow queue artifacts (from src/09_workflow_sim.py)
WORKFLOW_PATH = ROOT / "outputs" / "workflow_queue.csv"

st.title("OSHA Severe Injury Analytics Dashboard")
st.caption(
    "A safety analytics demo: KPIs + explainable risk scoring + text-based severity flagging + structured ML prediction + anomaly detection."
)

# -----------------------------
# Quick intro (keep short, recruiter-readable)
# -----------------------------
with st.container():
    st.info(
        "**What this app does**\n\n"
        "- **Overview & KPIs:** where incidents happen (state / NAICS / injury nature)\n"
        "- **Trends:** how incident volume changes over time\n"
        "- **Risk Score Intelligence:** transparent scoring using hospitalization/amputation + narrative keywords\n"
        "- **Tier-2 Text Model:** predicts likely high severity from incident narrative text\n"
        "- **Structured ML :** predicts severity from structured fields (state/NAICS/nature + event/source titles + flags)\n"
        "- **Anomaly Detection (B):** flags unusual records (rare combinations / outliers) for audit and investigation\n"
        "- **Decision Layer:** converts signals into priority + action + SLA + owner\n"
        "- **Cost & Impact :** estimates cost burden, transparent assumptions)\n"
        "- **Workflow Queue :** turns incidents into an EHS work queue \n\n"
        "Focus: practical EHS workflows like triage, prioritization, and early warning."
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

# A4 fix: titles used in structured model
event_title_col = find_col(["eventtitle", "event title", "event_title"])
source_title_col = find_col(["sourcetitle", "source title", "source_title"])

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


def _infer_required_cols(model_obj):
    """
    Best-effort: detect which dataframe columns the trained pipeline expects.
    Works for many sklearn pipelines (feature_names_in_).
    """
    cols = getattr(model_obj, "feature_names_in_", None)
    if cols is not None:
        return list(cols)

    named_steps = getattr(model_obj, "named_steps", None)
    if isinstance(named_steps, dict):
        for step in named_steps.values():
            cols = getattr(step, "feature_names_in_", None)
            if cols is not None:
                return list(cols)

    guess = []
    for c in [
        state_col, industry_col, injury_col, event_title_col, source_title_col,
        hospital_col, amputation_col, inspection_col
    ]:
        if c:
            guess.append(c)

    for c in ["EventTitle", "SourceTitle"]:
        if c not in guess:
            guess.append(c)

    return guess


# -----------------------------
# Decision Layer (EHS Triage)
# -----------------------------
def decision_layer(
    risk_score=None,
    structured_p=None,
    text_p=None,
    anomaly_score=None,
    hospitalized=0,
    amputation=0,
    inspection=0,
):
    """
    Returns: (priority, action, sla, owner, reasons_list)
    Priority: P1 (highest) ... P4 (lowest)
    """
    # normalize
    risk_score = float(risk_score) if risk_score is not None and not pd.isna(risk_score) else None
    structured_p = float(structured_p) if structured_p is not None and not pd.isna(structured_p) else None
    text_p = float(text_p) if text_p is not None and not pd.isna(text_p) else None
    anomaly_score = float(anomaly_score) if anomaly_score is not None and not pd.isna(anomaly_score) else None

    hosp = int(hospitalized) if hospitalized is not None else 0
    amp = int(amputation) if amputation is not None else 0
    insp = int(inspection) if inspection is not None else 0

    reasons = []

    # Direct severity indicators
    if amp == 1:
        reasons.append("Amputation indicated")
    if hosp == 1:
        reasons.append("Hospitalization indicated")
    if insp == 1:
        reasons.append("Inspection involved")

    # Risk score tiers
    if risk_score is not None:
        if risk_score >= 14:
            reasons.append("High risk score")
        elif risk_score >= 9:
            reasons.append("Medium risk score")

    # Model signals (optional)
    model_signal = 0
    if structured_p is not None and structured_p >= 0.70:
        model_signal += 1
        reasons.append("Structured model flags high severity")
    if text_p is not None and text_p >= 0.70:
        model_signal += 1
        reasons.append("Text model flags high severity")

    # Anomaly signal (optional)
    is_anomaly = False
    if anomaly_score is not None and anomaly_score >= 0.80:
        is_anomaly = True
        reasons.append("Unusual / outlier pattern")

    # Priority rules (simple + defensible)
    if amp == 1 or (hosp == 1 and (risk_score is not None and risk_score >= 12)) or (model_signal >= 2):
        priority = "P1"
        action = "Immediate investigation; isolate hazard; notify management; preserve evidence; start incident review"
        sla = "0–4 hours"
        owner = "EHS Lead + Site Supervisor"
    elif hosp == 1 or (risk_score is not None and risk_score >= 12) or (is_anomaly and (risk_score is not None and risk_score >= 9)) or (model_signal == 1):
        priority = "P2"
        action = "Fast review; collect supervisor statement; verify outcome; launch corrective actions"
        sla = "Within 24 hours"
        owner = "EHS Specialist / Coordinator"
    elif (risk_score is not None and risk_score >= 9) or is_anomaly:
        priority = "P3"
        action = "Add to audit queue; inspect work area; verify controls/training; trend monitoring"
        sla = "Within 7 days"
        owner = "Supervisor + EHS (weekly review)"
    else:
        priority = "P4"
        action = "Record & monitor; include in monthly trends; no immediate escalation needed"
        sla = "Monthly review"
        owner = "Supervisor (with EHS oversight)"

    return priority, action, sla, owner, reasons


def decision_badge(priority: str):
    if priority == "P1":
        st.error("Priority P1 — Immediate")
    elif priority == "P2":
        st.warning("Priority P2 — 24h")
    elif priority == "P3":
        st.info("Priority P3 — 7 days")
    else:
        st.success("Priority P4 — Monitor")


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

with st.sidebar.expander("Glossary"):
    st.markdown(
        "- **NAICS**: industry classification code\n"
        "- **Nature / Injury type**: injury category in the dataset\n"
        "- **Risk score**: a 1–20 score from transparent rules (not a black-box)\n"
        "- **Text model**: classifier using incident narratives\n"
        "- **Confusion matrix**: correct vs incorrect prediction counts\n"
        "- **Accuracy**: overall correctness (can look high when classes are imbalanced)\n"
        "- **Structured ML**: model trained on fields like NAICS/state/nature + titles + flags\n"
        "- **Anomaly detection**: highlights unusual records (outliers) for review\n"
        "- **Decision layer**: turns signals into priority + action + SLA + owner\n"
        "- **Cost & impact **: estimates cost burden \n"
        "- **Workflow queue **: converts incidents into an EHS work queue"
    )

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    [
        "Overview + KPIs",
        "Trends",
        "Risk Score Intelligence",
        "Tier-2 Text Model",
        "Structured ML ",
        "Anomaly Detection (B)",
        "Decision Layer",
        "Cost & Impact ",
        "Workflow Queue ",
    ]
)

# -----------------------------
# Tab 1: Overview + KPIs
# -----------------------------
with tab1:
    st.subheader("Overview + KPIs")
    st.caption("Where incidents happen and which categories appear most often.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", f"{len(df_filtered):,}")
    c2.metric("States covered", f"{safe_unique_count(df_filtered[state_col]) if state_col else 'N/A'}")
    c3.metric("Industries / NAICS covered", f"{safe_unique_count(df_filtered[industry_col]) if industry_col else 'N/A'}")

    st.divider()

    if industry_col:
        label = "Top NAICS (by incident count)" if "naics" in industry_col.lower() else "Top Industries (by incident count)"
        st.markdown(f"### {label}")
        top_ind = df_filtered[industry_col].astype(str).value_counts().head(15).reset_index()
        top_ind.columns = ["industry", "count"]
        fig = px.bar(top_ind, x="count", y="industry", orientation="h")
        st.plotly_chart(fig, use_container_width=True)

        if hospital_col or amputation_col:
            st.caption("Severity signals by NAICS (based on available fields).")
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

    if injury_col:
        st.markdown("### Top Injury / Nature Types (by incident count)")
        top_inj = df_filtered[injury_col].astype(str).value_counts().head(15).reset_index()
        top_inj.columns = ["injury", "count"]
        fig2 = px.bar(top_inj, x="count", y="injury", orientation="h")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Injury/Nature column not detected in this dataset.")

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
    st.subheader("Trends")
    st.caption("Year-by-year incident count (based on the event date field).")

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
    else:
        st.info("Date column not detected, so the trend chart is unavailable.")

# -----------------------------
# Tab 3: Risk Score Intelligence
# -----------------------------
with tab3:
    st.subheader("Risk Score Intelligence")
    st.caption("A transparent score based on hospitalization/amputation + narrative keyword signals.")

    with st.expander("Risk score rules", expanded=True):
        st.markdown(
            "- **+4** if **amputation**\n"
            "- **+3** if **hospitalization**\n"
            "- **+1** if **inspection** (if available)\n"
            "- Plus a **keyword signal** from narrative text (if available)\n"
            "- Final score is clipped to **1–20**"
        )

    if RISK_PATH.exists():
        rdf = load_csv_safely(RISK_PATH)
        if "risk_score" not in rdf.columns:
            st.warning("Risk file exists but has no 'risk_score' column. Computing live score instead.")
            rdf = build_risk_df(df_filtered)
        else:
            if industry_col and "industry" not in rdf.columns and industry_col in rdf.columns:
                rdf["industry"] = rdf[industry_col].astype(str)
            if injury_col and "injury" not in rdf.columns and injury_col in rdf.columns:
                rdf["injury"] = rdf[injury_col].astype(str)
    else:
        st.info("No precomputed risk file found in repo. Computing risk score live now.")
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
    st.subheader("Tier-2 Text Model")
    st.caption("Incident narrative → predicted severity flag (baseline).")

    if not TIER2_PRED_PATH.exists():
        st.warning(
            "Tier-2 predictions file not found.\n\n"
            "Generate locally:\n"
            "- `python src/04_make_target.py`\n"
            "- `python src/05_train_text_model.py`\n\n"
            "If you want it visible on Streamlit Cloud, commit `outputs/tier2_baseline_predictions.csv`."
        )
    else:
        pred_df = load_csv_safely(TIER2_PRED_PATH)

        required_cols = {"text", "y_true", "y_pred"}
        if not required_cols.issubset(set(pred_df.columns)):
            st.error(f"Predictions file is missing required columns: {required_cols}. Found: {list(pred_df.columns)}")
        else:
            y_true = pred_df["y_true"].astype(int).values
            y_pred = pred_df["y_pred"].astype(int).values
            acc = float((y_true == y_pred).mean())

            tn = int(((y_true == 0) & (y_pred == 0)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())
            tp = int(((y_true == 1) & (y_pred == 1)).sum())

            st.markdown("### Performance")
            st.metric("Test accuracy", f"{acc:.3f}")

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
    st.markdown("### Live prediction (text)")

    user_text = st.text_area("Paste an incident narrative", height=110)

    if not TIER2_MODEL_PATH.exists():
        st.info(
            "Model file not found: `outputs/tier2_text_model.joblib`.\n\n"
            "To enable live prediction, run `python src/05_train_text_model.py` and ensure it saves the joblib file."
        )
    else:
        if joblib_load is None:
            st.error("joblib is not available. Add `joblib` to requirements.txt.")
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
                        st.error("Prediction: HIGH severity")
                    else:
                        st.success("Prediction: Lower severity")

                    if proba is not None:
                        st.caption(f"Probability(high severity) ≈ {proba:.2f}")
            except Exception as e:
                st.error(f"Could not load/use the model file: {e}")

# -----------------------------
# Tab 5: Structured ML (A4)
# -----------------------------
with tab5:
    st.subheader("Structured ML ")
    st.caption("Structured fields → predicted severity (uses your saved structured model).")

    st.markdown("### Live prediction")

    if not STRUCT_MODEL_PATH.exists():
        st.info(
            "Structured model file not found: `outputs/tier2_structured_model.joblib`.\n\n"
            "Train/export your structured model and save it with that filename to enable this tab."
        )
    else:
        if joblib_load is None:
            st.error("joblib is not available. Add `joblib` to requirements.txt.")
        else:
            try:
                model = joblib_load(STRUCT_MODEL_PATH)
                required_cols = _infer_required_cols(model)

                input_row = {}

                if state_col:
                    options = sorted(df[state_col].dropna().astype(str).unique().tolist())
                    input_row[state_col] = st.selectbox("State", options)

                if industry_col:
                    options = sorted(df[industry_col].dropna().astype(str).unique().tolist())
                    input_row[industry_col] = st.selectbox("NAICS / Industry", options)

                if injury_col:
                    options = sorted(df[injury_col].dropna().astype(str).unique().tolist())
                    input_row[injury_col] = st.selectbox("Nature / Injury type", options)

                if event_title_col:
                    options = sorted(df[event_title_col].dropna().astype(str).unique().tolist())
                    input_row[event_title_col] = st.selectbox("Event title", options)
                else:
                    input_row["EventTitle"] = st.text_input("Event title", value="")

                if source_title_col:
                    options = sorted(df[source_title_col].dropna().astype(str).unique().tolist())
                    input_row[source_title_col] = st.selectbox("Source title", options)
                else:
                    input_row["SourceTitle"] = st.text_input("Source title", value="")

                if hospital_col:
                    input_row[hospital_col] = 1 if st.checkbox("Hospitalized") else 0
                if amputation_col:
                    input_row[amputation_col] = 1 if st.checkbox("Amputation") else 0
                if inspection_col:
                    input_row[inspection_col] = 1 if st.checkbox("Inspection") else 0

                if st.button("Predict severity (structured)"):
                    x = pd.DataFrame([input_row])

                    for col in required_cols:
                        if col not in x.columns:
                            if any(k in col.lower() for k in ["hosp", "amput", "inspect", "hospital", "amputation"]):
                                x[col] = 0
                            else:
                                x[col] = ""

                    try:
                        x = x[required_cols]
                    except Exception:
                        pass

                    try:
                        p = float(model.predict_proba(x)[0][1])
                        pred = int(p >= 0.5)
                        label = "High severity" if pred == 1 else "Lower severity"
                        st.success(f"{label} (p={p:.2f})")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

            except Exception as e:
                st.error(f"Could not load/use the structured model file: {e}")

# -----------------------------
# Tab 6: Anomaly Detection (B)
# -----------------------------
with tab6:
    st.subheader("Anomaly Detection (B)")
    st.caption("Flags unusual records (outliers) for audit, data quality checks, and investigation prioritization.")

    if ANOMALY_SCORES_PATH.exists():
        adf = load_csv_safely(ANOMALY_SCORES_PATH)
        if "anomaly_score" not in adf.columns:
            st.error("Found anomaly_scores.csv but it has no `anomaly_score` column.")
        else:
            st.markdown("### Overview")
            st.metric("Rows in anomaly file", f"{len(adf):,}")

            if "anomaly_flag" in adf.columns:
                st.metric("Flagged anomalies", f"{int(adf['anomaly_flag'].sum()):,}")

            st.divider()

            st.markdown("### Anomaly score distribution")
            fig = px.histogram(adf, x="anomaly_score", nbins=30)
            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            st.markdown("### Top flagged records")
            cols_to_show = []
            for c in [state_col, industry_col, injury_col, event_title_col, source_title_col, hospital_col, amputation_col, inspection_col, narrative_col]:
                if c and c in adf.columns:
                    cols_to_show.append(c)

            for c in ["anomaly_score", "anomaly_flag", "anomaly_rank", "combo_count", "combo_rarity_index", "review_hints"]:
                if c in adf.columns and c not in cols_to_show:
                    cols_to_show.append(c)

            top_n = st.slider("Show top N anomalies", min_value=10, max_value=200, value=25, step=5)
            view = adf.sort_values("anomaly_score", ascending=False).head(top_n)

            if cols_to_show:
                st.dataframe(view[cols_to_show], use_container_width=True)
            else:
                st.dataframe(view, use_container_width=True)

            st.divider()
            st.markdown("### Download")
            st.download_button(
                "Download anomaly scores (CSV)",
                data=adf.to_csv(index=False).encode("utf-8"),
                file_name="anomaly_scores.csv",
                mime="text/csv",
            )
    else:
        st.warning(
            "No anomaly file found yet.\n\n"
            "Run locally:\n"
            "`python src/07_anomaly_model.py`\n\n"
            "This will create:\n"
            "- `outputs/anomaly_scores.csv`\n"
            "- `outputs/anomaly_model.joblib` (optional)\n\n"
            "If you want this tab to work on Streamlit Cloud, commit `outputs/anomaly_scores.csv` to your repo."
        )

# -----------------------------
# Tab 7: Decision Layer
# -----------------------------
with tab7:
    st.subheader("Decision Layer")
    st.caption("Converts signals into priority + recommended action + SLA + owner.")

    recruiter_mode = st.toggle("Explanation mode ", value=True)

    if recruiter_mode:
        st.info(
            "**What this is :**\n\n"
            "This tab turns safety signals into a real EHS decision:\n"
            "**Priority (P1–P4) + Recommended action + SLA + Owner**.\n\n"
            "Think of it like a *triage desk* for incidents: what needs attention now vs later."
        )

        with st.expander("Why anomaly detection matters (non-technical)", expanded=True):
            st.markdown(
                "- **Catches rare/odd cases** that may hide a serious hazard\n"
                "- **Finds data issues** (wrong NAICS/state/category) before reporting\n"
                "- **Helps audits**: tells the team where to investigate first"
            )
    else:
        st.info("Decision Layer converts signals (risk score + model outputs + anomaly score) into priority, action, SLA and owner.")

    st.markdown("### Live triage")

    # Demo presets
    st.markdown("#### Quick demos (one click)")
    d1, d2, d3 = st.columns(3)
    if d1.button("Severe case (P1)"):
        st.session_state["demo_case"] = "P1"
    if d2.button("Investigate fast (P2)"):
        st.session_state["demo_case"] = "P2"
    if d3.button("Audit unusual (P3)"):
        st.session_state["demo_case"] = "P3"

    demo = st.session_state.get("demo_case", None)

    # defaults
    _default_rs = 8
    _default_hosp = False
    _default_amp = False
    _default_insp = False
    _default_sp = 0.35
    _default_tp = 0.30
    _default_anom = 0.20

    if demo == "P1":
        _default_rs = 16
        _default_hosp = True
        _default_amp = True
        _default_sp = 0.85
        _default_tp = 0.80
        _default_anom = 0.30
    elif demo == "P2":
        _default_rs = 12
        _default_hosp = True
        _default_amp = False
        _default_sp = 0.75
        _default_tp = 0.40
        _default_anom = 0.25
    elif demo == "P3":
        _default_rs = 7
        _default_hosp = False
        _default_amp = False
        _default_sp = 0.20
        _default_tp = 0.20
        _default_anom = 0.90

    c1, c2, c3 = st.columns(3)
    with c1:
        rs = st.slider("Risk score (1–20)", 1, 20, _default_rs)
        hosp = 1 if st.checkbox("Hospitalized", value=_default_hosp, key="dl_hosp") else 0
        amp = 1 if st.checkbox("Amputation", value=_default_amp, key="dl_amp") else 0
        insp = 1 if st.checkbox("Inspection", value=_default_insp, key="dl_insp") else 0

    with c2:
        sp_label = "Model confidence (structured)" if recruiter_mode else "Structured severity probability"
        tp_label = "Model confidence (text)" if recruiter_mode else "Text severity probability"
        st_p = st.slider(sp_label, 0.0, 1.0, _default_sp, 0.01)
        tx_p = st.slider(tp_label, 0.0, 1.0, _default_tp, 0.01)

    with c3:
        an_label = "Unusual pattern score" if recruiter_mode else "Anomaly score"
        anom = st.slider(an_label, 0.0, 1.0, _default_anom, 0.01)
        if recruiter_mode:
            st.caption("Higher = more unusual. Useful for audits + catching weird patterns early.")
        else:
            st.write("Higher = more unusual")

    priority, action, sla, owner, reasons = decision_layer(
        risk_score=rs,
        structured_p=st_p,
        text_p=tx_p,
        anomaly_score=anom,
        hospitalized=hosp,
        amputation=amp,
        inspection=insp,
    )

    decision_badge(priority)

    st.markdown("### Recommended action")
    st.write(action)

    st.markdown("### Operational details")
    st.write(f"**SLA:** {sla}")
    st.write(f"**Owner:** {owner}")

    if reasons:
        st.markdown("### Why it was flagged")
        if recruiter_mode:
            mapped = []
            for r in reasons:
                if r == "Structured model flags high severity":
                    mapped.append("The model has seen similar cases that often become severe")
                elif r == "Text model flags high severity":
                    mapped.append("The narrative text looks similar to severe incidents")
                elif r == "Unusual / outlier pattern":
                    mapped.append("Rare combination — could be a hidden hazard or data issue")
                else:
                    mapped.append(r)
            for r in mapped:
                st.write(f"- {r}")
        else:
            for r in reasons:
                st.write(f"- {r}")

    st.divider()
    st.markdown("### Apply decision layer to the dataset (top 50)")
    st.caption("This view helps a safety team decide what to review first.")

    rdf_local = build_risk_df(df_filtered)

    if ANOMALY_SCORES_PATH.exists():
        try:
            adf = load_csv_safely(ANOMALY_SCORES_PATH)
            if "anomaly_score" in adf.columns and len(adf) == len(rdf_local):
                rdf_local["anomaly_score"] = adf["anomaly_score"].values
        except Exception:
            pass

    structured_model = None
    text_model = None

    if joblib_load is not None and STRUCT_MODEL_PATH.exists():
        try:
            structured_model = joblib_load(STRUCT_MODEL_PATH)
        except Exception:
            structured_model = None

    if joblib_load is not None and TIER2_MODEL_PATH.exists():
        try:
            text_model = joblib_load(TIER2_MODEL_PATH)
        except Exception:
            text_model = None

    if structured_model is not None:
        req_cols = _infer_required_cols(structured_model)
        x_struct = df_filtered.copy()
        for col in req_cols:
            if col not in x_struct.columns:
                if any(k in col.lower() for k in ["hosp", "amput", "inspect", "hospital", "amputation"]):
                    x_struct[col] = 0
                else:
                    x_struct[col] = ""
        try:
            x_struct = x_struct[req_cols]
            rdf_local["structured_p"] = structured_model.predict_proba(x_struct)[:, 1]
        except Exception:
            pass

    if text_model is not None and narrative_col and narrative_col in df_filtered.columns:
        try:
            texts = df_filtered[narrative_col].fillna("").astype(str).values
            if hasattr(text_model, "predict_proba"):
                rdf_local["text_p"] = text_model.predict_proba(texts)[:, 1]
        except Exception:
            pass

    if hospital_col and hospital_col in df_filtered.columns:
        rdf_local["_hosp_flag"] = _to01(df_filtered[hospital_col])
    else:
        rdf_local["_hosp_flag"] = 0

    if amputation_col and amputation_col in df_filtered.columns:
        rdf_local["_amp_flag"] = _to01(df_filtered[amputation_col])
    else:
        rdf_local["_amp_flag"] = 0

    if inspection_col and inspection_col in df_filtered.columns:
        rdf_local["_insp_flag"] = _to01(df_filtered[inspection_col])
    else:
        rdf_local["_insp_flag"] = 0

    def _row_decision(row):
        p, act, sla2, own, reas = decision_layer(
            risk_score=row.get("risk_score", None),
            structured_p=row.get("structured_p", None),
            text_p=row.get("text_p", None),
            anomaly_score=row.get("anomaly_score", None),
            hospitalized=row.get("_hosp_flag", 0),
            amputation=row.get("_amp_flag", 0),
            inspection=row.get("_insp_flag", 0),
        )
        return pd.Series({"priority": p, "sla": sla2, "owner": own})

    dec_cols = rdf_local.apply(_row_decision, axis=1)
    out_view = pd.concat([rdf_local, dec_cols], axis=1)

    priority_order = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}
    out_view["_p_rank"] = out_view["priority"].map(priority_order).fillna(9).astype(int)
    out_view = out_view.sort_values(["_p_rank", "risk_score"], ascending=[True, False]).head(50)

    show_cols = []
    for c in [state_col, industry_col, injury_col, event_title_col, source_title_col]:
        if c and c in df_filtered.columns:
            show_cols.append(c)

    for c in ["risk_score", "anomaly_score", "structured_p", "text_p", "priority", "sla", "owner"]:
        if c in out_view.columns:
            show_cols.append(c)

    st.dataframe(out_view[show_cols], use_container_width=True)

    st.download_button(
        "Download triage queue (CSV)",
        data=out_view[show_cols].to_csv(index=False).encode("utf-8"),
        file_name="triage_queue_top50.csv",
        mime="text/csv",
    )

# -----------------------------
# Tab 8: Cost & Impact (8) — NON-ML
# -----------------------------
with tab8:
    st.subheader("Cost & Impact ")
    st.caption("Turns incidents into an estimated cost burden and highlights cost drivers (transparent assumptions).")

    recruiter_mode_8 = st.toggle("Explanation mode ", value=True, key="rec_mode_8")
    if recruiter_mode_8:
        st.info(
            "**What this is :**\n\n"
            "A practical estimator that converts incidents into an **estimated cost impact**.\n"
            "EHS teams use this to explain why certain hazards need budget and fast action."
        )

    if COST_EST_PATH.exists():
        cost_df = load_csv_safely(COST_EST_PATH)
        if "cost_estimate" not in cost_df.columns:
            st.error("Found cost_estimates.csv but it has no `cost_estimate` column.")
            cost_df = None
    else:
        cost_df = None

    if cost_df is None:
        st.warning(
            "No `outputs/cost_estimates.csv` found yet.\n\n"
            "Run locally:\n"
            "`python src/08_cost_model.py`\n\n"
            "Then commit/push `outputs/cost_estimates.csv` so it also works on Streamlit Cloud."
        )

        st.markdown("### Live fallback (computed in-app)")
        st.caption("This uses fixed assumptions. Replace numbers with your company’s rates if needed.")

        BASE_ADMIN = st.number_input("Base admin cost per record", min_value=0.0, value=250.0, step=50.0)
        INSPECTION_COST = st.number_input("Inspection handling cost", min_value=0.0, value=1500.0, step=100.0)
        HOSPITAL_COST = st.number_input("Hospitalization cost estimate", min_value=0.0, value=35000.0, step=1000.0)
        AMPUTATION_COST = st.number_input("Amputation cost estimate", min_value=0.0, value=120000.0, step=5000.0)

        tmp = df_filtered.copy()
        tmp["_hosp"] = _to01(tmp[hospital_col]) if hospital_col else 0
        tmp["_amp"] = _to01(tmp[amputation_col]) if amputation_col else 0
        tmp["_insp"] = _to01(tmp[inspection_col]) if inspection_col else 0

        tmp["cost_estimate"] = (
            BASE_ADMIN
            + tmp["_insp"] * INSPECTION_COST
            + tmp["_hosp"] * HOSPITAL_COST
            + tmp["_amp"] * AMPUTATION_COST
        )
        cost_df = tmp

    total_cost = float(cost_df["cost_estimate"].fillna(0).sum())
    avg_cost = float(cost_df["cost_estimate"].fillna(0).mean()) if len(cost_df) else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated total cost (filtered)", f"{total_cost:,.0f}")
    c2.metric("Average cost per record", f"{avg_cost:,.0f}")
    c3.metric("Records used", f"{len(cost_df):,}")

    st.divider()
    st.markdown("### Cost distribution")
    fig = px.histogram(cost_df, x="cost_estimate", nbins=40)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### Top cost drivers (grouped)")
    group_choice = st.selectbox(
        "Group by",
        options=["State", "Industry / NAICS", "Injury / Nature type"],
        index=0,
        key="cost_group_by",
    )

    if group_choice == "State" and state_col and state_col in cost_df.columns:
        grp_col = state_col
        label = "State"
    elif group_choice == "Industry / NAICS" and industry_col and industry_col in cost_df.columns:
        grp_col = industry_col
        label = "Industry / NAICS"
    elif group_choice == "Injury / Nature type" and injury_col and injury_col in cost_df.columns:
        grp_col = injury_col
        label = "Injury / Nature"
    else:
        grp_col = None
        st.info("That grouping column is not available in your dataset.")

    if grp_col:
        top_cost = (
            cost_df.groupby(grp_col)["cost_estimate"]
            .sum()
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
        )
        top_cost.columns = [label, "estimated_total_cost"]
        fig2 = px.bar(top_cost, x="estimated_total_cost", y=label, orientation="h")
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(top_cost, use_container_width=True)

    st.divider()
    st.download_button(
        "Download cost estimates (CSV)",
        data=cost_df.to_csv(index=False).encode("utf-8"),
        file_name="cost_estimates_filtered.csv",
        mime="text/csv",
    )

# -----------------------------
# Tab 9: Workflow Queue (9) — NON-ML
# -----------------------------
with tab9:
    st.subheader("Workflow Queue ")
    st.caption("A realistic incident triage queue: priority + SLA + owner.")

    recruiter_mode_9 = st.toggle("Explanation mode ", value=True, key="rec_mode_9")
    if recruiter_mode_9:
        st.info(
            "**What this is :**\n\n"
            "This turns incidents into a **work queue** for an EHS team:\n"
            "**Priority → SLA → Owner** so the team knows what to act on first."
        )

    if WORKFLOW_PATH.exists():
        wf = load_csv_safely(WORKFLOW_PATH)
        if not {"priority", "sla", "owner"}.issubset(set(wf.columns)):
            st.error("Found workflow_queue.csv but required columns are missing (priority/sla/owner).")
            wf = None
    else:
        wf = None

    if wf is None:
        st.warning(
            "No `outputs/workflow_queue.csv` found yet.\n\n"
            "Run locally:\n"
            "`python src/09_workflow_sim.py`\n\n"
            "Then commit/push `outputs/workflow_queue.csv` so it works on Streamlit Cloud."
        )

        tmp = df_filtered.copy()
        tmp["_hosp"] = _to01(tmp[hospital_col]) if hospital_col else 0
        tmp["_amp"] = _to01(tmp[amputation_col]) if amputation_col else 0
        tmp["_insp"] = _to01(tmp[inspection_col]) if inspection_col else 0

        def _triage_row(r):
            if int(r["_amp"]) == 1:
                return "P1", "0–4 hours", "EHS Lead + Site Supervisor"
            if int(r["_hosp"]) == 1:
                return "P2", "Within 24 hours", "EHS Specialist / Coordinator"
            if int(r["_insp"]) == 1:
                return "P3", "Within 7 days", "Supervisor + EHS (weekly review)"
            return "P4", "Monthly review", "Supervisor (with EHS oversight)"

        tri = tmp.apply(lambda r: pd.Series(_triage_row(r), index=["priority", "sla", "owner"]), axis=1)
        wf = pd.concat([tmp, tri], axis=1)

    p_counts = wf["priority"].value_counts(dropna=False).to_dict() if "priority" in wf.columns else {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total in queue", f"{len(wf):,}")
    c2.metric("P1", f"{p_counts.get('P1', 0):,}")
    c3.metric("P2", f"{p_counts.get('P2', 0):,}")
    c4.metric("P3+P4", f"{(p_counts.get('P3', 0) + p_counts.get('P4', 0)):,}")

    st.divider()
    st.markdown("### Queue view")

    order = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}
    wf["_rank"] = wf["priority"].map(order).fillna(9).astype(int)

    top_n = st.slider("Show top N records", min_value=10, max_value=300, value=50, step=10, key="wf_top_n")
    wf_view = wf.sort_values(["_rank"], ascending=True).head(top_n)

    show_cols = []
    for c in [state_col, industry_col, injury_col, event_title_col, source_title_col]:
        if c and c in wf_view.columns:
            show_cols.append(c)

    for c in ["priority", "sla", "owner"]:
        if c in wf_view.columns:
            show_cols.append(c)

    for c in [hospital_col, amputation_col, inspection_col]:
        if c and c in wf_view.columns and c not in show_cols:
            show_cols.append(c)

    st.dataframe(wf_view[show_cols], use_container_width=True)

    st.download_button(
        "Download workflow queue (CSV)",
        data=wf.sort_values(["_rank"]).drop(columns=["_rank"], errors="ignore").to_csv(index=False).encode("utf-8"),
        file_name="workflow_queue.csv",
        mime="text/csv",
    )

st.success("Dashboard ready.")