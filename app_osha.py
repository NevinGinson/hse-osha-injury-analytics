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
# Keep these names consistent with your training script output
STRUCT_MODEL_PATH = ROOT / "outputs" / "tier2_structured_model.joblib"

st.title("OSHA Severe Injury Analytics Dashboard")
st.caption(
    "A safety analytics demo: KPIs + explainable risk scoring + text-based severity flagging + structured ML prediction."
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
        "- **Structured ML (A4):** predicts severity from structured fields (state/NAICS/nature + event/source titles + flags)\n\n"
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

# A4 fix: these two were missing in your live prediction payload
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
    # pipeline itself might have it
    cols = getattr(model_obj, "feature_names_in_", None)
    if cols is not None:
        return list(cols)

    # try steps
    named_steps = getattr(model_obj, "named_steps", None)
    if isinstance(named_steps, dict):
        for step in named_steps.values():
            cols = getattr(step, "feature_names_in_", None)
            if cols is not None:
                return list(cols)

    # fallback: use available known columns
    guess = []
    for c in [
        state_col, industry_col, injury_col, event_title_col, source_title_col,
        hospital_col, amputation_col, inspection_col
    ]:
        if c:
            guess.append(c)

    # also include canonical names if your training used them directly
    for c in ["EventTitle", "SourceTitle"]:
        if c not in guess:
            guess.append(c)

    return guess


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
        "- **Structured ML**: model trained on fields like NAICS/state/nature + titles + flags"
    )


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview + KPIs", "Trends", "Risk Score Intelligence", "Tier-2 Text Model", "Structured ML (A4)"]
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

    # Prefer precomputed CSV, otherwise compute live
    if RISK_PATH.exists():
        rdf = load_csv_safely(RISK_PATH)
        if "risk_score" not in rdf.columns:
            st.warning("Risk file exists but has no 'risk_score' column. Computing live score instead.")
            rdf = build_risk_df(df_filtered)
        else:
            # normalize columns for grouping
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
    st.subheader("Structured ML (A4)")
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

                # Categorical selectors
                if state_col:
                    options = sorted(df[state_col].dropna().astype(str).unique().tolist())
                    input_row[state_col] = st.selectbox("State", options)

                if industry_col:
                    options = sorted(df[industry_col].dropna().astype(str).unique().tolist())
                    input_row[industry_col] = st.selectbox("NAICS / Industry", options)

                if injury_col:
                    options = sorted(df[injury_col].dropna().astype(str).unique().tolist())
                    input_row[injury_col] = st.selectbox("Nature / Injury type", options)

                # These two were the reason for your error
                if event_title_col:
                    options = sorted(df[event_title_col].dropna().astype(str).unique().tolist())
                    input_row[event_title_col] = st.selectbox("Event title", options)
                else:
                    # still allow a manual entry so the pipeline can be satisfied
                    input_row["EventTitle"] = st.text_input("Event title", value="")

                if source_title_col:
                    options = sorted(df[source_title_col].dropna().astype(str).unique().tolist())
                    input_row[source_title_col] = st.selectbox("Source title", options)
                else:
                    input_row["SourceTitle"] = st.text_input("Source title", value="")

                # Flags
                if hospital_col:
                    input_row[hospital_col] = 1 if st.checkbox("Hospitalized") else 0
                if amputation_col:
                    input_row[amputation_col] = 1 if st.checkbox("Amputation") else 0
                if inspection_col:
                    input_row[inspection_col] = 1 if st.checkbox("Inspection") else 0

                if st.button("Predict severity (structured)"):
                    x = pd.DataFrame([input_row])

                    # Ensure required columns exist
                    for col in required_cols:
                        if col not in x.columns:
                            if any(k in col.lower() for k in ["hosp", "amput", "inspect", "hospital", "amputation"]):
                                x[col] = 0
                            else:
                                x[col] = ""

                    # Keep only required columns if possible (avoids column mismatch issues)
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


st.success("Dashboard ready.")