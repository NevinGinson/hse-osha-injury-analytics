# OSHA Severe Injury Analytics Dashboard (EHS / HSE Decision Support)

Live app:
https://hse-osha-injury-analytics-55ntxd34atyewwsrrtz49g.streamlit.app/

Interactive Streamlit dashboard analyzing OSHA severe workplace injury reports (2015–2017) and turning them into **EHS-ready triage insights**.

---

## Project Goal
Build a practical **EHS / HSE analytics + decision-support** dashboard that helps:
- understand injury patterns and trends
- identify high-risk industries and injury types
- support **incident triage** and **prioritization**
- demonstrate real-world safety analytics workflows (not just charts)

Designed to showcase **EHS/HSE data analytics skills** using Python + Streamlit, with simple and explainable ML signals.

---

## What this app does (high level)
This dashboard converts raw OSHA reports into **actionable safety intelligence**:

1) **Descriptive Analytics**
- Where incidents happen (state / NAICS / injury nature)
- What types are most common
- How incidents change over time

2) **Severity Signals**
- Explainable **Risk Score (1–20)** using clear rules (hospitalization/amputation/inspection + keywords)

3) **Predictive Signals**
- **Tier-2 Text Model:** uses narrative text to estimate high severity likelihood
- **Structured ML (A4):** uses structured fields (state/NAICS/nature + event/source titles + flags)

4) **Anomaly Detection (B)**
- Flags unusual / rare combinations (outliers) to support:
  - audit & QA checks
  - finding “weird” cases worth reviewing
  - surfacing potential misclassification or emerging hazards

5) **Decision Layer (EHS Triage)**
- Converts signals into:
  - **Priority (P1–P4)**
  - **Recommended action**
  - **SLA**
  - **Owner**
- Outputs a **ranked triage queue** (who should review what first)

---

## Features
### Core dashboard
- Top industries (NAICS) by severe injuries
- Top injury / nature types
- State distribution tables
- Yearly trend visualization
- Interactive charts (Plotly)
- Clean layout + filters

### Risk Score Intelligence
- Transparent severity score using:
  - hospitalization / amputation / inspection signals (if available)
  - narrative keyword signals (if narrative exists)
- Risk score distribution + top high-risk NAICS and injury types
- Download scored dataset

### Tier-2 Text Model
- Shows model evaluation (accuracy + confusion matrix) if predictions file exists
- Live prediction (if trained model file exists)

### Structured ML (A4)
- Live prediction using your trained structured model (if model file exists)
- Works with flexible column names (robust column detection)

### Anomaly Detection (B)
- Shows anomaly score distribution
- Displays top unusual records for review
- Download anomaly scores file

### Decision Layer (Operational EHS output)
- Live triage sandbox (adjust inputs and see priority/action/SLA/owner)
- Applies the decision layer to the dataset and produces a **top 50 action list**

---

## Tech Stack
- Python
- Pandas / NumPy
- Streamlit
- Plotly
- scikit-learn (for ML models)
- joblib (model loading)

---

## Dataset
OSHA Severe Injury Reports (Public dataset, Kaggle)

- ~22,000 reports
- Period: 2015–2017

---

## Repository Structure (expected)