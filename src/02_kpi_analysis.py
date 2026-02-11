import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/severeinjury.csv")

# -----------------------------
# LOAD DATA (same robust loader)
# -----------------------------
for enc in ["utf-8", "cp1252", "latin1"]:
    try:
        df = pd.read_csv(DATA_PATH, encoding=enc, encoding_errors="replace", low_memory=False)
        print("Loaded with:", enc)
        break
    except:
        continue

# -----------------------------
# BASIC CLEANING
# -----------------------------
print("\nRows before cleaning:", len(df))

df.columns = df.columns.str.strip()

# try parse date column if exists
date_cols = [c for c in df.columns if "date" in c.lower()]
for c in date_cols:
    df[c] = pd.to_datetime(df[c], errors="coerce")

print("Detected date columns:", date_cols)

# -----------------------------
# KPI 1 — INCIDENTS PER YEAR
# -----------------------------
if date_cols:
    year_col = date_cols[0]
    df["year"] = df[year_col].dt.year

    print("\n📅 Incidents per year:")
    print(df["year"].value_counts().sort_index())

# -----------------------------
# KPI 2 — TOP INDUSTRIES
# -----------------------------
industry_cols = [c for c in df.columns if "industry" in c.lower()]

if industry_cols:
    col = industry_cols[0]
    print("\n🏭 Top industries (severe injuries):")
    print(df[col].value_counts().head(10))

# -----------------------------
# KPI 3 — TOP INJURY TYPES
# -----------------------------
injury_cols = [c for c in df.columns if "nature" in c.lower() or "injury" in c.lower()]

if injury_cols:
    col = injury_cols[0]
    print("\n🩹 Top injury types:")
    print(df[col].value_counts().head(10))

# -----------------------------
# KPI 4 — STATES / REGIONS
# -----------------------------
state_cols = [c for c in df.columns if "state" in c.lower()]

if state_cols:
    col = state_cols[0]
    print("\n🗺️ Top states by severe injuries:")
    print(df[col].value_counts().head(10))

# -----------------------------
# SAVE SUMMARY TABLES (for dashboard later)
# -----------------------------
out = Path("outputs")
out.mkdir(exist_ok=True)

if industry_cols:
    df[industry_cols[0]].value_counts().head(20).to_csv(out / "top_industries.csv")

if injury_cols:
    df[injury_cols[0]].value_counts().head(20).to_csv(out / "top_injuries.csv")

print("\n✅ KPI analysis complete — outputs saved.")