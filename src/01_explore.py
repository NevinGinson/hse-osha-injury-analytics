import pandas as pd
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = Path("data/severeinjury.csv")

print("📂 Loading dataset:", DATA_PATH)

# -----------------------------
# ROBUST CSV LOADER (handles encoding issues)
# -----------------------------
df = None
for enc in ["utf-8", "cp1252", "latin1"]:
    try:
        df = pd.read_csv(
            DATA_PATH,
            encoding=enc,
            encoding_errors="replace",
            low_memory=False
        )
        print(f"✅ Loaded successfully with encoding: {enc}")
        break
    except Exception as e:
        print(f"❌ Failed with encoding {enc} -> {e}")

if df is None:
    raise RuntimeError("🚨 Could not load CSV with tried encodings")

# -----------------------------
# BASIC OVERVIEW
# -----------------------------
print("\n📊 SHAPE")
print("Rows, Cols:", df.shape)

print("\n🧾 COLUMNS")
print(df.columns.tolist())

print("\n🔍 SAMPLE ROWS")
print(df.head(5))

# -----------------------------
# MISSING VALUES
# -----------------------------
print("\n⚠️ TOP 20 MISSING VALUE COUNTS")
missing = df.isna().sum().sort_values(ascending=False)
print(missing.head(20))

# -----------------------------
# DATA TYPES
# -----------------------------
print("\n🧠 DATA TYPES")
print(df.dtypes)

# -----------------------------
# QUICK VALUE COUNTS (for recruiters — shows thinking)
# -----------------------------
common_cols = ["Industry", "State", "NatureTitle", "EventTitle"]

print("\n📌 QUICK CATEGORY SNAPSHOT")
for col in common_cols:
    if col in df.columns:
        print(f"\n— {col} top 5 —")
        print(df[col].value_counts().head(5))

print("\n✅ Exploration step complete.")