import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest

try:
    from joblib import dump
except Exception:
    dump = None

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "severeinjury.csv"
OUT_DIR = ROOT / "outputs"
OUT_SCORES = OUT_DIR / "anomaly_scores.csv"
OUT_MODEL = OUT_DIR / "anomaly_model.joblib"


def load_csv_safely(path: Path) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", errors="replace")


def find_col(df: pd.DataFrame, possible_names):
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


def to01(series: pd.Series) -> pd.Series:
    s = series.fillna(0).astype(str).str.strip().str.lower()
    return s.isin(["1", "true", "t", "yes", "y"]).astype(int)


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA} (expected data/severeinjury.csv)")

    df = load_csv_safely(DATA)

    # Core columns (robust guessing)
    state_col = find_col(df, ["state", "state_name"])
    industry_col = find_col(df, [
        "primary naics", "naics", "naics code", "primary_naics",
        "industry", "industry_description", "naics_title", "industry name"
    ])
    injury_col = find_col(df, [
        "naturetitle", "nature title", "nature",
        "injury", "injury_description"
    ])
    event_title_col = find_col(df, ["eventtitle", "event title", "event_title"])
    source_title_col = find_col(df, ["sourcetitle", "source title", "source_title"])

    hospital_col = find_col(df, ["hospitalized", "hospitalised", "hospitalized?"])
    amputation_col = find_col(df, ["amputation", "amputated"])
    inspection_col = find_col(df, ["inspection", "inspected"])

    used_cat = [c for c in [state_col, industry_col, injury_col] if c]
    used_txt = [c for c in [event_title_col, source_title_col] if c]

    if len(used_cat) == 0 and len(used_txt) == 0:
        raise ValueError(
            "No usable columns detected. Need at least one of: "
            "State / NAICS / Nature / EventTitle / SourceTitle."
        )

    work = df.copy()

    # numeric flags
    work["_hospitalized"] = to01(work[hospital_col]) if hospital_col else 0
    work["_amputation"] = to01(work[amputation_col]) if amputation_col else 0
    work["_inspection"] = to01(work[inspection_col]) if inspection_col else 0

    # fill categoricals
    for c in used_cat:
        work[c] = work[c].fillna("").astype(str).str.strip()

    # Build a single text field from titles if available (SAFE)
    work["_titles_text"] = ""
    if len(used_txt) > 0:
        parts = []
        for c in used_txt:
            parts.append(work[c].fillna("").astype(str))
        work["_titles_text"] = parts[0]
        for p in parts[1:]:
            work["_titles_text"] = (work["_titles_text"] + " " + p)
        work["_titles_text"] = work["_titles_text"].str.strip()

    # rarity signal (transparent)
    combo_cols = [c for c in [state_col, industry_col, injury_col] if c]
    if len(combo_cols) >= 2:
        combo_key = work[combo_cols].astype(str).agg(" | ".join, axis=1)
        combo_counts = combo_key.value_counts()
        work["combo_count"] = combo_key.map(combo_counts).fillna(1).astype(int)
        work["combo_rarity_index"] = (1.0 / work["combo_count"]).round(6)
    else:
        work["combo_count"] = 0
        work["combo_rarity_index"] = 0.0

    # model pipeline
    feature_cols = []
    if len(used_cat) > 0:
        feature_cols += used_cat
    feature_cols += ["_titles_text", "_hospitalized", "_amputation", "_inspection"]

    X = work[feature_cols].copy()

    categorical_features = used_cat[:]
    text_feature = "_titles_text"
    numeric_features = ["_hospitalized", "_amputation", "_inspection"]

    transformers = []
    if len(categorical_features) > 0:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features))

    transformers.append((
        "txt",
        HashingVectorizer(n_features=2**14, alternate_sign=False, norm="l2", lowercase=True),
        text_feature
    ))

    transformers.append(("num", "passthrough", numeric_features))

    pre = ColumnTransformer(transformers, remainder="drop")

    clf = IsolationForest(
        n_estimators=300,
        contamination=0.01,
        random_state=42
    )

    pipe = Pipeline([("pre", pre), ("model", clf)])
    pipe.fit(X)

    Z = pipe.named_steps["pre"].transform(X)
    normality = pipe.named_steps["model"].decision_function(Z)
    work["anomaly_score"] = np.round((-normality).astype(float), 6)
    work["anomaly_flag"] = (pipe.named_steps["model"].predict(Z) == -1).astype(int)

    # review hints (simple triage cues)
    reason = []
    reason.append(np.where(work["_amputation"] == 1, "amputation_flag", ""))
    reason.append(np.where(work["_hospitalized"] == 1, "hospitalized_flag", ""))
    reason.append(np.where(work["_inspection"] == 1, "inspection_flag", ""))
    reason.append(np.where((work["combo_count"] > 0) & (work["combo_count"] <= 3), "rare_combo", ""))

    work["review_hints"] = (
        pd.DataFrame(reason).T
        .apply(lambda r: ", ".join([x for x in r if x]), axis=1)
        .replace("", "none")
    )

    # output columns (keep key original fields)
    keep_cols = []
    for c in [state_col, industry_col, injury_col, event_title_col, source_title_col]:
        if c and c in work.columns:
            keep_cols.append(c)
    for c in [hospital_col, amputation_col, inspection_col]:
        if c and c in work.columns and c not in keep_cols:
            keep_cols.append(c)

    keep_cols += ["anomaly_flag", "anomaly_score", "combo_count", "combo_rarity_index", "review_hints"]
    out_df = work[keep_cols].copy()

    OUT_DIR.mkdir(exist_ok=True)
    out_df.to_csv(OUT_SCORES, index=False)

    if dump is not None:
        try:
            dump(pipe, OUT_MODEL)
        except Exception:
            pass

    n = int(out_df["anomaly_flag"].sum())
    total = len(out_df)

    print("\n=== Anomaly Detection  ===")
    print("Purpose: flag unusual records for review (audit, data quality checks, investigation prioritization).")
    print(f"Rows scored: {total:,}")
    print(f"Flagged anomalies: {n:,} ({(100*n/total if total else 0):.2f}%)")
    print(f"Saved: {OUT_SCORES}")
    if OUT_MODEL.exists():
        print(f"Saved model: {OUT_MODEL}")

    top = out_df.sort_values("anomaly_score", ascending=False).head(15)
    print("\nTop flagged records (highest anomaly_score):")
    with pd.option_context("display.max_colwidth", 80, "display.width", 140):
        print(top.to_string(index=False))


if __name__ == "__main__":
    main()