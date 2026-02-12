import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression

from joblib import dump

ROOT = Path(__file__).resolve().parents[1]

LABELED = ROOT / "outputs" / "osha_tier2_labeled.csv"  # from src/04_make_target.py
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

OUT_MODEL = OUT / "tier2_structured_model.joblib"
OUT_PRED  = OUT / "tier2_structured_predictions.csv"
OUT_METR  = OUT / "tier2_structured_metrics.json"

TARGET_COL = "is_high_severity"

def _to01(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.strip().str.lower()
    return s.isin(["1", "true", "t", "yes", "y"]).astype(int)

def _find_col(df: pd.DataFrame, names):
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    for c in df.columns:
        cl = c.lower()
        for n in names:
            if n.lower() in cl:
                return c
    return None

def main():
    if not LABELED.exists():
        raise FileNotFoundError(f"Missing: {LABELED}. Run: python src/04_make_target.py")

    df = pd.read_csv(LABELED)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in labeled dataset.")

    # Pick useful structured features (dataset-dependent, so we auto-detect)
    col_state  = _find_col(df, ["state"])
    col_naics  = _find_col(df, ["primary naics", "naics", "naics code"])
    col_nature = _find_col(df, ["naturetitle", "nature title", "nature"])
    col_event  = _find_col(df, ["eventtitle", "event title"])
    col_source = _find_col(df, ["sourcetitle", "source title", "source"])
    col_hosp   = _find_col(df, ["hospitalized", "hospitalised"])
    col_amp    = _find_col(df, ["amputation", "amputated"])
    col_insp   = _find_col(df, ["inspection", "inspected"])

    feat_cols = [c for c in [col_state, col_naics, col_nature, col_event, col_source, col_hosp, col_amp, col_insp] if c]
    if len(feat_cols) < 2:
        raise ValueError(f"Not enough structured columns found. Columns: {df.columns.tolist()}")

    X = df[feat_cols].copy()
    y = df[TARGET_COL].astype(int)

    # Convert flag-like cols to 0/1
    for col in [col_hosp, col_amp, col_insp]:
        if col and col in X.columns:
            X[col] = _to01(X[col])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Identify numeric vs categorical
    numeric_cols = []
    cat_cols = []
    for c in X.columns:
        # treat 0/1 as numeric
        try:
            uniq = set(pd.Series(X[c].dropna().unique()).tolist())
            if uniq.issubset({0, 1}):
                numeric_cols.append(c)
            else:
                cat_cols.append(c)
        except Exception:
            cat_cols.append(c)

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", numeric_cols),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")

    pipe = Pipeline([
        ("pre", pre),
        ("clf", clf),
    ])

    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)

    print("\n=== Tier-2 Structured Model (No Text) ===")
    print("Features used:", feat_cols)
    print("\nAccuracy:", round(acc, 3))
    print("ROC-AUC :", round(auc, 3))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification report:")
    print(classification_report(y_test, preds, digits=3))

    # Save predictions (for Streamlit)
    out_pred = X_test.copy()
    out_pred["y_true"] = y_test.values
    out_pred["y_pred"] = preds
    out_pred["p_high"] = proba
    out_pred.to_csv(OUT_PRED, index=False)
    print(f"\nSaved: {OUT_PRED}")

    # Save model (for Streamlit live prediction)
    dump(pipe, OUT_MODEL)
    print(f"Saved model: {OUT_MODEL}")

if __name__ == "__main__":
    main()