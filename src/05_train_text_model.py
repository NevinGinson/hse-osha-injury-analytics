import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from joblib import dump


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "osha_tier2_labeled.csv"
OUT  = ROOT / "outputs"

OUT_PRED  = OUT / "tier2_baseline_predictions.csv"
OUT_MODEL = OUT / "tier2_text_model.joblib"


def pick_text_col(df: pd.DataFrame) -> str:
    candidates = [
        "final narrative", "narrative", "description", "incident description",
        "eventtitle", "event title",
        "naturetitle", "nature title", "nature",
        "summary", "what_happened"
    ]
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]

    for c in df.columns:
        cl = c.lower()
        if "narrat" in cl or "desc" in cl or "summary" in cl:
            return c

    raise ValueError("No text/description column found. Print df.columns and tell me.")


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Missing {DATA}. Run: python src/04_make_target.py first.")

    df = pd.read_csv(DATA)

    if "is_high_severity" not in df.columns:
        raise ValueError("Column 'is_high_severity' not found. Re-run src/04_make_target.py.")

    text_col = pick_text_col(df)

    df[text_col] = df[text_col].fillna("").astype(str)
    y = df["is_high_severity"].astype(int)
    X = df[text_col]

    # CRITICAL FIX: prevent one-class training crash
    classes = sorted(y.unique().tolist())
    if len(classes) < 2:
        raise ValueError(
            f"Only one class found in target: {classes}. "
            "Your labeling produced all 0 or all 1. "
            "Fix keywords/fields in src/04_make_target.py and re-run it."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            min_df=3,
            ngram_range=(1, 2),
            max_features=60000
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ))
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    print("\n=== Tier-2 Baseline: TF-IDF + Logistic Regression ===")
    print(f"Text column used: {text_col}")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification report:")
    print(classification_report(y_test, preds, digits=3))

    OUT.mkdir(exist_ok=True)

    pred_df = pd.DataFrame({
        "text": X_test.values,
        "y_true": y_test.values,
        "y_pred": preds
    })
    pred_df.to_csv(OUT_PRED, index=False)
    print(f"\nSaved predictions -> {OUT_PRED}")

    # ✅ Save model for Streamlit live prediction
    dump(pipe, OUT_MODEL)
    print(f"Saved model -> {OUT_MODEL}")


if __name__ == "__main__":
    main()