import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "severeinjury.csv"
OUT_PATH = ROOT / "outputs" / "osha_tier2_labeled.csv"


def read_csv_safely(path: Path) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", errors="replace")


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    for c in df.columns:
        cl = c.lower()
        for name in candidates:
            if name.lower() in cl:
                return c
    return None


def to01(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.strip().str.lower()
    return s.isin(["1", "true", "t", "yes", "y"]).astype(int)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = read_csv_safely(DATA_PATH)

    hospitalized_col = find_col(df, ["hospitalized", "hospitalised", "hospitalized?"])
    amputation_col   = find_col(df, ["amputation", "amputated"])
    narrative_col    = find_col(df, ["final narrative", "narrative", "description", "incident description"])
    nature_col       = find_col(df, ["naturetitle", "nature title", "nature"])
    event_col        = find_col(df, ["eventtitle", "event title"])

    # ✅ Correct way: start with an empty Series, then append available columns
    text_all = pd.Series("", index=df.index, dtype="string")

    for col in [narrative_col, nature_col, event_col]:
        if col:
            text_all = text_all + " " + df[col].fillna("").astype(str)

    text_all = text_all.str.lower()

    # Strong keywords → ensures we actually get positive labels
    high_kw = [
        "amputation", "amputat", "hospital", "inpatient", "admitted",
        "fatal", "death", "died",
        "fracture", "broken", "crush",
        "burn", "electric", "electroc",
        "loss of eye", "eye removed", "blind",
        "concussion", "head injury",
        "multiple", "severe", "critical"
    ]

    kw_hit = pd.Series(0, index=df.index)
    for k in high_kw:
        kw_hit = kw_hit + text_all.str.contains(k, regex=False).astype(int)

    amp  = to01(df[amputation_col]) if amputation_col else pd.Series(0, index=df.index)
    hosp = to01(df[hospitalized_col]) if hospitalized_col else pd.Series(0, index=df.index)

    # Label rule (explainable)
    df["is_high_severity"] = ((amp == 1) | (hosp == 1) | (kw_hit >= 1)).astype(int)

    OUT_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8")

    rate = df["is_high_severity"].mean()
    print(f"Saved labeled dataset -> {OUT_PATH}")
    print(f"High severity rate: {rate:.4f}  ({df['is_high_severity'].sum()} / {len(df)})")
    print("Classes present:", sorted(df["is_high_severity"].unique().tolist()))


if __name__ == "__main__":
    main()