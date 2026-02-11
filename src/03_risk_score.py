# src/03_risk_score.py
from __future__ import annotations

from pathlib import Path
import re
import pandas as pd


DATA_PATH = Path("data") / "severeinjury.csv"
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


def _norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = list(df.columns)
    norm_map = {_norm(c): c for c in cols}

    for cand in candidates:
        key = _norm(cand)
        if key in norm_map:
            return norm_map[key]

    norm_cols = list(norm_map.keys())
    for cand in candidates:
        key = _norm(cand)
        for nc in norm_cols:
            if key and (key in nc):
                return norm_map[nc]

    return None


def read_csv_robust(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")
    except Exception:
        return pd.read_csv(path, encoding="cp1252")


def make_risk_score(df: pd.DataFrame, text_col: str | None) -> pd.DataFrame:
    df = df.copy()

    weights = {
        "amputation": 10,
        "hospital": 7,
        "in-patient": 7,
        "inpatient": 7,
        "fracture": 6,
        "burn": 6,
        "laceration": 4,
        "cut": 3,
        "eye": 8,
        "loss of eye": 10,
        "fatal": 10,
        "death": 10,
        "crush": 6,
        "electrocution": 9,
        "fall": 5,
        "ladder": 4,
        "scaffold": 5,
        "machine": 4,
        "forklift": 5,
        "chemical": 5,
    }

    def score_text(x: str) -> int:
        if pd.isna(x):
            return 0
        t = str(x).lower()
        s = 0
        for k, w in weights.items():
            if k in t:
                s += w
        return s

    if text_col and text_col in df.columns:
        df["risk_score"] = df[text_col].apply(score_text)
    else:
        df["risk_score"] = 0

    def band(x: float) -> str:
        if x >= 12:
            return "High"
        elif x >= 5:
            return "Medium"
        return "Low"

    df["risk_band"] = df["risk_score"].apply(band)
    return df


def main():
    df = read_csv_robust(DATA_PATH)

    state_col = find_col(df, ["state", "st", "state_code", "state name"])
    industry_col = find_col(
        df,
        [
            "industry",
            "industry_description",
            "industry description",
            "naics",
            "naics_title",
            "naics title",
            "naics_code",
            "naics code",
            "industry_title",
            "industry title",
            "establishment_naics",
        ],
    )
    injury_col = find_col(
        df,
        [
            "injury",
            "injury_type",
            "injury type",
            "nature_of_injury",
            "nature of injury",
            "injury_description",
            "injury description",
            "injury sustained",
            "illness",
            "illness_description",
            "illness description",
        ],
    )
    desc_col = find_col(
        df,
        [
            "description",
            "incident_description",
            "incident description",
            "event_description",
            "event description",
            "summary",
            "narrative",
            "accident_description",
            "accident description",
        ],
    )
    date_col = find_col(
        df,
        [
            "event_date",
            "event date",
            "date",
            "incident_date",
            "incident date",
            "injury_date",
            "injury date",
            "received_date",
            "received date",
            "open_date",
            "open date",
        ],
    )

    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Prefer description-like column, otherwise injury-like column
    text_col = desc_col or injury_col

    df = make_risk_score(df, text_col)

    enriched_path = OUT / "osha_with_risk_score.csv"
    df.to_csv(enriched_path, index=False)

    # Aggregations (only if columns exist)
    if state_col:
        df.groupby(state_col)["risk_score"].agg(["count", "mean", "sum"]).sort_values("sum", ascending=False).to_csv(
            OUT / "risk_by_state.csv"
        )

    if industry_col:
        df.groupby(industry_col)["risk_score"].agg(["count", "mean", "sum"]).sort_values("sum", ascending=False).to_csv(
            OUT / "risk_by_industry.csv"
        )

    if injury_col:
        df.groupby(injury_col)["risk_score"].agg(["count", "mean", "sum"]).sort_values("sum", ascending=False).to_csv(
            OUT / "risk_by_injury_type.csv"
        )

    print("✅ Risk score created.")
    print(f"Saved: {enriched_path}")
    if state_col:
        print("Saved: outputs/risk_by_state.csv")
    if industry_col:
        print("Saved: outputs/risk_by_industry.csv")
    if injury_col:
        print("Saved: outputs/risk_by_injury_type.csv")


if __name__ == "__main__":
    main()