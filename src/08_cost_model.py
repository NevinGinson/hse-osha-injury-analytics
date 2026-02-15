import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "severeinjury.csv"
OUT_PATH = ROOT / "outputs" / "cost_estimates.csv"


def load_csv_safely(path: Path) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", errors="replace")


def find_col(df, possible_names):
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
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset at: {DATA_PATH}")

    df = load_csv_safely(DATA_PATH)

    # Common OSHA-ish columns
    state_col = find_col(df, ["state", "state_name"])
    naics_col = find_col(df, ["primary naics", "naics", "naics code", "primary_naics"])
    injury_col = find_col(df, ["naturetitle", "nature title", "nature", "injury", "injury_description"])
    hosp_col = find_col(df, ["hospitalized", "hospitalised", "hospitalized?"])
    amp_col = find_col(df, ["amputation", "amputated"])
    insp_col = find_col(df, ["inspection", "inspected"])

    # flags (0/1)
    df["_hosp"] = to01(df[hosp_col]) if hosp_col else 0
    df["_amp"] = to01(df[amp_col]) if amp_col else 0
    df["_insp"] = to01(df[insp_col]) if insp_col else 0

    # ---- Cost assumptions (editable) ----
    # Keep it transparent. Recruiters like that.
    BASE_ADMIN = 250.0            # basic admin / reporting overhead
    INSPECTION_COST = 1500.0      # internal time, paperwork, corrective-action tracking
    HOSPITAL_COST = 35000.0       # medical + lost time (rough order of magnitude)
    AMPUTATION_COST = 120000.0    # severe case (rough order of magnitude)

    # compute cost
    df["cost_estimate"] = (
        BASE_ADMIN
        + df["_insp"] * INSPECTION_COST
        + df["_hosp"] * HOSPITAL_COST
        + df["_amp"] * AMPUTATION_COST
    )

    # add readable group columns if present
    if state_col: df["state"] = df[state_col].astype(str)
    if naics_col: df["naics"] = df[naics_col].astype(str)
    if injury_col: df["injury"] = df[injury_col].astype(str)

    # Save
    OUT_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()