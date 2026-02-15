import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "severeinjury.csv"
OUT_PATH = ROOT / "outputs" / "workflow_queue.csv"


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


def triage_priority(hosp, amp, insp):
    # simple defensible rules (non-ML)
    if amp == 1:
        return "P1", "0–4 hours", "EHS Lead + Site Supervisor"
    if hosp == 1:
        return "P2", "Within 24 hours", "EHS Specialist / Coordinator"
    if insp == 1:
        return "P3", "Within 7 days", "Supervisor + EHS (weekly review)"
    return "P4", "Monthly review", "Supervisor (with EHS oversight)"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset at: {DATA_PATH}")

    df = load_csv_safely(DATA_PATH)

    state_col = find_col(df, ["state", "state_name"])
    naics_col = find_col(df, ["primary naics", "naics", "naics code", "primary_naics"])
    injury_col = find_col(df, ["naturetitle", "nature title", "nature", "injury", "injury_description"])
    hosp_col = find_col(df, ["hospitalized", "hospitalised", "hospitalized?"])
    amp_col = find_col(df, ["amputation", "amputated"])
    insp_col = find_col(df, ["inspection", "inspected"])

    df["_hosp"] = to01(df[hosp_col]) if hosp_col else 0
    df["_amp"] = to01(df[amp_col]) if amp_col else 0
    df["_insp"] = to01(df[insp_col]) if insp_col else 0

    out = pd.DataFrame()
    out["state"] = df[state_col].astype(str) if state_col else ""
    out["naics"] = df[naics_col].astype(str) if naics_col else ""
    out["injury"] = df[injury_col].astype(str) if injury_col else ""
    out["hospitalized"] = df["_hosp"]
    out["amputation"] = df["_amp"]
    out["inspection"] = df["_insp"]

    # compute triage outputs
    priorities, slas, owners = [], [], []
    for _, r in out.iterrows():
        p, sla, owner = triage_priority(r["hospitalized"], r["amputation"], r["inspection"])
        priorities.append(p); slas.append(sla); owners.append(owner)

    out["priority"] = priorities
    out["sla"] = slas
    out["owner"] = owners

    # Sort queue: P1 first → P4 last
    order = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}
    out["_rank"] = out["priority"].map(order).fillna(9).astype(int)
    out = out.sort_values(["_rank", "amputation", "hospitalized", "inspection"], ascending=[True, False, False, False])
    out = out.drop(columns=["_rank"])

    OUT_PATH.parent.mkdir(exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()