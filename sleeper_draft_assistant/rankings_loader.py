import pandas as pd

REQUIRED_COLS = ["player_name", "position", "rank"]
OPTIONAL_COLS = ["team", "bye", "sleeper_id", "adp", "tier", "sleeper_tag"]

POSITION_ORDER = ["QB", "RB", "WR", "TE", "DST", "K"]

def load_rankings_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in REQUIRED_COLS:
        if c not in df.columns:
            raise ValueError(f"Rankings CSV missing required column: {c}")
    for c in OPTIONAL_COLS:
        if c not in df.columns:
            df[c] = None
    # Normalize columns
    df["position"] = df["position"].astype(str).str.upper().str.strip()
    # sleeper_tag to 0/1
    if "sleeper_tag" in df.columns:
        df["sleeper_tag"] = df["sleeper_tag"].fillna(0).astype(int)
    # tier as int if exists
    if "tier" in df.columns:
        try:
            df["tier"] = df["tier"].astype(float).astype("Int64")
        except:
            pass
    # adp numeric if exists
    if "adp" in df.columns:
        df["adp"] = pd.to_numeric(df["adp"], errors="coerce")
    df = df.sort_values(by=["rank", "player_name"], ascending=[True, True]).reset_index(drop=True)
    return df
