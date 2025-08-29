#!/usr/bin/env python3
"""
ingest_fantasypros_csv.py

Usage:
    python ingest_fantasypros_csv.py --in path/to/fantasypros.csv --out data/rankings_fp_clean.csv

Takes a FantasyPros "Download CSV" export (overall cheat sheet) and converts it to the format
expected by draft_assistant.py:
    player_name,position,team,rank,bye,adp,tier,sleeper_tag,sleeper_id

- rank: uses the CSV's overall rank (ECR). If not present, derives from row order.
- adp: uses ADP column if present.
- tier: uses "Tier" if present; else auto-filled later by the assistant.
- sleeper_tag: left blank (0).
- sleeper_id: left blank (auto-matched by name+position at runtime if possible).
"""

import argparse
import pandas as pd
import numpy as np

KEEP_COLS = ["player_name","position","team","rank","bye","adp","tier","sleeper_tag","sleeper_id"]

def normalize_pos(p):
    if pd.isna(p): return None
    p = str(p).strip().upper()
    # Map FantasyPros variants to our set
    if p in {"D/ST","DST","DEF"}: return "DST"
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Path to FantasyPros CSV export")
    ap.add_argument("--out", dest="outp", required=True, help="Path to write cleaned CSV")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    cols = {c.lower(): c for c in df.columns}

    # Try to find standard FantasyPros columns
    # Common column names: "Player", "POS", "Team", "ECR", "ADP", "Tier", "Bye"
    def pick(name_options):
        for opt in name_options:
            for c in df.columns:
                if c.strip().lower() == opt.lower():
                    return c
        return None

    c_player = pick(["player","name","player name"])
    c_pos    = pick(["pos","position"])
    c_team   = pick(["team"])
    c_rank   = pick(["ecr","overall","rank"])
    c_adp    = pick(["adp"])
    c_tier   = pick(["tier"])
    c_bye    = pick(["bye","bye week","bye_wk","bye_wk."])

    if not c_player or not c_pos:
        raise SystemExit("Could not find Player or POS columns in the input CSV. Please export the OVERALL cheat sheet CSV from FantasyPros.")

    out = pd.DataFrame()
    out["player_name"] = df[c_player].astype(str).str.replace(r"\s+\(.*?\)$", "", regex=True)  # strip team/pos suffixes if present
    out["position"]    = df[c_pos].map(normalize_pos)
    out["team"]        = df[c_team] if c_team else None
    if c_rank and df[c_rank].notna().any():
        out["rank"] = pd.to_numeric(df[c_rank], errors="coerce")
    else:
        out["rank"] = range(1, len(df)+1)  # fallback from order

    out["bye"]  = pd.to_numeric(df[c_bye], errors="coerce") if c_bye else None
    out["adp"]  = pd.to_numeric(df[c_adp], errors="coerce") if c_adp else None
    out["tier"] = pd.to_numeric(df[c_tier], errors="coerce").astype("Int64") if c_tier else pd.Series([None]*len(out))
    out["sleeper_tag"] = 0
    out["sleeper_id"]  = None

    # Drop any rows missing critical info (player, position, rank)
    out = out.dropna(subset=["player_name","position","rank"])
    out = out.sort_values(["rank","player_name"], ascending=[True, True]).reset_index(drop=True)

    # Write
    out.to_csv(args.outp, index=False, columns=KEEP_COLS)
    print(f"Wrote {len(out)} rows to {args.outp}")

if __name__ == "__main__":
    main()
