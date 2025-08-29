# strategy.py
import pandas as pd
from typing import Dict, Any, List, Tuple

DEFAULT_STRATEGY: Dict[str, Any] = {
    # soft caps on total players per position (including bench)
    "max_cap": {"QB": 3, "RB": 7, "WR": 7, "TE": 3, "DST": 2, "K": 2},
    # early-round dampeners so K/DST don’t show up too early
    "early_tax_round": {"K": 12, "DST": 12},  # before this round, penalize heavily
    "early_tax_points": {"K": 150.0, "DST": 150.0},
    # bonus when we still need a STARTER at that position
    "starter_need_bonus": {"QB": 160.0, "RB": 140.0, "WR": 140.0, "TE": 140.0, "DST": 200.0, "K": 200.0},
    # small bonus for first bench at a position (depth)
    "bench_first_bonus": {"QB": 25.0, "RB": 40.0, "WR": 40.0, "TE": 25.0},
    # penalty after exceeding soft caps
    "over_cap_penalty": 120.0,
    # generic scarcity/tier nudges (if you want to pass a tier column)
    "tier_step": 8.0,
    # profile keeps backward-compat with your CLI
    "profile": "balanced",
}

def _count_starters(roster_positions: List[str]) -> Dict[str, int]:
    """Count required starters by POS from league roster positions."""
    # Example roster_positions: ["QB","RB","RB","WR","WR","TE","FLEX","K","DST","BN","BN","BN"...]
    starters = {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "DST": 0, "K": 0}
    for pos in roster_positions:
        p = (pos or "").upper()
        if p in starters:
            starters[p] += 1
        elif p in ("WR/RB", "WR/RB/TE", "FLEX"):  # FLEX can be RB/WR/TE; handle later by “need anywhere”
            # We’ll treat flex dynamically (see below) — not counted here
            pass
    return starters

def _position_counts(df: pd.DataFrame) -> Dict[str, int]:
    if df.empty:
        return {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "DST": 0, "K": 0}
    vc = df["position"].astype(str).str.upper().value_counts().to_dict()
    for p in ["QB","RB","WR","TE","DST","K"]:
        vc.setdefault(p, 0)
    return vc

def _current_round(total_picks_made: int, num_teams: int) -> int:
    if num_teams <= 0:
        return 1
    return (total_picks_made // num_teams) + 1

def _picks_made_by_me(my_picks_df: pd.DataFrame) -> int:
    return 0 if my_picks_df.empty else len(my_picks_df)

def _total_rounds_fallback(roster_positions: List[str]) -> int:
    # conservative fallback if total_rounds not provided: starters + ~6 bench
    starters_est = sum(1 for p in roster_positions if p not in ("BN","TAXI","IR"))
    return starters_est + 6

def _need_map(
    roster_positions: List[str],
    my_picks_df: pd.DataFrame
) -> Tuple[Dict[str,int], int, Dict[str,int]]:
    starters_req = _count_starters(roster_positions)
    mine = _position_counts(my_picks_df)

    # FLEX logic: estimate FLEX needs as count of flex-like slots
    flex_slots = sum(1 for p in roster_positions if (p or "").upper() in ("FLEX","WR/RB","WR/RB/TE"))
    # how many of RB/WR/TE we have above strict starters
    rb_excess = max(0, mine["RB"] - starters_req["RB"])
    wr_excess = max(0, mine["WR"] - starters_req["WR"])
    te_excess = max(0, mine["TE"] - starters_req["TE"])
    flex_filled = min(flex_slots, rb_excess + wr_excess + te_excess)
    flex_remaining = max(0, flex_slots - flex_filled)

    # “starters still needed” (not counting flex) per hard slot
    need_starters = {
        "QB": max(0, starters_req["QB"] - mine["QB"]),
        "RB": max(0, starters_req["RB"] - mine["RB"]),
        "WR": max(0, starters_req["WR"] - mine["WR"]),
        "TE": max(0, starters_req["TE"] - mine["TE"]),
        "DST": max(0, starters_req["DST"] - mine["DST"]),
        "K":  max(0, starters_req["K"]  - mine["K"]),
    }
    return need_starters, flex_remaining, mine

def score_candidates(
    available_df: pd.DataFrame,
    my_picks_df: pd.DataFrame,
    league_roster_positions: List[str],
    num_teams: int,
    total_picks_made: int,
    my_slot: int,
    neighbor_slots: List[int],
    neighbor_needs: Dict[int, Dict[str,int]],
    strategy: Dict[str, Any],
    total_rounds: int = None,   # <-- allow draft_assistant to pass settings['rounds']
) -> Tuple[pd.DataFrame, Dict[str,int], Dict[str,int]]:
    """
    Returns (scored_df, starters_needed_map, my_position_counts).
    Scored DF includes a 'score' column sorted descending.
    """

    if available_df.empty:
        return available_df.assign(score=0.0), {}, {}

    # derive round math
    current_round = _current_round(total_picks_made, num_teams)
    picks_mine = _picks_made_by_me(my_picks_df)

    if total_rounds is None or int(total_rounds) <= 0:
        total_rounds = _total_rounds_fallback(league_roster_positions)
    picks_left_for_me = max(0, int(total_rounds) - picks_mine)

    # compute needs
    need_starters, flex_remaining, my_counts = _need_map(league_roster_positions, my_picks_df)
    starters_left_total = sum(need_starters.values()) + flex_remaining

    # base value from rank/adp (lower is better)
    df = available_df.copy()
    df["rank"] = pd.to_numeric(df.get("rank"), errors="coerce").fillna(10_000)
    df["adp"]  = pd.to_numeric(df.get("adp"), errors="coerce").fillna(999.9)
    df["tier"] = pd.to_numeric(df.get("tier"), errors="coerce").fillna(99)
    df["position"] = df["position"].astype(str).str.upper().replace({"DEF":"DST","D/ST":"DST"})

    # start score with negative rank (so rank 1 > rank 2)
    df["score"] = -df["rank"]

    # light ADP nudge (earlier ADP -> slightly better)
    df["score"] += (1000.0 - df["adp"]) * 0.02  # small weight

    # tier nudge (lower tier number is better)
    df["score"] += (100.0 - df["tier"]) * (strategy.get("tier_step", 8.0) / 100.0)

    # EARLY TAX to keep K/DST out of top early unless insane value
    early_tax_round = strategy.get("early_tax_round", {})
    early_tax_points = strategy.get("early_tax_points", {})
    for pos in ("K","DST"):
        tax_r = early_tax_round.get(pos, 12)
        tax_p = early_tax_points.get(pos, 150.0)
        mask = (df["position"] == pos) & (current_round < tax_r)
        df.loc[mask, "score"] -= tax_p

    # BONUS: if we still need a starter at a position, push it up
    starter_bonus = strategy.get("starter_need_bonus", {})
    for pos, need in need_starters.items():
        if need > 0:
            df.loc[df["position"] == pos, "score"] += starter_bonus.get(pos, 140.0)

    # FLEX handling: tiny generic boost to RB/WR/TE if we still have flex slots
    if flex_remaining > 0:
        df.loc[df["position"].isin(["RB","WR","TE"]), "score"] += 22.0

    # BENCH-first bonus: first bench at a position
    bench_bonus = strategy.get("bench_first_bonus", {})
    for pos in ("QB","RB","WR","TE"):
        # if starters are already filled AND we have 0 bench at that pos -> small bonus
        starters_done = need_starters.get(pos, 0) == 0
        if starters_done:
            have = my_counts.get(pos, 0)
            req = _count_starters(league_roster_positions).get(pos, 0)
            benches = max(0, have - req)
            if benches == 0:
                df.loc[df["position"] == pos, "score"] += bench_bonus.get(pos, 0.0)

    # SOFT CAPS: penalize exceeding caps
    caps = strategy.get("max_cap", {})
    for pos, cap in caps.items():
        if my_counts.get(pos, 0) >= cap:
            df.loc[df["position"] == pos, "score"] -= strategy.get("over_cap_penalty", 120.0)

    # MUST-FILL rule: if picks left equals starters left, ONLY allow starter-needed positions
    # (This is what guarantees you will pick K/DST before the draft ends.)
    if picks_left_for_me <= starters_left_total and starters_left_total > 0:
        allowed_positions = set([p for p, n in need_starters.items() if n > 0])
        # flex can be filled by RB/WR/TE; allow those if we still need flex
        if flex_remaining > 0:
            allowed_positions.update({"RB","WR","TE"})
        # heavily penalize anything not in allowed (instead of hard filter to keep list stable)
        df.loc[~df["position"].isin(allowed_positions), "score"] -= 5000.0

    # Final sort & return
    df = df.sort_values(["score", "rank"], ascending=[False, True]).reset_index(drop=True)
    return df, need_starters, my_counts
