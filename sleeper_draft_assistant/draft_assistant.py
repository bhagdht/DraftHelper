#!/usr/bin/env python3
"""
Sleeper Draft Assistant (CSV-compatible, recovery-safe, mock-safe)

- Rankings CSV header expected:
  player_name,position,team,rank,bye,adp,tier,sleeper_tag,sleeper_id
- Normalizes positions like WR1/RB2 -> WR/RB (keeps the number as tier when missing)
- rank/adp/bye/tier coerced to numeric (with safe defaults)
- Removes taken players via robust anti-joins (id, name+pos, last+team+pos, last+pos)
- Safe fallback so recommendations always appear
- Strategy scoring in Python (LLM narration optional via --use-openai)

Requires (same folder):
  sleeper_api.py
  rankings_loader.py
  strategy.py
  excel_format.py
"""

import argparse
import os
import time
import re
import unicodedata
from typing import Dict, Any, List

import pandas as pd
from dotenv import load_dotenv

# Local modules
from sleeper_api import (
    get_user,
    get_user_leagues,
    get_league,
    get_league_users,
    get_league_rosters,
    get_league_drafts,
    get_draft,
    get_draft_picks,
    get_players,
)
from rankings_loader import load_rankings_csv
from strategy import score_candidates, DEFAULT_STRATEGY
from excel_format import apply_tier_colors, bold_headers, autosize


# ---------------- Optional OpenAI narration ----------------
def llm_narration(candidates: List[Dict[str, Any]], my_needs: Dict[str, int]) -> str:
    try:
        from openai import OpenAI
        if not os.getenv("OPENAI_API_KEY"):
            return ""
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        client = OpenAI()
        brief = {
            "roster_needs": my_needs,
            "top_candidates": [
                {k: v for k, v in c.items() if k in ("player_name", "position", "team", "rank", "score")}
                for c in candidates[:5]
            ],
        }
        res = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a concise fantasy draft assistant. Offer a 2-3 sentence justification."},
                {"role": "user", "content": f"Given roster needs {brief['roster_needs']} and candidates {brief['top_candidates']}, pick ONE best and explain briefly."},
            ],
            temperature=0.2,
            max_tokens=120,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"(LLM narration unavailable: {e})"


# ---------------- Helpers (robust name/alias handling) ----------------
SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?$", re.IGNORECASE)

def normalize_name(name: str) -> str:
    """Lowercase, remove accents, punctuation, spaces, parentheticals, and suffixes (jr/iii)."""
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    for ch in ["'", ".", "-", ",", "’"]:
        s = s.replace(ch, "")
    s = re.sub(r"\s+\(.*?\)\s*", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = SUFFIX_RE.sub("", s).strip()
    s = s.replace(" ", "")
    return s

def last_name_norm(full: str) -> str:
    """Normalized last name for backstop matching."""
    if not full:
        return ""
    base = re.sub(r"\s*\(.*?\)\s*", " ", full).strip()
    parts = base.split()
    if not parts:
        return ""
    last = parts[-1]
    last = SUFFIX_RE.sub("", last.lower()).strip()
    last = unicodedata.normalize("NFKD", last)
    last = "".join(ch for ch in last if not unicodedata.combining(ch))
    last = re.sub(r"[^\w]", "", last)
    return last

def snake_pick_to_slot(pick_index_zero_based: int, num_teams: int) -> int:
    """Return draft slot (0..num_teams-1) for a pick index in snake format."""
    if num_teams <= 0:
        return 0
    r = pick_index_zero_based // num_teams
    k = pick_index_zero_based % num_teams
    return k if r % 2 == 0 else (num_teams - 1 - k)

def map_players_dict(players: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for pid, p in players.items():
        name = p.get("full_name") or p.get("last_name") or p.get("first_name") or ""
        pos = p.get("position")
        team = p.get("team")
        out[pid] = {
            "player_id": pid,
            "player_name": name,
            "name_norm": normalize_name(name),
            "position": "DST" if (pos in {"DEF", "D/ST", "DST"}) else pos,
            "team": team,
        }
    return out

def anti_join(left: pd.DataFrame, right: pd.DataFrame, on: list) -> pd.DataFrame:
    """Return rows from `left` with no match in `right` on the key columns `on`."""
    if right.empty or not on or left.empty:
        return left
    rhs = right[on].drop_duplicates()
    merged = left.merge(rhs, how="left", on=on, indicator=True)
    out = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
    return out


# ---------------- Excel builder ----------------
def build_spreadsheet(out_path: str, picks_df: pd.DataFrame, available_df: pd.DataFrame,
                      my_df: pd.DataFrame, recs_df: pd.DataFrame, warnings: pd.DataFrame = None):
    with pd.ExcelWriter(out_path, engine="openpyxl", mode="w") as xw:
        if not picks_df.empty:
            picks_df.to_excel(xw, sheet_name="Claimed_All", index=False)
            ws = xw.book["Claimed_All"]; bold_headers(ws); autosize(ws)

        for pos in ["QB","RB","WR","TE","K","DST"]:
            sub = picks_df[picks_df["position"]==pos] if not picks_df.empty else pd.DataFrame(columns=picks_df.columns if not picks_df.empty else [])
            sub.to_excel(xw, sheet_name=f"Claimed_{pos}", index=False)
            ws = xw.book[f"Claimed_{pos}"]; bold_headers(ws); autosize(ws)

        if not available_df.empty:
            available_df.to_excel(xw, sheet_name="Available_By_Rank", index=False)
            ws = xw.book["Available_By_Rank"]; bold_headers(ws); autosize(ws)
            tier_col_idx = None
            for i, c in enumerate([cell.value for cell in ws[1]], start=1):
                if str(c).lower()=="tier": tier_col_idx = i
            if tier_col_idx: apply_tier_colors(ws, tier_col_idx)

        if not my_df.empty:
            my_df.to_excel(xw, sheet_name="My_Roster", index=False)
            ws = xw.book["My_Roster"]; bold_headers(ws); autosize(ws)

        if warnings is not None and not warnings.empty:
            warnings.to_excel(xw, sheet_name="Warnings", index=False)
            ws = xw.book["Warnings"]; bold_headers(ws); autosize(ws)

        if not recs_df.empty:
            recs_df.to_excel(xw, sheet_name="Recommendations", index=False)
            ws = xw.book["Recommendations"]; bold_headers(ws); autosize(ws)
            tier_col_idx = None
            for i, c in enumerate([cell.value for cell in ws[1]], start=1):
                if str(c).lower()=="tier": tier_col_idx = i
            if tier_col_idx: apply_tier_colors(ws, tier_col_idx)


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Sleeper Draft Assistant")
    ap.add_argument("--username", required=True)
    ap.add_argument("--season", default="2025")
    ap.add_argument("--league-id")
    ap.add_argument("--draft-id")
    ap.add_argument("--rankings", required=True)
    ap.add_argument("--out", default="draft_board.xlsx")
    ap.add_argument("--poll", type=int, default=7)
    ap.add_argument("--use-openai", action="store_true")
    ap.add_argument("--profile", choices=["balanced","hero_rb","zero_rb","late_qb"], default="balanced")
    ap.add_argument("--debug", action="store_true", help="Print debug info each update")
    args = ap.parse_args()

    load_dotenv()

    # ---------------- Rankings (normalize WR1/RB2 → WR/RB; keep number as tier if tier missing) ----------------
    rank_df = load_rankings_csv(args.rankings).copy()

    # safety: ensure expected columns exist
    for col in ["player_name","position","team","rank","bye","adp","tier","sleeper_tag","sleeper_id"]:
        if col not in rank_df.columns:
            rank_df[col] = None

    # Normalize position letters & pull numeric suffix (tier if missing)
    pos_raw = rank_df["position"].fillna("").astype(str).str.upper()
    pos_letters = pos_raw.str.replace(r"[^A-Z/]", "", regex=True).replace({"DEF": "DST", "D/ST": "DST"})
    tier_from_pos = pos_raw.str.extract(r"(\d+)", expand=False)

    if "tier" not in rank_df.columns:
        rank_df["tier"] = tier_from_pos
    else:
        rank_df["tier"] = rank_df["tier"].fillna(tier_from_pos)

    rank_df["position"] = pos_letters
    rank_df["team"] = rank_df["team"].fillna("")

    # helper columns for matching
    rank_df["name_norm"] = rank_df["player_name"].apply(normalize_name)
    rank_df["last_norm"] = rank_df["player_name"].apply(last_name_norm)
    rank_df["team_up"]   = rank_df["team"].astype(str).str.upper()
    rank_df["_sid_str"]  = rank_df["sleeper_id"].astype(str)

    # Ensure numeric columns for scoring
    for col in ["rank", "adp", "bye", "tier"]:
        if col not in rank_df.columns:
            rank_df[col] = None
        rank_df[col] = pd.to_numeric(rank_df[col], errors="coerce")

    # Safe defaults
    rank_df["rank"] = rank_df["rank"].fillna(10_000)
    rank_df["adp"]  = rank_df["adp"].fillna(999.9)
    rank_df["bye"]  = rank_df["bye"].fillna(0)
    rank_df["tier"] = rank_df["tier"].fillna(99)

    # ---------------- User & Draft/League ----------------
    user = get_user(args.username)
    my_user_id = user.get("user_id")

    draft_id = args.draft_id
    league_id = args.league_id
    league = None
    if draft_id:
        draft = get_draft(draft_id)
        lid = draft.get("league_id")
        if league_id is None and lid: league_id = lid
        if league_id: league = get_league(league_id)
    else:
        if not league_id:
            leagues = get_user_leagues(my_user_id, args.season)
            if not leagues: raise SystemExit("No leagues found for that user/season.")
            league = leagues[0]; league_id = league["league_id"]
        else:
            league = get_league(league_id)
        drafts = get_league_drafts(league_id)
        if not drafts: raise SystemExit("No drafts found for this league.")
        draft = drafts[0]; draft_id = draft["draft_id"]

    league_users = get_league_users(league_id) if league_id else []
    rosters = get_league_rosters(league_id) if league_id else []
    settings = draft.get("settings", {}) if isinstance(draft, dict) else {}
    num_teams = league.get("total_rosters") if league and league.get("total_rosters") else settings.get("teams") or (len(rosters) if rosters else 12)
    roster_positions = league.get("roster_positions", []) if league else ["QB","RB","RB","WR","WR","TE","FLEX","K","DST"]
    total_rounds = settings.get("rounds") or (sum(1 for p in roster_positions if p not in ("BN","TAXI","IR")) + 6)


    # Build mapping for slots (real leagues may map to roster_id; mocks to user_id)
    order = draft.get("order") or draft.get("draft_order") or {}
    slot_to_roster: Dict[int,int] = {}
    roster_to_slot: Dict[int,int] = {}
    slot_to_user: Dict[int,str] = {}
    user_to_slot: Dict[str,int] = {}

    if isinstance(order, dict) and order:
        for k, val in order.items():
            try: slot_zero = int(k) - 1
            except: slot_zero = int(k)
            if isinstance(val, int):
                slot_to_roster[slot_zero] = val; roster_to_slot[val] = slot_zero
            else:
                slot_to_user[slot_zero] = str(val); user_to_slot[str(val)] = slot_zero

    if not slot_to_roster and rosters:
        rosters_sorted = sorted(rosters, key=lambda r: int(r.get("roster_id", 0)))
        slot_to_roster = {i: r["roster_id"] for i, r in enumerate(rosters_sorted)}
        roster_to_slot = {r["roster_id"]: i for i, r in enumerate(rosters_sorted)}

    # My roster_id (reals) and my slot (mocks/reals)
    my_roster_id = None
    for r in rosters:
        if r.get("owner_id") == my_user_id:
            my_roster_id = r.get("roster_id"); break
    if my_roster_id is None and rosters:
        my_roster_id = rosters[0].get("roster_id")

    my_slot = roster_to_slot.get(my_roster_id) if (my_roster_id is not None and roster_to_slot) else None
    if my_slot is None and user_to_slot:
        my_slot = user_to_slot.get(my_user_id)

    # Players + indexes to attach sleeper_id to rankings
    players = get_players()
    players_map = map_players_dict(players)

    name_pos_to_id = {}
    last_team_pos_to_id = {}
    last_pos_to_id = {}
    for pid, info in players_map.items():
        nm = info.get("name_norm", "")
        pos = (info.get("position") or "").upper()
        tm  = (info.get("team") or "").upper()
        if nm and pos:
            name_pos_to_id[(nm, pos)] = pid
        last = last_name_norm(info.get("player_name", ""))
        if last:
            if tm and pos:
                last_team_pos_to_id[(last, tm, pos)] = pid
            if pos and (last, pos) not in last_pos_to_id:
                last_pos_to_id[(last, pos)] = pid  # crude backstop

    # Fill missing sleeper_id via multiple passes
    mask_missing = ~rank_df["sleeper_id"].astype(str).str.strip().astype(bool)
    for idx in rank_df[mask_missing].index:
        nm = rank_df.at[idx, "name_norm"]
        pos = (rank_df.at[idx, "position"] or "").upper()
        tm  = (str(rank_df.at[idx, "team"]) or "").upper()
        sid = name_pos_to_id.get((nm, pos))
        if not sid:
            last = rank_df.at[idx, "last_norm"]
            if last and tm and pos:
                sid = last_team_pos_to_id.get((last, tm, pos))
            if not sid and last and pos:
                sid = last_pos_to_id.get((last, pos))
        if sid:
            rank_df.at[idx, "sleeper_id"] = sid

    # Extra fill: name-only unique mapping
    name_to_pids = {}
    for pid, info in players_map.items():
        nm = info.get("name_norm", "")
        if not nm:
            continue
        name_to_pids.setdefault(nm, set()).add(pid)
    unique_name_to_pid = {nm: list(pids)[0] for nm, pids in name_to_pids.items() if len(pids) == 1}

    mask_missing2 = ~rank_df["sleeper_id"].astype(str).str.strip().astype(bool)
    for idx in rank_df[mask_missing2].index:
        nm = rank_df.at[idx, "name_norm"]
        pid = unique_name_to_pid.get(nm)
        if pid:
            rank_df.at[idx, "sleeper_id"] = pid

    # refresh helper cols after fills
    rank_df["_sid_str"] = rank_df["sleeper_id"].astype(str)

    print(f"Watching draft {draft_id} in league {league_id} for user {args.username}.")
    print(f"Slot→Roster: {slot_to_roster}")
    print(f"Slot→User:   {slot_to_user}")
    print(f"My user_id={my_user_id}, my roster_id={my_roster_id}, my slot={my_slot}")
    print(f"Writing board to: {args.out}")

    last_pick_count = -1

    # -------- Helper to build robust but safe available list --------
    def build_available(rank_df_in: pd.DataFrame, picks_df_in: pd.DataFrame) -> pd.DataFrame:
        if picks_df_in.empty:
            return rank_df_in.copy().reset_index(drop=True)

        claimed = picks_df_in.copy()
        # normalize claimed keys (strip digits from positions if any leak in)
        claimed_pos_raw = claimed["position"].astype(str).str.upper()
        claimed["position"] = claimed_pos_raw.str.replace(r"[^A-Z/]", "", regex=True).replace({"DEF":"DST","D/ST":"DST"})
        claimed["name_norm"] = claimed["player_name"].apply(normalize_name)
        claimed["last_norm"] = claimed["player_name"].apply(last_name_norm)
        claimed["team_up"]   = claimed["team"].fillna("").astype(str).str.upper()
        claimed["_sid_str"]  = claimed["player_id"].astype(str)

        # Start from full rankings
        avail = rank_df_in.copy()

        # 1) Remove by exact id
        avail = avail[~avail["_sid_str"].isin(set(claimed["_sid_str"]))]

        # 2) Remove by (name_norm, position)
        avail = anti_join(avail, claimed, on=["name_norm", "position"])

        # 3) Remove by (last_norm, team_up, position)
        avail = anti_join(avail, claimed, on=["last_norm", "team_up", "position"])

        # 4) Remove by (last_norm, position)
        avail = anti_join(avail, claimed, on=["last_norm", "position"])

        avail = avail.drop(columns=["_sid_str"], errors="ignore").reset_index(drop=True)

        # Safety net: if pool got nuked due to extreme aliasing, fall back to ID-only
        if avail.empty:
            fallback = rank_df_in[~rank_df_in["_sid_str"].isin(set(claimed["_sid_str"]))].copy().reset_index(drop=True)
            return fallback

        return avail

    while True:
        picks = get_draft_picks(draft_id) or []
        pick_count = len(picks)

        # Learn my_slot dynamically from my first pick in mocks (if still None)
        if my_slot is None:
            for p in picks:
                picked_by = p.get("picked_by")
                if picked_by and str(picked_by) == str(my_user_id) and p.get("draft_slot"):
                    my_slot = int(p["draft_slot"]) - 1
                    break

        if pick_count != last_pick_count:
            last_pick_count = pick_count

            # -------- CLAIMED from live picks --------
            rows = []
            for i, p in enumerate(picks):
                pid = str(p.get("player_id"))
                pl = players_map.get(pid, {"player_name":"Unknown","name_norm":"", "position":"", "team":""})
                draft_slot_1b = p.get("draft_slot")
                slot_zero = (int(draft_slot_1b)-1) if draft_slot_1b else None

                roster_id = p.get("roster_id")
                if roster_id is None:
                    roster_id = slot_to_roster.get(slot_zero) if slot_zero is not None else None
                    if roster_id is None and slot_zero is not None:
                        roster_id = slot_zero  # synthetic in mocks

                rows.append({
                    "overall": i+1,
                    "round": p.get("round"),
                    "pick_in_round": None,
                    "roster_id": roster_id,
                    "slot": slot_zero,
                    "picked_by": p.get("picked_by"),
                    "player_id": pid,
                    "player_name": pl.get("player_name"),
                    "name_norm": pl.get("name_norm",""),
                    "last_norm": last_name_norm(pl.get("player_name","")),
                    "position": "DST" if pl.get("position") in {"DEF","D/ST","DST"} else pl.get("position"),
                    "team": pl.get("team"),
                })

            expected_cols = ["overall","round","pick_in_round","roster_id","slot","picked_by","player_id","player_name","name_norm","last_norm","position","team"]
            picks_df = pd.DataFrame(rows, columns=expected_cols)
            if not picks_df.empty:
                picks_df["pick_in_round"] = picks_df["overall"].apply(lambda x: ((x-1) % num_teams) + 1)

            # -------- AVAILABLE = rankings MINUS claimed (robust & safe) --------
            avail_df = build_available(rank_df, picks_df)

            if args.debug:
                sample = ", ".join(avail_df.head(3)["player_name"].tolist()) if not avail_df.empty else "NONE"
                print(f"[DEBUG] claimed={len(picks_df)}; available={len(avail_df)}; sample={sample}")

            # Fill/repair team from players_map if missing (cosmetic)
            if not avail_df.empty:
                fill_teams = []
                for sid, nm in zip(avail_df["sleeper_id"], avail_df["name_norm"]):
                    info = None
                    if pd.notna(sid) and str(sid).strip():
                        info = players_map.get(str(sid))
                    fill_teams.append(info.get("team") if info else None)
                if "team" not in avail_df.columns:
                    avail_df["team"] = fill_teams
                else:
                    avail_df["team"] = avail_df["team"].fillna(pd.Series(fill_teams))

            # Last resort: never let pool be empty so we always show recs
            if avail_df.empty:
                avail_df = rank_df.sort_values("rank").head(50).copy().reset_index(drop=True)

            # -------- My picks (slot for mocks, roster_id for reals) --------
            if "slot" in picks_df.columns and my_slot is not None:
                my_picks_df = picks_df[picks_df["slot"] == my_slot].copy()
            elif my_roster_id is not None:
                my_picks_df = picks_df[picks_df["roster_id"] == my_roster_id].copy()
            else:
                my_picks_df = pd.DataFrame(columns=expected_cols)

            # -------- Whose turn? --------
            total_picks_made = pick_count
            on_clock_slot = snake_pick_to_slot(total_picks_made, num_teams)
            on_clock_roster = slot_to_roster.get(on_clock_slot)
            on_clock_user = None
            if not on_clock_roster and (on_clock_slot in slot_to_user):
                on_clock_user = slot_to_user[on_clock_slot]

            if my_slot is not None:
                my_turn = (on_clock_slot == my_slot)
            elif on_clock_user is not None:
                my_turn = (str(on_clock_user) == str(my_user_id))
            else:
                my_turn = (on_clock_roster == my_roster_id)

            # -------- Recommendations --------
            needs: Dict[str,int] = {}
            recs_df = pd.DataFrame()
            warnings_df = pd.DataFrame(columns=["type","detail"])

            if not avail_df.empty:
                roster_picks = {rid: picks_df[picks_df["roster_id"] == rid] for rid in picks_df["roster_id"].dropna().unique()} if not picks_df.empty else {}
                roster_positions = league.get("roster_positions", []) if league else ["QB","RB","RB","WR","WR","TE","FLEX","K","DST"]

                def needs_for_roster(rid:int) -> Dict[str,int]:
                    rp = roster_picks.get(rid, pd.DataFrame(columns=picks_df.columns))
                    have_pos = rp["position"].value_counts().to_dict() if not rp.empty else {}
                    starters = {pos: roster_positions.count(pos) for pos in set(roster_positions)}
                    starters = {k: starters.get(k,0) for k in ["QB","RB","WR","TE","DST","K"]}
                    return {pos: max(starters.get(pos,0) - have_pos.get(pos,0), 0) for pos in starters.keys()}

                neighbor_needs = {}
                for slot, rid in slot_to_roster.items():
                    if my_slot is not None and slot == my_slot: continue
                    neighbor_needs[slot] = needs_for_roster(rid)

                st = DEFAULT_STRATEGY.copy(); st["profile"] = args.profile

                scored, needs, _ = score_candidates(
                    available_df=avail_df.assign(bye=avail_df.get("bye", None)),
                    my_picks_df=my_picks_df,
                    league_roster_positions=league.get("roster_positions", []) if league else ["QB","RB","RB","WR","WR","TE","K","DST"],
                    num_teams=num_teams,
                    total_picks_made=total_picks_made,
                    my_slot=(my_slot if my_slot is not None else 0),
                    neighbor_slots=list(neighbor_needs.keys()),
                    neighbor_needs=neighbor_needs,
                    strategy=st,
                    total_rounds=total_rounds, 
                )
                recs_df = scored.head(15)

                # Warnings
                if not my_picks_df.empty and "bye" in my_picks_df.columns:
                    byes = my_picks_df["bye"].dropna()
                    if not byes.empty:
                        counts = byes.astype(int).value_counts()
                        for wk, cnt in counts.items():
                            if cnt >= 3:
                                warnings_df.loc[len(warnings_df)] = ["BYE_STACK", f"You have {cnt} starters off in Week {wk}"]

                if "tier" in scored.columns:
                    for pos in ["RB","WR","TE","QB"]:
                        if needs.get(pos,0) > 0:
                            cur = scored[scored["position"]==pos]
                            if not cur.empty:
                                tmin = cur["tier"].min()
                                remain = (cur["tier"]==tmin).sum()
                                if 0 < remain <= 2:
                                    warnings_df.loc[len(warnings_df)] = ["TIER_PRESSURE", f"{pos}: only {int(remain)} left in current tier"]

            # -------- Write Excel --------
            build_spreadsheet(
                args.out,
                picks_df,
                avail_df.sort_values(["rank"]) if not avail_df.empty else avail_df,
                my_picks_df,
                recs_df,
                warnings=warnings_df,
            )

            # Console
            oc_bits = [f"slot {on_clock_slot}"]
            if on_clock_roster is not None: oc_bits.append(f"roster {on_clock_roster}")
            if on_clock_user is not None: oc_bits.append(f"user {on_clock_user}")
            mine = f"slot {my_slot}" if my_slot is not None else (f"roster {my_roster_id}" if my_roster_id is not None else "unknown")
            print("\n--- Draft Update ---")
            print(f"Picks made: {total_picks_made}. On the clock: {', '.join(oc_bits)}. Your {mine}. {'YOUR TURN!' if my_turn else ''}")
            if args.debug:
                print(f"[DEBUG] my_picks: {len(my_picks_df)}; recs: {len(recs_df)}")
            if my_turn and not recs_df.empty:
                top5 = recs_df.head(5)[["player_name","position","team","rank","score"]]
                print("Top 5 suggestions:")
                for i, row in top5.iterrows():
                    print(f"  {i+1}. {row['player_name']} ({row['position']} - {row['team']}) rank={row['rank']} score={row['score']:.2f}")
                if args.use_openai:
                    brief = llm_narration(recs_df.to_dict(orient="records"), needs)
                    if brief:
                        print("LLM says:", brief)

        time.sleep(args.poll)


if __name__ == "__main__":
    main()
