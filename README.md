# DraftHelper
Sleeper Draft Assistant 🏈

A Python draft helper that connects to Sleeper (mock or real drafts), tracks claimed players, maintains an up-to-date Available board from your rankings CSV, and gives position-aware, strategy-driven pick recommendations. It’s restart-safe mid-draft and exports a live Excel draft board.

Features

Works with mock & real Sleeper drafts (start anytime, even mid-draft)

Removes taken players using robust matching (by Sleeper ID, name+pos, last name+team+pos, last name+pos)

Strategy engine that values RB/WR/TE early and guarantees K/DST (and all starters) are filled before you run out of picks

Recommendations refresh every few seconds (configurable)

Excel output: Claimed by position, Available by rank, My roster, Recommendations, Warnings

CSV-friendly: Accepts WR1/RB2 styles (auto-normalized to WR/RB) and keeps numeric part as tier

OpenAI narration (optional): terse “why this pick” blurbs (--use-openai)

Debug mode to understand filtering & recs (--debug)

One-line run command
python draft_assistant.py --username YOUR_SLEEPER_USERNAME --draft-id YOUR_DRAFT_ID --rankings path/to/rankings.csv --out draft_board.xlsx --debug


Example:

python draft_assistant.py --username NutStorage --draft-id 1266919498678534144 --rankings data/rankings_fp_clean.csv --out draft_board.xlsx --debug

Requirements

Python 3.10+

Packages

pandas

requests

openpyxl

python-dotenv

(optional) openai (only if using --use-openai)

Files in the repo (same folder):

draft_assistant.py (main script)

sleeper_api.py (Sleeper HTTP helpers)

rankings_loader.py (CSV loader)

strategy.py (scoring; includes must-fill K/DST logic)

excel_format.py (styling helpers)

Install deps:

pip install pandas requests openpyxl python-dotenv openai

Setup

Find your Sleeper draft_id

Mock draft: open your mock room in a browser; copy the ID from the URL (looks like a long number).

Real draft: open your league’s Draft room; copy the ID from the URL (/draft/nfl/{draft_id}).

Prepare your rankings CSV

Expected header (order can vary):

player_name,position,team,rank,bye,adp,tier,sleeper_tag,sleeper_id


Examples (these are valid):

Ja'Marr Chase,WR1,CIN,1,10.0,,,0,
Bijan Robinson,RB1,ATL,2,5.0,,,0,
Saquon Barkley,RB2,PHI,3,9.0,,,0,


Notes:

position like WR1/RB2 is fine — the assistant normalizes it to WR/RB and uses the number for tier if you didn’t provide one.

sleeper_id is optional; the assistant attempts to auto-map.

(Optional) OpenAI narration

Create a .env file next to the script:

OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini


Add --use-openai when running to enable short justification blurbs in the console.

How to run

Basic (most users):

python draft_assistant.py --username YOUR_SLEEPER_USERNAME --draft-id YOUR_DRAFT_ID --rankings path/to/rankings.csv --out draft_board.xlsx --debug


Useful flags:

--profile balanced|hero_rb|zero_rb|late_qb (default: balanced)

--poll 7 seconds between refreshes (default: 7)

--use-openai enable narration if you set OPENAI_API_KEY

--league-id (optional; not needed if you already have --draft-id)

Restart-safe: If the script stops, just run it again with the same args. It rebuilds state from Sleeper and resumes.

What you’ll see

Console:

Who’s on the clock, your slot, top 5 recommendations

[DEBUG] counts for claimed/available and quick samples

Excel (draft_board.xlsx) with sheets:

Claimed_All, Claimed_QB/RB/WR/TE/K/DST

Available_By_Rank

My_Roster

Recommendations

Warnings (bye-week stacks, tier pressure, etc.)

Strategy (why recommendations make sense)

Early rounds: prioritize RB/WR/TE/QB (K/DST are heavily penalized).

Mid/late rounds: value keeps driving choices with roster-need nudges.

Must-fill rule: if your remaining picks equal your remaining required starters, only starter-needed positions are allowed (this guarantees you’ll draft K and DST before the draft ends).

Soft caps prevent position hoarding (e.g., max RB/WR depth).

Small bonus for first bench at a position once starters are filled.

You can tweak dials in strategy.py:

early_tax_round/early_tax_points (delay K/DST)

starter_need_bonus, bench_first_bonus

max_cap, over_cap_penalty

Notes & tips

Mock vs Real: Same command; both are supported. Just use the correct draft_id.

Start mid-draft: Absolutely — the assistant fetches all historical picks and filters your rankings accordingly.

If taken players appear in “Available”:

Run with --debug and check the console sample.

Make sure your CSV has player_name, position, team filled and reasonable (team as standard NFL abbreviations).

The script already normalizes WR1 → WR; you don’t need to edit the CSV for that.

Troubleshooting

No recommendations:
Usually means your available pool went empty. The script now has a safe fallback; ensure you’re on the latest code and run with --debug. You should see non-empty available counts.

Excel won’t open:
Close the file if it’s already open while the script tries to write it. It rewrites on each update.

“Your roster None”:
Fixed in current code. If you still see it, you can still draft: recommendations don’t depend on that string; it’s just a display label. But share your console log if it persists.

License

MIT — do what you like; attribution appreciated.

Acknowledgments

Sleeper public endpoints for draft and player data

Community rankings CSVs (e.g., FantasyPros) — please follow their licensing/usage terms

Quick start (copy/paste)
python draft_assistant.py --username YOUR_SLEEPER_USERNAME --draft-id YOUR_DRAFT_ID --rankings path/to/rankings.csv --out draft_board.xlsx --debug


Happy drafting! 🏆
