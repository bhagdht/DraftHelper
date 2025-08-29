# Sleeper Draft Assistant (Python)

This tool watches your **Sleeper** draft, writes an **Excel draft board** grouped by position, and suggests the **best available pick** based on live picks + your roster needs. It can run **purely local** (no OpenAI), or optionally use OpenAI for strategy narration.

## Quick Start

1) Install deps:
```bash
pip install -r requirements.txt
```

2) Prepare rankings CSV (example: `data/rankings_example.csv`). You can paste fresh rankings from your favorite source. Columns required:
- `player_name` (string)
- `position` (QB/RB/WR/TE/K/DST)
- `team` (e.g., KC)
- `rank` (integer or float; lower = better)
- `bye` (optional)
- `sleeper_id` (optional; if empty, script tries to match by name/position)

3) Set your environment (optional for OpenAI):
```bash
cp .env.example .env
# then set OPENAI_API_KEY in .env if you want LLM suggestions
```

4) Run:
```bash
python draft_assistant.py --username YOUR_SLEEPER_USERNAME --season 2025   --rankings data/rankings_example.csv --out draft_board.xlsx
```
If you know your `--league-id` or `--draft-id`, pass them for faster start.

### During the draft

- The script polls the Sleeper API (default every 7s), updates the spreadsheet, and prints suggestions.
- It detects whose turn it is using snake-order logic. When it's **your** turn, it prints **Top 5** suggestions and writes them to the `Recommendations` sheet.

### Output

- An Excel file (default `draft_board.xlsx`) with:
  - `Claimed_All`: all picks with round, pick, roster, position
  - `Claimed_QB`, `Claimed_RB`, `Claimed_WR`, `Claimed_TE`, `Claimed_K`, `Claimed_DST`: by position
  - `Available_By_Rank`: remaining players sorted by rank
  - `My_Roster`: your picks so far
  - `Recommendations`: current top suggestions when it’s your turn

## Notes

- The script is **read-only**; no API keys are needed for Sleeper.
- OpenAI is optional. If enabled (`--use-openai`), the model will provide a short justification for the top pick using the latest data already in memory (no web).

---
