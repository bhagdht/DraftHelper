# Sleeper Draft Assistant 🏈

> A Python helper for **Sleeper** drafts (mock or real) that tracks **claimed players**, maintains an **Available** board from your rankings CSV, and gives **position-aware, strategy-driven pick recommendations**. Restart it anytime—even **mid-draft**—and it catches up automatically. Exports a live **Excel draft board**.

---

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📌 Quick Start

One-line run command:

```bash
python draft_assistant.py --username YOUR_SLEEPER_USERNAME --draft-id YOUR_DRAFT_ID --rankings path/to/rankings.csv --out draft_board.xlsx --debug
