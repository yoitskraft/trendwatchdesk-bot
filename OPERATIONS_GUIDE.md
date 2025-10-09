# TrendWatchDesk – Operations Guide (Single Source of Truth)

_Last updated: 2025-10-09 (UTC)_

This guide defines the **canonical behavior** of the TrendWatchDesk automation system.  
It is the **source of truth** for charts, posters, captions, workflows, branches, and assets.  
The **docs-guard** workflow enforces compliance; the **docs-autostamp** workflow keeps this date updated automatically.

---

## 1. Repository Branches

- **`main`**  
  Source of truth for all code, workflows, and documentation. PRs must target this branch.  

- **`charts`**  
  Auto-published artifacts branch. Contains only the generated outputs (`output/*`) from CI runs.  
  - Rewritten each run (force-pushed, single commit).  
  - Used for external access, IG-ready visuals, and archiving runs.

- **`docs`** (optional)  
  For static site/docs builds. Syncs automatically from `OPERATIONS_GUIDE.md` and `README.md`.  

---

## 2. Directory Structure

```text
.
├── assets/                 # Static resources
│   ├── logos/              # Per-ticker logo PNGs (color, transparent)
│   ├── fonts/              # Grift fonts
│   └── brand_logo.png      # White TrendWatchDesk logo
│
├── output/                 # Auto-generated files (cleaned per run)
│   ├── charts/             # Daily candlestick charts
│   ├── posters/            # News-driven poster PNGs
│   ├── caption_YYYYMMDD.txt# Captions per run
│   └── run.log             # Execution log
│
├── .github/workflows/      # CI/CD automation
│   ├── daily.yml           # Generates charts on cron/dispatch
│   ├── docs-autostamp.yml  # Updates dates in docs
│   ├── docs-guard.yml      # Enforces docspec compliance
│   └── ci.yml              # Optional combined charts + posters job
│
├── main.py                 # Core logic (charts + posters)
├── OPERATIONS_GUIDE.md     # This file (single source of truth)
├── README.md               # Project overview
└── update_docs.py          # Script to regenerate docs
```

---

## 3. Workflows Overview

- **daily.yml**  
  Runs charts on Mon/Wed/Fri (07:10 UTC). Saves outputs, enforces chart presence.

- **ci.yml**  
  Extended run (charts + posters). Optional but recommended.

- **docs-autostamp.yml**  
  Updates `_Last updated:` automatically when code/docs change.

- **docs-guard.yml**  
  Blocks merge if guide/README drift from spec.

---

## 4. Charts

- 6 tickers selected per run from **watchlist pools** (`AI`, `MAG7`, `Semis`, `Fintech`, `Healthcare`, `Quantum`, `Wildcards`).  
- Data source: Yahoo Finance via `yfinance`.  
- Interval: weekly (`1wk`) for clarity, candlesticks style.  
- Support zone: translucent rectangle (barely visible, but useful).  
- Layout: blue gradient background, white percentage/metadata bottom-left.  
- Logos: ticker logo (color, transparent background).  
- Branding: TrendWatchDesk logo bottom-right.  

Outputs → `output/charts/TICKER_chart.png`

---

## 5. Posters

- Source: Yahoo Finance news polling (lightweight endpoint).  
- Selection: clustered by popularity (same headline ≥2 mentions).  
- Design:  
  - Blue gradient + light beams background.  
  - Headline: Grift-Bold, wrapped to 2–3 lines.  
  - Subtext: Grift-Regular, 3–4 lines, news context + sector sentiment.  
  - Logos: ticker logo top-right, TWD logo bottom-right.  
- Variety: ticker sector-aware emojis, no overlap.  
- Caption file alongside each poster: `output/posters/TICKER_poster_DATE_caption.txt`

Outputs → `output/posters/TICKER_poster_DATE.png`

---

## 6. Captions

- **Daily captions** (charts): summarize ticker momentum, support, % changes.  
  - Must include emojis (sector-aware).  
  - No repetitive structures.  
  - Avoid raw price quoting unless meaningful.  

- **Poster captions**: headline context + investor sentiment + forward guidance.  

Stored in → `output/caption_YYYYMMDD.txt` (daily run)  
and alongside each poster.  

---

## 7. Logging

- Every run logs to `output/run.log`  
- Includes ticker selection, errors, saved paths.  
- Warnings logged but do not halt CI unless no charts are generated.

---

## 8. CI/CD Rules

- Workflows must **fail** if:  
  - No charts generated.  
  - No posters generated when news is available.  

- Artifacts branch (`charts`) always force-pushed with last run results.  

- Docs must always be in sync (guard + autostamp).  

---

## 9. Assets

- Logos → `assets/logos/TICKER.png` (transparent background, color logo).  
- Fonts → Grift-Bold + Grift-Regular (`assets/fonts/`).  
- Brand logo → `assets/brand_logo.png` (white).  

---

## 10. Governance

- **Single source of truth** → this file (`OPERATIONS_GUIDE.md`).  
- Updates here cascade to CI/CD and enforce compliance.  
- All contributors must update this guide for any feature change.  

---


---

<!-- TWD_STATUS:BEGIN -->

## Automation Status (auto-generated)
- **Last run:** 2025-10-09 13:56:38 UTC
- **Triggered by:** TWD Breaking Posters (event-driven)
- **Mode:** `posters`   ·  **Timeframe:** `D`
- **Breaking-posters knobs:** recency=720m, min_sources=1, fallback=on, rss=on
- **Watchlist (preview):** AAPL, MSFT, NVDA, AMD, TSLA, SPY, QQQ, GLD, AMZN, META, GOOGL
- **Publish targets:** charts → `charts`, posters → `posters`

<!-- TWD_STATUS:END -->
