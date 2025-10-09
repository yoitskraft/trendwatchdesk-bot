# TrendWatchDesk 📊

Automated system for generating **stock market visuals** — candlestick charts and news-driven posters — optimized for **Instagram engagement**.

_Last updated: 2025-10-09 (UTC)_

---

## 🚀 Features

- **Candlestick charts** (weekly, 6 tickers per run) with support zones  
- **News-driven posters** (Yahoo Finance headlines + clustering)  
- **Captions** — natural, IG-native, emoji-rich, sector-aware  
- **Deterministic selection** of tickers (daily seed)  
- **Blue gradient visuals** with branded layout  
- **Auto CI publishing** → artifacts on `charts` branch  
- **Docs compliance** → enforced by `docs-guard` and `docs-autostamp`  

---

## 📂 Repository Structure

```text
.
├── assets/                 # Logos, fonts, brand identity
│   ├── logos/              # Per-ticker logos (PNG, transparent)
│   ├── fonts/              # Grift-Bold, Grift-Regular
│   └── brand_logo.png      # White TrendWatchDesk logo
│
├── output/                 # CI artifacts (cleaned per run)
│   ├── charts/             # Candlestick chart PNGs
│   ├── posters/            # News poster PNGs
│   ├── caption_YYYYMMDD.txt# Captions per run
│   └── run.log             # Execution log
│
├── .github/workflows/      # Automation configs
│   ├── daily.yml           # Daily chart generation (cron + dispatch)
│   ├── docs-autostamp.yml  # Auto-updates dates in docs
│   ├── docs-guard.yml      # Blocks merge if docs drift
│   └── ci.yml              # Extended charts + posters
│
├── main.py                 # Core logic (charts + posters)
├── OPERATIONS_GUIDE.md     # Canonical spec (single source of truth)
├── README.md               # This file
└── update_docs.py          # Script to regenerate docs
```

---

## 🖼️ Outputs

- **Charts** → `output/charts/TICKER_chart.png`  
- **Posters** → `output/posters/TICKER_poster_DATE.png`  
- **Captions** → `output/caption_YYYYMMDD.txt`  
- **Logs** → `output/run.log`  

All outputs are published to the **`charts` branch** as a clean single commit.  

---

## ⚙️ Workflows

- **daily.yml**  
  - Runs Mon/Wed/Fri 07:10 UTC  
  - Generates charts + caption file  
  - Publishes artifacts to `charts` branch  

- **ci.yml**  
  - Extended run (charts + posters)  
  - Optional for testing / combined runs  

- **docs-autostamp.yml**  
  - Updates `_Last updated:` in docs automatically  

- **docs-guard.yml**  
  - Blocks merge if README or Operations Guide are out of sync  

---

## 📖 Operations Guide

For canonical details, see [OPERATIONS_GUIDE.md](OPERATIONS_GUIDE.md).  
All workflows, captions, outputs, and branches must conform to this guide.

---

## 🛠️ Governance

- **Single source of truth** → `OPERATIONS_GUIDE.md`  
- PRs must update docs if behavior changes  
- Guard workflows enforce compliance  
