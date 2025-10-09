# TrendWatchDesk 📊

Automated system for generating **stock market visuals** — candlestick charts and news-driven posters — optimized for **Instagram engagement**.

---

## 🚀 Features

- **Daily/weekly candlestick charts** with support zones
- **News-driven posters** with sector-aware context
- **Auto captions** (IG-ready, with emojis & variety)
- **Deterministic ticker selection** (stable randomness per day)
- **CI publishing** to `charts` branch for easy artifact access

---

## 📂 Repo Structure

```text
.
├── assets/                 # Logos, fonts, brand identity
├── output/                 # Charts + posters (CI artifacts)
├── .github/workflows/      # Automation (daily, docs, CI)
├── main.py                 # Core engine
├── OPERATIONS_GUIDE.md     # Canonical ops reference
├── README.md               # This file
└── update_docs.py          # Script to regen docs
```

---

## ⚙️ Workflows

- `daily.yml` → Generates charts Mon/Wed/Fri 07:10 UTC  
- `ci.yml` → Extended charts + posters run (optional)  
- `docs-autostamp.yml` → Auto-updates timestamps in docs  
- `docs-guard.yml` → Ensures docs compliance  

---

## 📸 Outputs

- **Charts** → `output/charts/` → candlesticks + support zones  
- **Posters** → `output/posters/` → news headlines + sector logos  
- **Captions** → `output/caption_YYYYMMDD.txt`  
- **Run logs** → `output/run.log`  

---

## 📖 Operations Guide

See [OPERATIONS_GUIDE.md](OPERATIONS_GUIDE.md) for full canonical spec.


---

<!-- TWD_STATUS:BEGIN -->

## Automation Status (auto-generated)
- **Last run:** 2025-10-09 13:42:21 UTC
- **Triggered by:** TWD Breaking Posters (event-driven)
- **Mode:** `posters`   ·  **Timeframe:** `D`
- **Breaking-posters knobs:** recency=720m, min_sources=1, fallback=on, rss=on
- **Watchlist (preview):** AAPL, MSFT, NVDA, AMD, TSLA, SPY, QQQ, GLD, AMZN, META, GOOGL
- **Publish targets:** charts → `charts`, posters → `posters`

<!-- TWD_STATUS:END -->
