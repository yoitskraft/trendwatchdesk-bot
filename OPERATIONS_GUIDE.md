# TrendWatchDesk — Operations Guide

This document serves as the **single source of truth** for how the TrendWatchDesk system runs, maintained inside the repository root as `OPERATIONS_GUIDE.md`.

---

## 1. Repository Structure

```
trendwatchdesk-bot/
├── main.py                 # Core script: ticker selection, chart generation, caption generation, posters
├── assets/
│   ├── logos/              # Ticker logos (PNG, ~180px)
│   └── brand_logo.png      # TrendWatchDesk brand logo
├── output/                 # Auto-generated IG chart PNGs + captions
├── posters/                # Auto-generated poster-style news posts
└── .github/
    └── workflows/
        ├── daily.yml       # Mon/Wed/Fri chart runs
        └── posters.yml     # Breaking news poster generation
```

---

## 2. Workflows

### `daily.yml` — Chart Generation

- Runs **Mon/Wed/Fri** via cron.  
- Produces **6 square 1080×1080 IG-ready charts**.  
- Steps:
  1. Checkout repo
  2. Install deps (`yfinance`, `pandas`, `matplotlib`, `Pillow`, `pytz`, `requests`)
  3. Run `main.py`
  4. Save `output/*.png` and `output/caption_YYYYMMDD.txt` as artifacts
  5. Push all artifacts to `charts` branch (force, single commit per run)

### `posters.yml` — Poster Generation

- Runs **every 3h** AND manually triggered.  
- Produces **poster-style IG post** if breaking news detected.  
- News verification: headline must be visible on **≥3 credible outlets** (WSJ, FT, Bloomberg, Yahoo Finance, CNBC).  
- Outputs:
  - PNG saved to `/posters`
  - File committed to **`posters` branch** (single commit)
  - Artifact upload for immediate download

---

## 3. Chart Posts

- **Data source**: Yahoo Finance (via `yfinance`).
- **Timeframe**: Weekly candles (1y lookback).  
- **Visuals**:
  - Clean white background (no card border)
  - Candlesticks: green up / red down
  - Support zone: **blue shaded box** nearest to last price  
    - If uptrend → previous swing high = new support  
    - If downtrend → last swing low = possible bounce  
  - Brand logo bottom-right, ticker logo top-right  
  - Footer: support/resistance + “Not financial advice”

---

## 4. Poster-Style News

- Generated once every 3h (cron) or triggered on verified breaking news.  
- **Layout**:
  - Background: stock-related image @ 15% opacity (e.g. AMD → chips, META → Zuckerberg, Gold → bars)  
  - Headline in ALL CAPS  
  - Subheading = summary line  
  - Body = 1 engaging paragraph (not bullets)  
  - Logos: TrendWatchDesk brand bottom-right, ticker logo top-right  
- Output saved in `/posters` and pushed to `posters` branch.

---

## 5. Captions

- Human-like, not repetitive.  
- Blend **recent news + price action**.  
- Example style:

```
🧠 META — META TO DEEPEN AI CHIP PUSH … breakout pressure building 🚀; momentum looks strong 🔥; could have more room if momentum sticks ✅

🖥️ AMD — STRONG GPU DEMAND IN HEADLINES … testing overhead supply 🧱; momentum looks strong 🔥; watch for follow-through 🔎
```

- CTAs rotate randomly:
  - 💬 Drop your thoughts below!
  - 📌 Save this for later
  - 📊 Which ticker should we cover next?

---

## 6. Branches

- **main** → codebase  
- **charts** → all chart PNGs + captions (force-pushed each run, clean single commit)  
- **posters** → all poster-style news posts (force-pushed each run, clean single commit)  

---

## 7. Automation Strategy

- **dlvr.it** (or similar) pulls directly from `charts` branch (for carousels) and `posters` branch (for news).  
- Charts = 3× weekly, Posters = every 3h or on breaking news.  
- IG feed = mix of both, with captions from `output/caption_YYYYMMDD.txt`.

---

## 8. Known Good Build (Stable)

- This document reflects the **most stable build to date**:  
  - Charts render correctly (scaled text/logos, polished layout)  
  - Captions natural & news-linked  
  - Posters functional (backgrounds + paragraphs + logos)  
  - Branch outputs stable for automation

---

## 9. Future Enhancements

- Auto stories for breaking news (vertical 1080×1920)  
- Multi-language captions  
- AI-assisted summarization tuned for engagement  

---

**TrendWatchDesk Operations Guide v1.0**  
_Last updated: Oct 2025_
