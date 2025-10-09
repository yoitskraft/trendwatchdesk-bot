# TrendWatchDesk

_Last updated: 2025-10-09_

TrendWatchDesk auto-generates **Instagram-ready market content**:
- 📊 **Charts**: 6 clean, 1080×1080 candlestick charts with a single support zone, 3× per week.
- 📝 **Captions**: Human-style, news-aware lines with light technical cues + emojis.
- 📰 **News Posters**: All-caps headlines + short paragraphs with faint themed backgrounds, triggered by cross-confirmed breaking news.

## How it works

- **`main.py`** is the single entrypoint:
  - Mode `charts`: generates images into `output/` and writes a daily caption.
  - Mode `posters`: scans news, renders a news poster into `output/posters/` when criteria pass.
- **GitHub Actions** automate runs and publish artifacts to dedicated branches:
  - `charts` branch → chart PNGs + captions.
  - `posters` branch → news posters + dedupe state.

## Workflows

- `.github/workflows/daily.yml`  
  Runs **Mon/Wed/Fri 07:10 UTC**. Publishes chart images + caption to `charts` branch.

- `.github/workflows/breaking-posters.yml`  
  Runs **every 10 minutes**. Publishes 0–1 posters (if cross-confirmed news) to `posters` branch.

- `.github/workflows/docs-guard.yml`  
  **PR guard**: fails if core files change but docs (`README.md`/`OPERATIONS_GUIDE.md`) aren’t updated.

- `.github/workflows/docs-autostamp.yml`  
  **Auto-stamp**: updates the “Last updated” date in docs on pushes to `main`.

## Local run

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade yfinance pillow pandas numpy requests pytz urllib3
TWD_MODE=charts python main.py
# or
TWD_MODE=posters python main.py


---

<!-- TWD_STATUS:BEGIN -->

## Automation Status (auto-generated)
- **Last run:** 2025-10-09 07:57:00 UTC
- **Triggered by:** TWD Breaking Posters (event-driven)
- **Mode:** `posters`   ·  **Timeframe:** `D`
- **Breaking-posters knobs:** recency=720m, min_sources=1, fallback=on, rss=on
- **Watchlist (preview):** AAPL, MSFT, NVDA, AMD, TSLA, SPY, QQQ, GLD, AMZN, META, GOOGL
- **Publish targets:** charts → `charts`, posters → `posters`

<!-- TWD_STATUS:END -->




















