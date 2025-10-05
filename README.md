# TrendWatchDesk — IG-Friendly Charts (EMA + Support/Resistance)

This project generates a daily Instagram-friendly image with mini-charts (last 60 days),
EMA(20) overlay, and simple support/resistance levels. It also publishes an RSS feed
that dlvr.it can auto-post to Instagram Business.

## Setup
1. Create a GitHub repo and upload all files from this ZIP.
2. Enable GitHub Pages: **Settings → Pages → Source: `main`, Folder: `/docs`**.
3. Edit `.github/workflows/daily.yml` and replace `<your-username>` in `PAGES_BASE`.
4. (Optional) Add `logo.png` to the repo root for branding (placed top-right).
5. Go to **Actions** → run **TrendWatchDesk Daily** once manually.
6. Feed URL: `https://<your-username>.github.io/trendwatchdesk-bot/feed.xml`

## Notes
- Data source: yfinance (free). If rate-limited, the script falls back and still generates a post.
- Edit the `TICKERS` list in `main.py` to match your watchlist.
- Change schedule in `.github/workflows/daily.yml` via the cron expression.
