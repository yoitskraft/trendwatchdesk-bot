# TrendWatchDesk Auto Poster (dlvr.it + RSS) — with Logo

Generates a daily image + caption and updates an RSS feed that dlvr.it posts to Instagram Business.
Place your logo file named **logo.png** in the repo root; it will be added to each image automatically (top-right).

## Setup

1. Create a new GitHub repo named `trendwatchdesk-bot`.
2. Enable GitHub Pages: Settings → Pages → Source: `main` / `docs`.
3. Edit `.github/workflows/daily.yml` and replace `<your-username>` in `PAGES_BASE`.
4. Add your `logo.png` to the repo root (transparent PNG recommended).
5. Go to **Actions** and run the workflow once.
6. Your feed: `https://<your-username>.github.io/trendwatchdesk-bot/feed.xml`.
7. In dlvr.it:
   - Add Social → Instagram Business (connect via Facebook Page).
   - Add Feed → paste the feed URL.
   - Map description → caption; image → photo.

## Local test
```bash
pip install -r requirements.txt
PAGES_BASE=https://<your-username>.github.io/trendwatchdesk-bot python main.py
```
