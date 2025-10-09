# TrendWatchDesk – Operations Guide (Single Source of Truth)

_Last updated: 2025-10-09 (UTC)_

This guide defines the **canonical behavior** of the TrendWatchDesk automation system.  
It is the **source of truth** for charts, posters, captions, workflows, branches, and assets.  
All scripts and workflows must conform to this specification.  
The **docs-guard** workflow enforces compliance; the **docs-autostamp** workflow keeps this date updated automatically.

---

## 1. Repository Branches

- **`main`**  
  Source of truth for all code, workflows, and documentation. PRs must target this branch.  

- **`charts`**  
  Auto-published artifacts branch. Contains only the generated outputs (`output/*`) from CI runs.  
  - Rewritten each run (force-pushed, single commit).  
  - Used for external access, IG-ready visuals, and archiving runs.

- **`docs`** (optional, only if configured)  
  For static site/docs builds. Syncs automatically from `OPERATIONS_GUIDE.md` and `README.md`.  

---

## 2. Directory Structure

.
├── assets/                 # Static resources
│   ├── logos/              # Per-ticker logo PNGs (color, transparent)
│   │   ├── AAPL.png
│   │   ├── MSFT.png
│   │   └── …
│   ├── fonts/              # Grift fonts
│   │   ├── Grift-Bold.ttf
│   │   └── Grift-Regular.ttf
│   └── brand_logo.png      # White TrendWatchDesk logo
│
├── output/                 # Auto-generated files (cleaned/recreated per run)
│   ├── charts/             # Daily candlestick charts
│   │   ├── AAPL_chart.png
│   │   └── TSLA_chart.png
│   ├── posters/            # News-driven posters + captions
│   │   ├── NVDA_poster_20251009.png
│   │   ├── NVDA_poster_20251009_caption.txt
│   │   └── …
│   ├── caption_20251009.txt # Daily captions for 6 tickers
│   ├── run.log             # Run log (full stdout/stderr)
│   └── .gitkeep            # Keeps folder in repo
│
├── .github/
│   └── workflows/
│       ├── daily.yml        # Main CI workflow (charts + posters)
│       ├── docs-guard.yml   # Validates this guide is followed
│       └── docs-autostamp.yml # Updates timestamp in this guide/README
│
├── main.py                 # Core generator script
├── OPERATIONS_GUIDE.md     # (This file, single source of truth)
└── README.md               # Mirror of guide for quick reference

---

## 3. Outputs

- **Charts** → `output/charts/{TICKER}_chart.png`  
- **Posters** → `output/posters/{TICKER}_poster_YYYYMMDD.png`  
- **Poster captions** → `output/posters/{TICKER}_poster_YYYYMMDD_caption.txt`  
- **Daily captions** → `output/caption_YYYYMMDD.txt`  
- **Run log** → `output/run.log`  
- **Branch publish** → all of `output/*` pushed to `charts` branch.

---

## 4. Charts

- **Source**: Yahoo Finance (`yfinance`), 1y period, interval `1wk`  
- **Type**: Candlesticks (green = up, red = down)  
- **Background**: Blue gradient with subtle beams  
- **Grid**: Disabled  
- **Support zone**: Transparent feathered rectangle (not solid)  
- **Branding**:  
  - Company logo (**color**) → top-left  
  - TWD logo (**white**) → bottom-right  
- **Footer**: White text bottom-left (last price, % change 30d)  
- **Size**: `1080×720`

---

## 5. Posters

- **Purpose**: Convert Yahoo Finance headlines into IG-style posters  
- **Background**: Blue gradient with beams  
- **Headline**: Grift-Bold, uppercase, 1–2 lines  
- **Subtext**: Grift-Regular, 3–4 wrapped lines, tied to sector/price action  
- **Logos**:  
  - Company logo (**color**) top-right  
  - TWD logo (white) bottom-right  
- **NEWS badge**: top-left  
- **Size**: `1080×1080`

---

## 6. Captions

- **Charts**:
  - Must include sector emojis (🍏 AAPL, 🚗 TSLA, 🤖 NVDA, etc.)  
  - Phrases rotated: “momentum building”, “buyers defending support”, “range tightening” etc.  
  - Avoid repeating structure; never always mention price.  

- **Posters**:
  - Must tie to headline context.  
  - Include sector, price action, forward guidance.  
  - End with CTA: “What’s your take? 👇” / “Save this 📌” / “Share 🔄”.

---

## 7. Ticker Pools

Core pools for deterministic selections:

- **AI**: NVDA, MSFT, GOOGL, META, AMZN  
- **MAG7**: AAPL, MSFT, GOOGL, META, AMZN, NVDA, TSLA  
- **Semis**: NVDA, AMD, AVGO, TSM, INTC, ASML  
- **Healthcare**: UNH, JNJ, PFE, MRK, LLY  
- **Fintech**: MA, V, PYPL, SQ, SOFI  
- **Quantum**: IONQ, IBM, AMZN  
- **Robotics**: ISRG, FANUY, IRBT, ABB, ROK  
- **Wildcards**: NFLX, DIS, BABA, NIO, SHOP, PLTR  

---

## 8. Workflows

### `.github/workflows/daily.yml`
- Runs Mon/Wed/Fri 07:10 UTC + manual dispatch  
- Generates **charts** + **posters**  
- Publishes to `charts` branch  
- Uploads artifacts (PNGs + captions)

### `.github/workflows/docs-guard.yml`
- Fails if `main.py` / workflows diverge from this guide  
- Validates **DOCSPEC block**  

### `.github/workflows/docs-autostamp.yml`
- Updates “Last updated” line in this file and README when code changes.

---

## 9. Assets

- **Logos**: `assets/logos/{TICKER}.png` (color, ~512px, transparent)  
- **Fonts**: `assets/fonts/Grift-Bold.ttf`, `assets/fonts/Grift-Regular.ttf`  
- **Brand logo**: `assets/brand_logo.png` (white, transparent)  

---

## 10. Compliance Block

<!-- DOCSPEC:BEGIN -->
```json
{
  "version": "1.0",
  "branches": ["main", "charts", "docs"],
  "outputs": {
    "charts_dir": "output/charts",
    "posters_dir": "output/posters",
    "caption_pattern": "output/caption_YYYYMMDD.txt",
    "run_log": "output/run.log",
    "publish_branch": "charts"
  },
  "charts": {
    "size": [1080, 720],
    "interval": "1wk",
    "style": {
      "candles": true,
      "grid": false,
      "bg": "blue-gradient",
      "support_zone": "feathered-rectangle",
      "ticker_text": false,
      "logo_company": "color-top-left",
      "logo_twd": "white-bottom-right",
      "footer_info": "white-bottom-left"
    }
  },
  "posters": {
    "size": [1080, 1080],
    "news_source": "yahoo-finance",
    "logo_company": "color-top-right",
    "logo_twd": "white-bottom-right",
    "headline_font": "Grift-Bold",
    "subtext_font": "Grift-Regular"
  },
  "ci": {
    "workflow": ".github/workflows/daily.yml",
    "runs": ["charts", "posters"],
    "branch_publish": "charts"
  }
}

<!-- DOCSPEC:END -->
