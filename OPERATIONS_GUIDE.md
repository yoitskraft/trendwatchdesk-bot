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
