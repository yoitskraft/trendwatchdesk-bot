import os, io, math, random, datetime, pytz, traceback
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import yfinance as yf

# =========================
# CONFIG
# =========================
BRAND_NAME = "TrendWatchDesk"
TIMEZONE   = "Europe/London"
CANVAS_W, CANVAS_H = 1080, 1080
MARGIN = 40

DEBUG_BOS = False

# Colors / styling
BG        = (255,255,255)
TEXT_MAIN = (20,20,22)
TEXT_MUT  = (145,150,160)
GRID      = (238,241,244)
UP_COL    = (20,170,90)
DOWN_COL  = (230,70,70)

SUPPORT_FILL = (40,120,255,64)
SUPPORT_EDGE = (40,120,255,110)
RESIST_FILL  = (230,70,70,72)
RESIST_EDGE  = (230,70,70,120)

CHART_LOOKBACK   = 252
SUMMARY_LOOKBACK = 30
YAHOO_PERIOD     = "1y"
STOOQ_MAX_DAYS   = 500

POOLS = {
    "AI": ["NVDA","MSFT","GOOG","META","AMD","AVGO","CRM","SNOW","PLTR","NOW"],
    "QUANTUM": ["IONQ","IBM","RGTI","AMZN","MSFT"],
    "MAG7": ["AAPL","MSFT","GOOG","AMZN","META","NVDA","TSLA"],
    "HEALTHCARE": ["UNH","LLY","JNJ","ABBV","MRK"],
    "FINTECH": ["V","MA","PYPL","SQ","SOFI"],
    "SEMIS": ["TSM","ASML","QCOM","INTC","AMD","MU","TXN"]
}
QUOTAS    = [("AI",2), ("MAG7",1), ("HEALTHCARE",1), ("FINTECH",1), ("SEMIS",1)]
WILDCARDS = 2

OUTPUT_DIR = "output"
LOGO_DIR   = "assets/logos"
BRAND_DIR  = "assets"

# =========================
# HTTP with retry
# =========================
def make_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    retry = Retry(total=5, backoff_factor=0.7,
                  status_forcelist=[429,500,502,503,504],
                  allowed_methods=["GET","POST"])
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s
SESS = make_session()

# =========================
# Fonts
# =========================
def load_font(size=42, bold=False):
    pref = "fonts/Roboto-Bold.ttf" if bold else "fonts/Roboto-Regular.ttf"
    if os.path.exists(pref):
        try: return ImageFont.truetype(pref, size)
        except: pass
    fam = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try: return ImageFont.truetype(fam, size)
    except: return ImageFont.load_default()

F_TITLE = load_font(70,  bold=True)
F_PRICE = load_font(46,  bold=True)
F_CHG   = load_font(34,  bold=True)
F_SUB   = load_font(28,  bold=False)
F_META  = load_font(20,  bold=False)

# (… keep the rest of your script unchanged: fetchers, swing_points,
# BOS detection, fetch_one, render_single_post, etc …)

# =========================
# Main
# =========================
if __name__ == "__main__":
    # ✅ Guarantee output folder exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("[debug] OUTPUT_DIR:", os.path.abspath(OUTPUT_DIR))

    now = datetime.datetime.now(pytz.timezone(TIMEZONE))
    datestr = now.strftime("%Y%m%d")

    tickers = ["AAPL","MSFT","NVDA"]  # ← or your sampling logic
    print("[info] selected tickers:", tickers)

    for t in tickers:
        try:
            payload = fetch_one(t)
            print(f"[debug] fetched {t}: payload is {'ok' if payload else 'None'}")
            if not payload:
                print(f"[warn] no data for {t}, skipping")
                continue
            out_path = os.path.join(OUTPUT_DIR, f"twd_{t}_{datestr}.png")
            print(f"[debug] saving {out_path}")
            render_single_post(out_path, t, payload, None)
            print("done:", out_path)
        except Exception as e:
            print(f"[error] failed for {t}: {e}")
            traceback.print_exc()
