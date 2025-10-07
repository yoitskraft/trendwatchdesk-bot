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

DEBUG_BOS = False  # set True for a visible TEST line if no BoS

# Colors / styling
BG        = (255,255,255)
TEXT_MAIN = (20,20,22)
TEXT_MUT  = (145,150,160)
GRID      = (238,241,244)            # lighter grid
UP_COL    = (20,170,90)
DOWN_COL  = (230,70,70)

# Softer zones (≈25–28% opacity)
SUPPORT_FILL = (40,120,255,64)
SUPPORT_EDGE = (40,120,255,110)
RESIST_FILL  = (230,70,70,72)
RESIST_EDGE  = (230,70,70,120)

# Show ~1 year of daily bars
CHART_LOOKBACK   = 252
SUMMARY_LOOKBACK = 30
YAHOO_PERIOD     = "1y"
STOOQ_MAX_DAYS   = 500

# Pools / quotas (8 images/run: quotas + 2 wildcards)
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
LOGO_DIR   = "assets/logos"   # company logos: <TICKER>.png
BRAND_DIR  = "assets"         # brand_logo.png or logo.png

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
F_SUB   = load_font(28,  bold=False)   # also BOS label
F_META  = load_font(20,  bold=False)   # tiny captions + BOS timeframe

# =========================
# Helpers
# =========================
def y_map(v, vmin, vmax, y0, y1):
    if vmax - vmin < 1e-6: return (y0 + y1)//2
    return int(y1 - (v - vmin) * (y1 - y0) / (vmax - vmin))

def case_insensitive_find(directory, base_no_ext):
    if not os.path.isdir(directory): return None
    for fn in os.listdir(directory):
        stem, ext = os.path.splitext(fn)
        if stem.upper() == base_no_ext.upper() and ext.lower() in (".png",".jpg",".jpeg",".webp"):
            return os.path.join(directory, fn)
    return None

def find_ticker_logo_path(ticker):
    exact = os.path.join(LOGO_DIR, f"{ticker}.png")
    if os.path.exists(exact): return exact
    return case_insensitive_find(LOGO_DIR, ticker)

def find_brand_logo_path():
    candidates = [os.path.join(BRAND_DIR, "brand_logo.png"), os.path.join(BRAND_DIR, "logo.png")]
    for p in candidates:
        if os.path.exists(p): return p
    if os.path.isdir(BRAND_DIR):
        for fn in os.listdir(BRAND_DIR):
            low = fn.lower()
            if ("logo" in low) and os.path.splitext(low)[1] in (".png",".jpg",".jpeg"):
                return os.path.join(BRAND_DIR, fn)
    return None

def sample_with_quotas_and_wildcards(quotas, wildcards, pools, seed):
    rnd = random.Random(seed)
    chosen = []
    def pick_from(category, k):
        src = list(pools.get(category, []))
        rnd.shuffle(src)
        for t in src:
            if t not in chosen:
                chosen.append(t); k -= 1
                if k == 0: break
    for cat, q in quotas: pick_from(cat, q)
    universe = []
    for arr in pools.values():
        for t in arr:
            if t not in universe: universe.append(t)
    rnd.shuffle(universe)
    for t in universe:
        if len(chosen) >= sum(q for _, q in quotas) + wildcards: break
        if t not in chosen: chosen.append(t)
    return chosen

# =========================
# Indicators & S/R primitives
# =========================
def atr(df, n=14):
    high = df["High"]; low = df["Low"]; close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def sma(series, n): return series.rolling(n).mean()

def swing_points(df, w=2):
    """Return lists of pivot lows and highs as (i, timestamp, price)."""
    highs, lows = [], []
    H, L = df["High"], df["Low"]
    for i in range(w, len(df)-w):
        if H.iloc[i] == H.iloc[i-w:i+w+1].max():
            highs.append((i, df.index[i], float(H.iloc[i])))
        if L.iloc[i] == L.iloc[i-w:i+w+1].min():
            lows.append((i, df.index[i], float(L.iloc[i])))
    return lows, highs

# ---- 4H pivot fetchers ----
def last_left_swing_low_high_4h(ticker, w=2, period="60d", interval="4h"):
    """
    Return last confirmed swing low and swing high from 4H timeframe (prices only).
    Uses Yahoo Finance for intraday aggregated candles.
    """
    try:
        df4 = yf.download(tickers=ticker, period=period, interval=interval,
                          auto_adjust=False, progress=False, session=SESS)
    except Exception:
        return None, None
    if df4 is None or df4.empty:
        return None, None
    df4 = df4[["Open","High","Low","Close","Volume"]].dropna()
    lows, highs = swing_points(df4, w=w)
    s_val = float(lows[-1][2]) if lows else None
    r_val = float(highs[-1][2]) if highs else None
    return s_val, r_val

# =========================
# BoS detection (Weekly + Daily fallback)
# =========================
def resample_weekly(df):
    w = pd.DataFrame({
        "Open":  df["Open"].resample("W-FRI").first(),
        "High":  df["High"].resample("W-FRI").max(),
        "Low":   df["Low"].resample("W-FRI").min(),
        "Close": df["Close"].resample("W-FRI").last(),
        "Volume":df["Volume"].resample("W-FRI").sum()
    }).dropna()
    return w

def detect_bos_daily(dfd, buffer_pct=0.0003, vol_threshold_pct=0.3, d_fractal=2):
    if dfd is None or len(dfd) < 20: return (None, None, None)
    lows_d, highs_d = swing_points(dfd, w=d_fractal)
    if not (lows_d or highs_d): return (None, None, None)
    close_last = float(dfd["Close"].iloc[-1])
    vol_last   = float(dfd["Volume"].iloc[-1])
    vol_bar    = dfd["Volume"].tail(120).quantile(vol_threshold_pct)
    if highs_d:
        i_hi, _, hi_price = highs_d[-1]
        if (close_last > hi_price * (1 + buffer_pct)) and (vol_last >= vol_bar):
            return ("up", hi_price, i_hi)   # ← fixed: i_hi (not i_i)
    if lows_d:
        i_lo, _, lo_price = lows_d[-1]
        if (close_last < lo_price * (1 - buffer_pct)) and (vol_last >= vol_bar):
            return ("down", lo_price, i_lo)
    return (None, None, None)

def detect_bos_daily(dfd, buffer_pct=0.0003, vol_threshold_pct=0.3, d_fractal=2):
    if dfd is None or len(dfd) < 20: return (None, None, None)
    lows_d, highs_d = swing_points(dfd, w=d_fractal)
    if not (lows_d or highs_d): return (None, None, None)
    close_last = float(dfd["Close"].iloc[-1])
    vol_last   = float(dfd["Volume"].iloc[-1])
    vol_bar    = dfd["Volume"].tail(120).quantile(vol_threshold_pct)
    if highs_d:
        i_hi, _, hi_price = highs_d[-1]
        if (close_last > hi_price * (1 + buffer_pct)) and (vol_last >= vol_bar):
            return ("up", hi_price, i_i)
    if lows_d:
        i_lo, _, lo_price = lows_d[-1]
        if (close_last < lo_price * (1 - buffer_pct)) and (vol_last >= vol_bar):
            return ("down", lo_price, i_lo)
    return (None, None, None)

# =========================
# Data fetch/clean
# =========================
def to_stooq_symbol(t): return f"{t.lower()}.us"

def fetch_stooq_daily(t):
    try:
        url = f"https://stooq.com/q/d/l/?s={to_stooq_symbol(t)}&i=d"
        r = SESS.get(url, timeout=10); r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if df is None or df.empty: return None
        df = df.rename(columns=str.title)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna().set_index("Date").sort_index()
        return df.iloc[-STOOQ_MAX_DAYS:]
    except Exception:
        return None

def fetch_yahoo_daily(t):
    try:
        df = yf.download(tickers=t, period=YAHOO_PERIOD, interval="1d",
                         auto_adjust=False, progress=False, session=SESS)
        if df is not None and not df.empty: return df
    except Exception:
        return None
    return None

def clean_and_summarize(df, ticker):
    if df is None or df.empty: return None
    df = df[["Open","High","Low","Close","Volume"]].dropna()
    dfc = df.iloc[-CHART_LOOKBACK:].copy()
    if dfc.empty: return None

    dfs = df.iloc[-SUMMARY_LOOKBACK:].copy()
    last = float(dfs["Close"].iloc[-1])
    chg30 = (last/float(dfs["Close"].iloc[0]) - 1.0)*100 if len(dfs)>1 else 0.0

    # ATR (daily) for zone width scaling
    atr_series = atr(dfc, n=14)
    atr_val = float(atr_series.iloc[-1]) if not atr_series.dropna().empty else max(last*0.005, 0.5)
    half = max(0.35*atr_val, 0.0015*last)

    # --- Support/Resistance from 4H pivots (last confirmed to the left) ---
    s_val, r_val = last_left_swing_low_high_4h(ticker, w=2, period="60d", interval="4h")
    sup_low = sup_high = res_low = res_high = None
    sup_label = res_label = None
    if s_val and s_val <= last:
        sup_low, sup_high = s_val - half, s_val + half
        sup_label = f"Support (4H swing low) ~{s_val:.2f}"
    if r_val and r_val >= last:
        res_low, res_high = r_val - half, r_val + half
        res_label = f"Resistance (4H swing high) ~{r_val:.2f}"

    # --- BoS (weekly first, daily fallback) ---
    bos_dir, bos_level, bos_idx = detect_bos_weekly(dfc, buffer_pct=0.0005, vol_threshold_pct=0.4, w_fractal=2)
    bos_tf = "1W"
    if bos_dir is None:
        bos_dir, bos_level, bos_idx = detect_bos_daily(dfc, buffer_pct=0.0003, vol_threshold_pct=0.3, d_fractal=2)
        if bos_dir is not None: bos_tf = "1D"

    return (dfc,last,chg30,sup_low,sup_high,res_low,res_high,
            sup_label,res_label,bos_dir,bos_level,bos_idx,bos_tf)

def fetch_one(t):
    df = fetch_stooq_daily(t)
    if df is None: df = fetch_yahoo_daily(t)
    return clean_and_summarize(df, t)

# =========================
# BOS draw helper (faint dotted, stops at last candle)
# =========================
def draw_bos_line_with_chip(d, left, x_end, top, bot, y, color, font_lbl, font_tf, base_img, tf_text="1W"):
    """
    Fainter BoS line: semi-transparent dotted line, compact chip; ends at last candle (x_end).
    """
    y = int(max(top + 4, min(bot - 4, y)))
    x0 = left + 2
    xe = max(x0 + 20, int(x_end) - 6)

    overlay = Image.new("RGBA", (base_img.width, base_img.height), (0,0,0,0))
    od = ImageDraw.Draw(overlay)

    # fine dotted line (not dashed)
    dot_len, gap = 3, 6
    xx = x0
    faint = (color[0], color[1], color[2], 160)  # ~60-65% opacity
    while xx < xe:
        x2 = min(xx + dot_len, xe)
        od.line([(xx, y), (x2, y)], fill=faint, width=2)
        xx += dot_len + gap

    base_img.alpha_composite(overlay)

    # chip
    lbl = "BOS"
    d2 = ImageDraw.Draw(base_img)
    tw, th   = d2.textbbox((0,0), lbl, font=font_lbl)[2:]
    tw2, th2 = d2.textbbox((0,0), tf_text, font=font_tf)[2:]
    padx, pady = 6, 4
    box_w = min(max(tw, tw2) + 2*padx, 86)
    box_h = th + th2 + 3 + 2*pady
    bx = int(xe - box_w - 8)
    by = max(top + 8, min(bot - box_h - 8, y - box_h//2))
    d2.rectangle((bx, by, bx + box_w, by + box_h), fill=(255,255,255), outline=(220,220,220), width=1)
    d2.text((bx + max(0,(box_w - tw)//2),  by + pady),           lbl,    fill=color, font=font_lbl)
    d2.text((bx + max(0,(box_w - tw2)//2), by + pady + th + 3),  tf_text,fill=color, font=font_tf)

# =========================
# Render one post
# =========================
def render_single_post(path, ticker, payload, brand_logo_path):
    (df,last,chg30,sup_low,sup_high,res_low,res_high,
     sup_label,res_label,bos_dir,bos_level,bos_idx,bos_tf) = payload

    img = Image.new("RGBA", (CANVAS_W, CANVAS_H), BG + (255,))
    d = ImageDraw.Draw(img)

    # Header
    d.text((MARGIN, MARGIN), ticker, fill=TEXT_MAIN, font=F_TITLE)
    d.text((MARGIN, MARGIN+72), f"{last:,.2f} USD", fill=TEXT_MAIN, font=F_PRICE)
    chg_col = UP_COL if chg30>=0 else DOWN_COL
    d.text((MARGIN, MARGIN+122), f"{chg30:+.2f}% past {SUMMARY_LOOKBACK}d", fill=chg_col, font=F_CHG)
    d.text((MARGIN, MARGIN+122+38), "Daily chart • 4H pivots for S/R", fill=TEXT_MUT, font=F_SUB)

    # Ticker logo top-right
    t_logo = find_ticker_logo_path(ticker)
    if t_logo:
        try:
            logo = Image.open(t_logo).convert("RGBA")
            logo.thumbnail((140,140))
            img.alpha_composite(logo, (CANVAS_W - logo.width - MARGIN, MARGIN))
        except Exception as e:
            print("[warn] ticker logo draw failed:", e)

    # Chart area
    top = 260; bot = CANVAS_H - 100; left = MARGIN; right = CANVAS_W - MARGIN
    d.rectangle((left, top, right, bot), fill=(255,255,255))
    for gy in (0.25, 0.5, 0.75):
        y = int(top + gy*(bot-top)); d.line([(left+4,y),(right-4,y)], fill=GRID, width=1)

    vmin = float(df["Low"].min()); vmax = float(df["High"].max())
    n = len(df); step = (right-left)/max(1,n)
    wick = max(1, int(step*0.12)); body = max(3, int(step*0.4))

    # Candles
    for i,row in enumerate(df.itertuples(index=False)):
        o,h,l,c = float(row.Open), float(row.High), float(row.Low), float(row.Close)
        cx = int(left + i*step + step*0.5)
