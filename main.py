#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TrendWatchDesk - main.py (stable, scale-aware)
- One scale knob for everything (charts + posters)
- Blue gradient charts, NO grid, candlesticks (no overlap)
- Feathered support zone (drawn OVER candles, min band height)
- Logos: charts=white mono top-left, TWD bottom-right; posters=color on right
- Captions: variety lines with single CTA at end
- Posters: Yahoo Finance news polling (lightweight)
- CI flags: --ci (daily charts), --ci-posters (posters), --posters, --once TKR
"""

import os, re, random, hashlib, traceback, datetime
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# =========================
# ---- Paths & Outputs ----
# =========================
ASSETS_DIR   = "assets"
LOGO_DIR     = os.path.join(ASSETS_DIR, "logos")
FONT_DIR     = os.path.join(ASSETS_DIR, "fonts")
BRAND_LOGO   = os.path.join(ASSETS_DIR, "brand_logo.png")
FONT_BOLD    = os.path.join(FONT_DIR, "Grift-Bold.ttf")
FONT_REG     = os.path.join(FONT_DIR, "Grift-Regular.ttf")

OUTPUT_DIR   = "output"
CHART_DIR    = os.path.join(OUTPUT_DIR, "charts")
POSTER_DIR   = os.path.join(OUTPUT_DIR, "posters")
DATESTR      = datetime.date.today().strftime("%Y%m%d")
CAPTION_TXT  = os.path.join(OUTPUT_DIR, f"caption_{DATESTR}.txt")
RUN_LOG      = os.path.join(OUTPUT_DIR, "run.log")

for d in (OUTPUT_DIR, CHART_DIR, POSTER_DIR):
    os.makedirs(d, exist_ok=True)

# =========================
# ---- Logging ------------
# =========================
def log(msg: str):
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(RUN_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

# =========================
# ---- Deterministic RNG --
# =========================
SEED = int(hashlib.sha1(DATESTR.encode()).hexdigest(), 16) % (10**8)
rng  = random.Random(SEED)

# =========================
# ---- Pools & Sectors ----
# =========================
POOLS = {
    "AI":        ["NVDA","MSFT","GOOGL","META","AMZN"],
    "MAG7":      ["AAPL","MSFT","GOOGL","META","AMZN","NVDA","TSLA"],
    "Semis":     ["NVDA","AMD","AVGO","TSM","INTC","ASML"],
    "Fintech":   ["MA","V","PYPL","SQ","SOFI"],
    "Quantum":   ["IONQ","IBM","AMZN"],
    "Robotics":  ["ISRG","FANUY","IRBT","ABB","ROK"],
    "Wildcards": ["NFLX","DIS","BABA","NIO","SHOP","PLTR"],
}
SECTOR_EMOJI = {
    "AAPL":"🍏","MSFT":"🧠","NVDA":"🤖","AMD":"🔧","TSLA":"🚗","META":"📡","GOOGL":"🔎","AMZN":"📦",
    "SHOP":"🛒","NIO":"🔌","DIS":"🎬","NFLX":"🎬","BABA":"🛍️","SOFI":"🏦","IONQ":"🧪","IBM":"💼",
    "AVGO":"📶","TSM":"🏭","INTC":"💽","ASML":"🔬","PLTR":"🛰️","MA":"💳","V":"💳","PYPL":"💸","SQ":"📱"
}
POOL_LABEL = {"AI":"AI megacaps","MAG7":"Mega-cap tech","Semis":"semiconductors",
              "Fintech":"fintech","Quantum":"quantum computing","Robotics":"robotics",
              "Wildcards":"wildcards"}
SECTOR_OF: Dict[str,str] = {}
for k, arr in POOLS.items():
    for t in arr:
        SECTOR_OF.setdefault(t, POOL_LABEL.get(k,k))

# =========================
# ---- External Controls --
# =========================
# Default knobs (controls.py may override them safely)
CHART_SCALE = 1.0
POSTER_SCALE = None  # if None → uses CHART_SCALE

CHART_LOGO_SCALE  = 1.0
POSTER_LOGO_SCALE = 1.0

SUPPORT_FILL_ALPHA    = 96
SUPPORT_BLUR_RADIUS   = 6
SUPPORT_OUTLINE_ALPHA = 140
SUPPORT_MIN_PX        = 26

CANDLE_BODY_RATIO = 0.35
CANDLE_BODY_MAX   = 12
CANDLE_WICK_RATIO = 0.15
CANDLE_UP_RGBA    = (90,230,150,255)
CANDLE_DN_RGBA    = (245,110,110,255)
WICK_RGBA         = (245,250,255,170)

POSTER_COUNT    = 2
POSTERS_ENABLED = True
CAPTION_DECIMALS_30D = 1

try:
    import controls  # optional file you edit
    controls.apply_overrides(globals())
except Exception as e:
    print(f"[warn] controls.py not loaded: {e}")

# =========================
# ---- Global Scaling -----
# =========================
CHART_BASE_W,  CHART_BASE_H  = 1080, 720
POSTER_BASE_W, POSTER_BASE_H = 1080, 1080

if POSTER_SCALE is None:
    POSTER_SCALE = CHART_SCALE

CHART_W  = int(CHART_BASE_W  * float(CHART_SCALE))
CHART_H  = int(CHART_BASE_H  * float(CHART_SCALE))
POSTER_W = int(POSTER_BASE_W * float(POSTER_SCALE))
POSTER_H = int(POSTER_BASE_H * float(POSTER_SCALE))

# =========================
# ---- Fonts / Logos ------
# =========================
def _font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

def font_bold(sz:int): return _font(FONT_BOLD, sz)
def font_reg(sz:int):  return _font(FONT_REG, sz)

def load_logo_color(ticker: str, target_w: int) -> Optional[Image.Image]:
    path = os.path.join(LOGO_DIR, f"{ticker}.png")
    if not os.path.exists(path): return None
    try:
        im = Image.open(path).convert("RGBA")
        w,h = im.size
        r = target_w / max(1, w)
        return im.resize((int(w*r), int(h*r)), Image.Resampling.LANCZOS)
    except Exception:
        return None

# Back-compat alias (some poster code historically called load_logo)
def load_logo(ticker: str, target_w: int) -> Optional[Image.Image]:
    return load_logo_color(ticker, target_w)

def to_white_mono(logo_rgba: Image.Image, alpha: int = 255) -> Image.Image:
    if logo_rgba.mode != "RGBA":
        logo_rgba = logo_rgba.convert("RGBA")
    _, _, _, a = logo_rgba.split()
    white = Image.new("RGBA", logo_rgba.size, (255,255,255,alpha))
    white.putalpha(a)
    return white

def load_twd_white(target_w: int) -> Optional[Image.Image]:
    if not os.path.exists(BRAND_LOGO): return None
    try:
        im = Image.open(BRAND_LOGO).convert("RGBA")
        w,h = im.size
        r = target_w / max(1, w)
        im = im.resize((int(w*r), int(h*r)), Image.Resampling.LANCZOS)
        return to_white_mono(im, alpha=255)
    except Exception:
        return None

# =========================
# ---- Pandas-safe utils --
# =========================
def _series_f64(s):
    return pd.to_numeric(s, errors="coerce").astype("float64").dropna()

def extract_close(df: pd.DataFrame, ticker: Optional[str] = None) -> pd.Series:
    if df is None or df.empty: return pd.Series([], dtype="float64")
    if "Close" in df.columns and not isinstance(df["Close"], pd.DataFrame):
        return _series_f64(df["Close"])
    if isinstance(df.columns, pd.MultiIndex):
        if ticker and ("Close", ticker) in df.columns:
            return _series_f64(df[("Close", ticker)])
        if "Close" in df.columns:
            sub = df["Close"]
            if isinstance(sub, pd.Series): return _series_f64(sub)
            if isinstance(sub, pd.DataFrame) and not sub.empty: return _series_f64(sub.iloc[:,0])
    return _series_f64(df.iloc[:,0])

def extract_ohlc(df: pd.DataFrame, ticker: Optional[str] = None) -> pd.DataFrame:
    out = {}
    for name in ["Open","High","Low","Close"]:
        ser = None
        if name in df.columns and not isinstance(df[name], pd.DataFrame):
            ser = df[name]
        elif isinstance(df.columns, pd.MultiIndex):
            if ticker and (name, ticker) in df.columns:
                ser = df[(name, ticker)]
            elif name in df.columns:
                sub = df[name]
                if isinstance(sub, pd.Series): ser = sub
                elif isinstance(sub, pd.DataFrame) and not sub.empty: ser = sub.iloc[:,0]
        if ser is not None:
            out[name] = _series_f64(ser)
    return pd.DataFrame(out).dropna()

def pct_change(series: pd.Series, days: int = 30) -> float:
    s = _series_f64(series)
    if len(s) < days + 1: return 0.0
    prev, last = float(s.iloc[-days-1]), float(s.iloc[-1])
    if prev == 0: return 0.0
    return (last - prev) / prev * 100.0

def swing_levels(series: pd.Series, lookback: int = 10) -> Tuple[Optional[float], Optional[float]]:
    s = _series_f64(series)
    if s.empty: return (None, None)
    return (float(s.rolling(lookback).min().iloc[-1]), float(s.rolling(lookback).max().iloc[-1]))

def is_breakout(series: pd.Series, lookback: int = 20) -> bool:
    s = _series_f64(series)
    if len(s) < lookback + 1: return False
    return float(s.iloc[-1]) >= float(s.iloc[-(lookback+1):].max())

def is_breakdown(series: pd.Series, lookback: int = 20) -> bool:
    s = _series_f64(series)
    if len(s) < lookback + 1: return False
    return float(s.iloc[-1]) <= float(s.iloc[-(lookback+1):].min())

# =========================
# ---- Visual helpers -----
# =========================
def blue_gradient_bg(W: int, H: int) -> Image.Image:
    """Poster-style blue gradient with soft beams (no grid)."""
    base = Image.new("RGB", (W,H), (10,58,102))
    grad = Image.new("RGB", (W,H))
    for y in range(H):
        t = y / max(1, H-1)
        r = int(10 + (22-10)*t)
        g = int(58 + (130-58)*t)
        b = int(102 + (220-102)*t)
        for x in range(W):
            grad.putpixel((x,y), (r,g,b))
    bg = Image.blend(base, grad, 0.92).convert("RGBA")

    beams = Image.new("RGBA", (W,H), (0,0,0,0))
    d = ImageDraw.Draw(beams)
    for i, a in enumerate([60,45,30]):
        off = i*int(140*CHART_SCALE)
        d.polygon([(0, int(120*CHART_SCALE)+off), (W, off),
                   (W, int(110*CHART_SCALE)+off), (0, int(230*CHART_SCALE)+off)],
                  fill=(255,255,255,a))
    beams = beams.filter(ImageFilter.GaussianBlur(int(45*CHART_SCALE)))
    return Image.alpha_composite(bg, beams)

def feathered_support(img: Image.Image, x1:int, y1:int, x2:int, y2:int,
                      fill_alpha: int = None, blur_radius: int = None, outline_alpha: int = None):
    """Feathered white zone OVER candles."""
    fa = SUPPORT_FILL_ALPHA if fill_alpha is None else fill_alpha
    br = SUPPORT_BLUR_RADIUS if blur_radius is None else blur_radius
    oa = SUPPORT_OUTLINE_ALPHA if outline_alpha is None else outline_alpha

    W, H = img.size
    base = Image.new("RGBA", (W, H), (0,0,0,0))
    bd = ImageDraw.Draw(base)
    bd.rectangle([x1, y1, x2, y2], fill=(255, 255, 255, int(fa)))
    base = base.filter(ImageFilter.GaussianBlur(int(br)))
    img.alpha_composite(base)

    top = Image.new("RGBA", (W, H), (0,0,0,0))
    td = ImageDraw.Draw(top)
    td.rectangle([x1, y1, x2, y2], outline=(255, 255, 255, int(oa)), width=2)
    img.alpha_composite(top)

# =========================
# ---- Candles renderer ---
# =========================
def render_candles(draw: ImageDraw.ImageDraw, ohlc: pd.DataFrame, box: Tuple[int,int,int,int]):
    x1,y1,x2,y2 = box
    data = ohlc[["Open","High","Low","Close"]].dropna()
    if data.empty: return

    n = len(data)
    lows, highs = data["Low"].to_numpy(), data["High"].to_numpy()
    opens, closes = data["Open"].to_numpy(), data["Close"].to_numpy()
    pmin, pmax = float(np.nanmin(lows)), float(np.nanmax(highs))
    pr = max(1e-9, pmax - pmin)

    # X positions across plot area
    pad_inner = int(12 * CHART_SCALE)
    xs = np.linspace(x1 + pad_inner, x2 - pad_inner, n)

    # Compute spacing and ensure a visible gap so bodies never overlap
    if n > 1:
        dx = (xs[-1] - xs[0]) / (n - 1)
    else:
        dx = (x2 - x1)
    gap = max(1.0, dx * 0.18)                       # ~18% gap between candles
    body_full = max(1, int(min(dx - gap, CANDLE_BODY_MAX * CHART_SCALE)))
    body_half = max(1, body_full // 2)
    wick_w = max(1, int(dx * max(0.06, CANDLE_WICK_RATIO)))  # thin wicks

    def y(p): return y2 - (float(p)-pmin)/pr*(y2-y1)

    for i in range(n):
        ox, cx, hi, lo = opens[i], closes[i], highs[i], lows[i]
        X = int(xs[i])

        # wick
        draw.line([(X,int(y(lo))),(X,int(y(hi)))], fill=WICK_RGBA, width=wick_w)

        # body
        up = cx >= ox
        top, bot = y(max(ox,cx)), y(min(ox,cx))
        color = CANDLE_UP_RGBA if up else CANDLE_DN_RGBA
        draw.rectangle([X-body_half, int(top), X+body_half, int(bot)], fill=color, outline=None)

# =========================
# ---- Chart generator ----
# =========================

def pivots(series: pd.Series, kind: str = "high", window: int = 3) -> List[Tuple[int,float]]:
    """Return list of (index, price) for local extrema over the whole series."""
    s = _series_f64(series); n = len(s)
    out = []
    if n == 0: return out
    for i in range(window, n - window):
        seg = s.iloc[i - window:i + window + 1]
        val = float(s.iloc[i])
        if kind == "high":
            if np.isclose(val, float(seg.max())):
                out.append((i, val))
        else:
            if np.isclose(val, float(seg.min())):
                out.append((i, val))
    return out

def cluster_levels(points: List[Tuple[int,float]], atr: float,
                   bin_atr: float = 0.5, min_touches: int = 3,
                   decay: float = 0.995, total_len: Optional[int] = None) -> List[Dict]:
    """
    Histogram-like clustering of pivot prices with ATR-sized bins.
    Returns list of clusters: [{"price": level, "score": score, "count": k}]
    """
    if not points or atr <= 0: return []
    bin_size = max(1e-9, atr * bin_atr)
    buckets: Dict[int, Dict[str, float]] = {}
    N = total_len if total_len is not None else (max(i for i,_ in points) + 1)

    for idx, price in points:
        key = int(round(price / bin_size))
        age = max(0, (N - 1) - idx)          # bars since pivot
        w   = (decay ** age)                 # recency weight
        b   = buckets.get(key)
        if b is None:
            buckets[key] = {"w": w, "sum_pw": price * w, "count": 1}
        else:
            b["w"] += w; b["sum_pw"] += price * w; b["count"] += 1

    clusters = []
    for key, b in buckets.items():
        if b["count"] >= min_touches:
            level = b["sum_pw"] / max(1e-9, b["w"])
            clusters.append({"price": float(level), "score": float(b["w"]), "count": int(b["count"])})
    # strongest first by (touches, score)
    clusters.sort(key=lambda c: (c["count"], c["score"]), reverse=True)
    return clusters

def choose_revisit_level(bias: str, last: float, clusters: List[Dict], atr: float,
                         max_dist_atr: float = 2.5) -> Optional[float]:
    """Pick nearest strong level consistent with bias (above for bullish highs, below for bearish lows)."""
    if not clusters or atr <= 0 or last <= 0:
        return None
    # Filter by distance (in ATR)
    filt = [c for c in clusters if abs(c["price"] - last) / atr <= max_dist_atr]
    if not filt:
        filt = clusters[:]  # fallback: any cluster

    if bias == "bullish":
        above = [c for c in filt if c["price"] >= last]
        cand  = above if above else filt
    else:
        below = [c for c in filt if c["price"] <= last]
        cand  = below if below else filt

    # Prefer nearest; break ties by stronger cluster
    cand.sort(key=lambda c: (abs(c["price"] - last), -c["count"], -c["score"]))
    return float(cand[0]["price"]) if cand else None

def atr14(df: pd.DataFrame) -> float:
    o = df["Open"].to_numpy()
    h = df["High"].to_numpy()
    l = df["Low"].to_numpy()
    c = df["Close"].to_numpy()
    tr = []
    for i in range(len(c)):
        if i == 0:
            tr.append(h[i] - l[i])
        else:
            tr.append(max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])))
    if len(tr) < 14:
        return max(1e-6, float(np.nanmean(tr)))
    return max(1e-6, float(pd.Series(tr).rolling(14).mean().iloc[-1]))

def find_last_swing(series: pd.Series, kind: str = "high",
                    window: int = 3, lookback: int = 250) -> Optional[float]:
    """
    kind='high' → last local maximum; kind='low' → last local minimum.
    Scans backward over up to `lookback` bars, ignores very last bar for stability.
    Returns the swing price or None.
    """
    s = _series_f64(series)
    n = len(s)
    if n == 0: return None
    start = max(0, n - lookback - 1)
    end   = n - 2  # ignore current bar
    for i in range(end, start - 1, -1):
        lo = max(0, i - window)
        hi = min(n - 1, i + window)
        seg = s.iloc[lo:hi+1]
        val = float(s.iloc[i])
        if kind == "high":
            if np.isclose(val, float(seg.max())):
                return val
        else:
            if np.isclose(val, float(seg.min())):
                return val
    return None

def trend_bias(close: pd.Series) -> str:
    """
    Simple bias: bullish if above 20DMA AND 5-bar momentum >= 0, else bearish.
    """
    s = _series_f64(close)
    if len(s) < 25:
        return "bullish" if len(s) >= 2 and s.iloc[-1] >= s.iloc[-2] else "bearish"
    ma20 = float(pd.Series(s).rolling(20).mean().iloc[-1])
    mom5 = float(s.iloc[-1] - s.iloc[-6]) if len(s) >= 6 else 0.0
    return "bullish" if (float(s.iloc[-1]) >= ma20 and mom5 >= 0) else "bearish"

def generate_chart(ticker: str) -> Optional[str]:
    """Daily chart (1y): gradient, non-overlapping candles, tight swing-level zone, white logos."""
    try:
        # --- Data: DAILY for 1 year ---
        df = yf.download(ticker, period="1y", interval="1d",
                         progress=False, auto_adjust=False, threads=False)
        if df is None or df.empty:
            log(f"[warn] no data for {ticker}")
            return None

        ohlc = extract_ohlc(df, ticker)
        if ohlc.empty or any(c not in ohlc.columns for c in ("Open","High","Low","Close")):
            log(f"[warn] no ohlc for {ticker}")
            return None

        # --- Canvas ---
        W, H = CHART_W, CHART_H
        img  = blue_gradient_bg(W, H)
        d    = ImageDraw.Draw(img)

        # --- Plot area (scaled inner box) ---
        outer   = int(CHART_MARGIN * CHART_SCALE)
        top_off = int(PLOT_TOP_OFFSET * CHART_SCALE)
        avail_w = W - 2 * outer
        avail_h = H - 2 * outer
        plot_w  = int(avail_w * float(PLOT_SCALE))
        plot_h  = int(avail_h * float(PLOT_SCALE))
        pad_x   = (avail_w - plot_w) // 2
        pad_y   = (avail_h - plot_h) // 2
        x1 = outer + pad_x
        y1 = outer + top_off + pad_y
        x2 = x1 + plot_w
        y2 = y1 + plot_h

        # --- Candles first ---
        render_candles(d, ohlc, (x1, y1, x2, y2))

                # --- Revisit Zone from clustered swings over 1y (precise + realistic) ---
        close_s = ohlc["Close"]; high_s = ohlc["High"]; low_s = ohlc["Low"]
        last = float(close_s.iloc[-1])
        bias = trend_bias(close_s)                 # 'bullish' or 'bearish'
        atr  = atr14(ohlc)

        # Controls (fallbacks if not in controls.py)
        BIN_ATR   = float(globals().get("LEVEL_BIN_ATR",   0.5))   # cluster bin size (in ATRs)
        MIN_TOUCH = int(globals().get("LEVEL_MIN_TOUCHES", 3))     # min pivots per cluster
        DECAY     = float(globals().get("LEVEL_DECAY",     0.995)) # recency weight
        MAX_DATR  = float(globals().get("LEVEL_MAX_DIST_ATR", 2.5))# max distance from last (in ATRs)
        Z_ATR     = float(globals().get("ZONE_ATR_MULT",   0.40))  # half-height of zone (in ATRs)
        PIV_WIN   = int(globals().get("PIVOT_WINDOW", 3))          # pivot sensitivity
        inset     = int(globals().get("SUPPORT_INSET", 6) * CHART_SCALE)

        # Build pivot sets across the year
        highs = pivots(high_s, kind="high", window=PIV_WIN)
        lows  = pivots(low_s,  kind="low",  window=PIV_WIN)

        # Cluster pivots into levels using ATR-sized bins
        n_bars    = len(close_s)
        high_lvls = cluster_levels(highs, atr, BIN_ATR, MIN_TOUCH, DECAY, total_len=n_bars)
        low_lvls  = cluster_levels(lows,  atr, BIN_ATR, MIN_TOUCH, DECAY, total_len=n_bars)

        # Choose the nearest strong level consistent with bias:
        #  - bullish → last swing HIGH cluster (likely revisit above)
        #  - bearish → last swing LOW  cluster (likely revisit below)
        if bias == "bullish":
            level = choose_revisit_level("bullish", last, high_lvls, atr, MAX_DATR)
        else:
            level = choose_revisit_level("bearish", last, low_lvls,  atr, MAX_DATR)

        if (level is not None) and (atr > 0):
            half = max(1e-6, atr * Z_ATR) * 0.5      # tight ATR-based band
            lo_price = level - half
            hi_price = level + half

            # Map prices → y
            pmin = float(low_s.min()); pmax = float(high_s.max())
            pr   = max(1e-9, pmax - pmin)
            def y_from(p: float) -> int:
                return int(y2 - (float(p) - pmin) / pr * (y2 - y1))
            y_top, y_bot = y_from(hi_price), y_from(lo_price)

            # Enforce minimum visible thickness (px)
            MIN_PX = int(SUPPORT_MIN_PX)
            if (y_bot - y_top) < MIN_PX:
                mid = (y_top + y_bot) // 2
                pad = MIN_PX // 2
                y_top = max(y1 + 4, mid - pad)
                y_bot = min(y2 - 4, mid + pad)

            feathered_support(
                img,
                x1 + inset, min(y_top, y_bot),
                x2 - inset, max(y_top, y_bot),
                fill_alpha=int(SUPPORT_FILL_ALPHA),
                blur_radius=int(SUPPORT_BLUR_RADIUS),
                outline_alpha=int(SUPPORT_OUTLINE_ALPHA)
            )

        # --- Logos (charts: white mono top-left; TWD white bottom-right) ---
        lg = load_logo_color(ticker, int(170 * CHART_LOGO_SCALE * CHART_SCALE))
        if lg is not None:
            lg_white = to_white_mono(lg, alpha=255)
            img.alpha_composite(lg_white, (x1, int(16 * CHART_SCALE)))

        twd = load_twd_white(int(160 * CHART_SCALE))
        if twd is not None:
            img.alpha_composite(twd, (W - twd.width - int(18 * CHART_SCALE),
                                      H - twd.height - int(14 * CHART_SCALE)))

        # --- Save ---
        out = os.path.join(CHART_DIR, f"{ticker}_chart.png")
        img.convert("RGB").save(out, "PNG")
        return out

    except Exception as e:
        log(f"[error] generate_chart({ticker}): {e}")
        return None

# =========================
# ---- Captions -----------
# =========================
CHART_TEMPLATES = {
    "breakout": [
        "{e} {t} — up {p}% in 30d, pushing into new highs.",
        "{e} {t} — {p}% this month, breaking out of the range.",
        "{e} {t} — steady climb, +{p}% in 30d.",
    ],
    "near_support": [
        "{e} {t} — {p}% over 30d, holding firm around support.",
        "{e} {t} — still anchored at the base, +{p}% in a month.",
        "{e} {t} — buyers keeping it steady, {p}% gain in 30d.",
    ],
    "pullback": [
        "{e} {t} — {p}% in 30d, easing back after a strong run.",
        "{e} {t} — short pullback in play, still {p}% higher on the month.",
        "{e} {t} — {p}% over 30d, consolidating after recent strength.",
    ],
    "trend": [
        "{e} {t} — +{p}% in 30d, trend still pointing up.",
        "{e} {t} — higher lows, +{p}% this month.",
        "{e} {t} — {p}% gain in 30d, momentum intact.",
    ],
}
def _fmt_pct(v: float) -> str:
    d = int(CAPTION_DECIMALS_30D)
    return f"{v:.{d}f}"

def chart_caption_line_for(ticker: str, daily_close: pd.Series, weekly_close: pd.Series) -> Tuple[str,str]:
    e = SECTOR_EMOJI.get(ticker, "📈")
    p30 = _fmt_pct(pct_change(daily_close, 30))
    if is_breakout(daily_close, 20):
        mood = "breakout"
    elif pct_change(daily_close, 10) < 0:
        mood = "pullback"
    elif swing_levels(weekly_close, 10)[0] is not None:
        mood = "near_support"
    else:
        mood = "trend"
    line = random.choice(CHART_TEMPLATES[mood]).format(e=e, t=ticker, p=p30)
    return (re.sub(r"\s+"," ", line).strip(), mood)

def build_summary_cta(mood_counts: Dict[str,int], tickers: List[str]) -> str:
    mood = "trend"
    if mood_counts:
        mood = max(mood_counts.items(), key=lambda kv: kv[1])[0] or "trend"
    # sector flavor
    sectors, seen = [], set()
    for t in tickers:
        s = SECTOR_OF.get(t)
        if s and s not in seen: seen.add(s); sectors.append(s)
    if not sectors: sector_phrase = "the tape"
    elif len(sectors)==1: sector_phrase = sectors[0]
    else: sector_phrase = f"{sectors[0]} & {sectors[1]}"
    prompts = {
        "breakout": [
            f"{sector_phrase} showing breakouts — how far can momentum carry from here?",
            f"Breakout day for {sector_phrase}. Do you see follow-through next?",
        ],
        "near_support": [
            f"{sector_phrase} holding the base — does this look like a durable floor?",
            f"Bounces near support in {sector_phrase}. Would you add on dips?",
        ],
        "pullback": [
            f"Pullbacks across {sector_phrase} — healthy reset or trend change?",
            f"{sector_phrase} cooling off — are you buying this dip or waiting?",
        ],
        "trend": [
            f"{sector_phrase} still trending — is there more room to run?",
            f"Uptrends intact in {sector_phrase}. Ride it or trim here?",
        ],
    }
    import random as _r
    return _r.choice(prompts.get(mood, prompts["trend"]))

# =========================
# ---- Posters ------------
# =========================
YF_NEWS_SEARCH = "https://query1.finance.yahoo.com/v1/finance/search"

def poster_bg(W=POSTER_W, H=POSTER_H):
    # reuse blue gradient
    return blue_gradient_bg(W, H)

def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_w: int) -> str:
    words = text.split(); line=""; out=[]
    for w in words:
        cand=(line+" "+w).strip()
        if draw.textbbox((0,0), cand, font=font)[2] > max_w and line:
            out.append(line); line=w
        else:
            line=cand
    if line: out.append(line)
    return "\n".join(out)

def load_logo_color_safe(ticker: str, w: int) -> Optional[Image.Image]:
    try: return load_logo_color(ticker, w)
    except: return None

def generate_poster_image(ticker: str, headline: str, subtext: str) -> Optional[str]:
    try:
        W, H = POSTER_W, POSTER_H
        img = poster_bg(W,H)
        d   = ImageDraw.Draw(img)

        # tag
        tag_font=font_bold(int(44 * POSTER_SCALE)); pad=int(12 * POSTER_SCALE); txt="NEWS"
        tw, th = d.textbbox((0,0), txt, font=tag_font)[2:]
        d.rounded_rectangle([40,40,40+tw+2*pad,40+th+2*pad], int(14*POSTER_SCALE), fill=(0,36,73,210))
        d.text((40+pad,40+pad), txt, fill="white", font=tag_font)

        # headline
        hfont=font_bold(int(100 * POSTER_SCALE))
        head=wrap_text(d, headline.upper(), hfont, W-int(80*POSTER_SCALE))
        d.multiline_text((40,int(150*POSTER_SCALE)), head, font=hfont, fill="white", spacing=int(10*POSTER_SCALE), align="left")

        # sub
        sfont=font_reg(int(46 * POSTER_SCALE))
        subw=wrap_text(d, subtext, sfont, W-int(80*POSTER_SCALE))
        d.multiline_text((40,int(420*POSTER_SCALE)), subw, font=sfont, fill=(235,243,255,255), spacing=int(10*POSTER_SCALE), align="left")

        # logos (color on right)
        lscale = float(POSTER_LOGO_SCALE) * float(POSTER_SCALE)
        lg = load_logo_color_safe(ticker, int(220 * lscale))
        if lg is not None:
            img.alpha_composite(lg, (W - lg.width - int(40 * POSTER_SCALE), int(40 * POSTER_SCALE)))
        twd = load_twd_white(int(220 * POSTER_SCALE))
        if twd is not None:
            img.alpha_composite(twd, (W - twd.width - int(40 * POSTER_SCALE), H - twd.height - int(40 * POSTER_SCALE)))

        out = os.path.join(POSTER_DIR, f"{ticker}_poster_{DATESTR}.png")
        img.convert("RGB").save(out, "PNG")
        return out
    except Exception as e:
        log(f"[error] generate_poster_image({ticker}): {e}")
        return None

def build_poster_caption(ticker: str, headline: str, d_close: pd.Series) -> str:
    e = SECTOR_EMOJI.get(ticker, "📈")
    p5  = pct_change(d_close, 5)
    p30 = pct_change(d_close, 30)
    if is_breakout(d_close, 20): pa="breakout on the chart"
    elif is_breakdown(d_close, 20): pa="breakdown risk in view"
    elif pct_change(d_close, 10) > 0: pa="buy-the-dip flows showing up"
    else: pa="consolidation looks orderly"
    core = f"{e} {ticker} — still in focus as sentiment shifts."
    tail = f"30d: {p30:+.1f}% • 5d: {p5:+.1f}% · {pa}."
    return core+"\n"+tail

# =========================
# ---- Workflows ----------
# =========================
def pick_tickers(n:int=6) -> List[str]:
    picks: List[str] = []
    def take(pool,k):
        cand=[t for t in POOLS[pool] if t not in picks]
        rng.shuffle(cand); picks.extend(cand[:k])
    take("AI",2); take("MAG7",2); take("Semis",1)
    others=[t for k in POOLS if k not in ("AI","MAG7","Semis") for t in POOLS[k]]
    rng.shuffle(others)
    for t in others:
        if len(picks)>=n: break
        if t not in picks: picks.append(t)
    return picks[:n]

def run_daily_charts():
    tickers = pick_tickers(6)
    log(f"[info] selected tickers: {tickers}")
    generated, cap_lines = [], []
    mood_counts = {"breakout":0,"near_support":0,"pullback":0,"trend":0}

    for t in tickers:
        out = generate_chart(t)
        if out:
            generated.append(out)
            try:
                ddf = yf.download(t, period="6mo", interval="1d", progress=False, auto_adjust=False, threads=False)
                wdf = yf.download(t, period="1y",  interval="1wk", progress=False, auto_adjust=False, threads=False)
                daily_close  = extract_close(ddf, t)
                weekly_close = extract_close(wdf, t)
                line, mood = chart_caption_line_for(t, daily_close, weekly_close)
                mood_counts[mood] = mood_counts.get(mood,0)+1
                cap_lines.append(line)
            except Exception:
                pass
        else:
            log(f"[warn] chart failed for {t}")

    if cap_lines:
        cap_lines.append("")
        cap_lines.append(build_summary_cta(mood_counts, tickers))
        try:
            with open(CAPTION_TXT, "w", encoding="utf-8") as f:
                f.write("\n".join(cap_lines))
            log(f"[info] caption file written: {CAPTION_TXT}")
        except Exception as e:
            log(f"[warn] failed to write caption file: {e}")

    print("\n==============================")
    if generated:
        print("✅ Daily charts generated:")
        for p in generated: print(" -", p)
    else:
        print("❌ No charts generated")
    print("Caption file:", CAPTION_TXT if cap_lines else "(none)")
    print("==============================\n")

def run_posters():
    if not POSTERS_ENABLED:
        log("[info] posters disabled via controls")
        print("\n⚠️ Posters disabled via controls.py\n")
        return

    wl = sorted({t for arr in POOLS.values() for t in arr})
    items = []
    sess = requests.Session()
    sess.headers.update({"User-Agent":"TrendWatchDesk/1.0","Accept":"application/json"})
    for t in wl:
        try:
            r = sess.get(YF_NEWS_SEARCH, params={"q": t, "quotesCount": 0, "newsCount": 10}, timeout=10)
            if r.status_code != 200: continue
            for n in r.json().get("news", [])[:10]:
                title = n.get("title") or ""
                link  = n.get("link") or ""
                if title and link: items.append({"ticker":t, "title":title, "url":link})
        except Exception as e:
            log(f"[warn] yahoo fetch failed for {t}: {e}")

    # dedupe by normalized title
    seen, uniq = set(), []
    for it in items:
        k = re.sub(r"[^a-z0-9 ]+","", it["title"].lower()).strip()
        if k in seen: continue
        seen.add(k); uniq.append(it)

    if not uniq:
        print("\n⚠️ No news found → posters skipped\n")
        log("[info] no news → posters skipped")
        return

    made = 0
    for it in uniq[:int(POSTER_COUNT)]:
        t = it["ticker"]; title = it["title"].strip()
        sub = (f"{t} stays in focus as the story evolves across the sector. "
               "Traders are watching guidance tone, margins, and follow-through.")
        out = generate_poster_image(t, title, sub)
        if out:
            made += 1
            dd = yf.download(t, period="6mo", interval="1d", progress=False, auto_adjust=False, threads=False)
            dclose = extract_close(dd, t)
            cap = build_poster_caption(t, title, dclose)
            with open(os.path.splitext(out)[0] + "_caption.txt", "w", encoding="utf-8") as f:
                f.write(cap)
            log(f"[info] poster saved: {out}")

    print("\n==============================")
    print(f"Posters generated: {made}")
    print("==============================\n")

# =========================
# ---- CLI ----------------
# =========================
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--posters", action="store_true", help="Generate news-driven posters")
    ap.add_argument("--once", type=str, help="Generate a single ticker chart")
    ap.add_argument("--ci", action="store_true", help=argparse.SUPPRESS)         # legacy
    ap.add_argument("--ci-posters", action="store_true", help=argparse.SUPPRESS) # legacy
    args = ap.parse_args()

    try:
        if args.ci:
            log("[info] legacy --ci → daily charts")
            run_daily_charts()
        elif args.ci_posters:
            log("[info] legacy --ci-posters → posters")
            run_posters()
        elif args.posters:
            log("[info] running posters")
            run_posters()
        elif args.once:
            t = args.once.upper()
            log(f"[info] quick chart for {t}")
            out = generate_chart(t)
            if out: print("\n✅ Chart saved:", out, "\n")
            else:   print("\n❌ Chart failed (see run.log)\n")
        else:
            log("[info] default mode → daily charts")
            run_daily_charts()
    except Exception as e:
        log(f"[fatal] {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
