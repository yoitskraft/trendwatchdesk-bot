#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TrendWatchDesk - main.py (stable)
Default: generate daily charts (for daily.yml)
Optional: --posters to generate news-driven posters

Spec highlights:
- Charts: 1080x720, weekly candlesticks (1y), blue gradient bg, no grid,
  subtle translucent support zone rectangle, color company logo top-left,
  TWD white logo bottom-right, no ticker text.
- Captions (charts): emoji/sector-aware, no price, varied phrasing,
  saved to output/caption_YYYYMMDD.txt
- Posters: available via --posters (kept separate for stability)
"""

import os, sys, math, random, hashlib, datetime, traceback, re
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# -----------------------
# Paths & Constants
# -----------------------
ASSETS_DIR   = "assets"
LOGO_DIR     = os.path.join(ASSETS_DIR, "logos")
FONT_DIR     = os.path.join(ASSETS_DIR, "fonts")
BRAND_LOGO   = os.path.join(ASSETS_DIR, "brand_logo.png")
FONT_BOLD    = os.path.join(FONT_DIR, "Grift-Bold.ttf")
FONT_REG     = os.path.join(FONT_DIR, "Grift-Regular.ttf")

OUTPUT_DIR   = "output"
CHART_DIR    = os.path.join(OUTPUT_DIR, "charts")
POSTER_DIR   = os.path.join(OUTPUT_DIR, "posters")
RUN_LOG      = os.path.join(OUTPUT_DIR, "run.log")
DATESTR      = datetime.date.today().strftime("%Y%m%d")
CAPTION_TXT  = os.path.join(OUTPUT_DIR, f"caption_{DATESTR}.txt")

for d in (OUTPUT_DIR, CHART_DIR, POSTER_DIR):
    os.makedirs(d, exist_ok=True)

# Deterministic seed per day (stable picks)
SEED  = int(hashlib.sha1(DATESTR.encode()).hexdigest(), 16) % (10**8)
rng   = random.Random(SEED)

# Watch pools (includes SOFI, Robotics, Quantum)
POOLS = {
    "AI":        ["NVDA","MSFT","GOOGL","META","AMZN"],
    "MAG7":      ["AAPL","MSFT","GOOGL","META","AMZN","NVDA","TSLA"],
    "Semis":     ["NVDA","AMD","AVGO","TSM","INTC","ASML"],
    "Healthcare":["UNH","JNJ","PFE","MRK","LLY"],
    "Fintech":   ["MA","V","PYPL","SQ","SOFI"],
    "Quantum":   ["IONQ","IBM","AMZN"],
    "Robotics":  ["ISRG","FANUY","IRBT","ABB","ROK"],
    "Wildcards": ["NFLX","DIS","BABA","NIO","SHOP","PLTR"]
}

# Emoji by sector-ish mapping (fallback 📈)
SECTOR_EMOJI = {
    "AAPL":"🍏","MSFT":"🧠","NVDA":"🤖","AMD":"🔧","TSLA":"🚗","META":"📡","GOOGL":"🔎","AMZN":"📦",
    "SPY":"📊","QQQ":"📈","GLD":"🪙","SOFI":"🏦","IONQ":"🧪","IBM":"💼","ISRG":"🩺","ROK":"🏭",
    "AVGO":"📶","TSM":"🏭","INTC":"💽","ASML":"🔬","NFLX":"🎬","DIS":"🏰","BABA":"🛍️","NIO":"🔌","SHOP":"🛒","PLTR":"🛰️","MA":"💳","V":"💳","PYPL":"💸","SQ":"📱"
}

YF_NEWS_SEARCH = "https://query1.finance.yahoo.com/v1/finance/search"

# -----------------------
# Logging
# -----------------------
def log(msg: str):
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(RUN_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

# -----------------------
# Utilities (pandas-safe)
# -----------------------
def _as_float(x):
    try:
        return float(x.item())
    except AttributeError:
        return float(x)

def _series_f64(s: pd.Series) -> pd.Series:
    """Ensure float64 Series without NaNs (dropna) to avoid ambiguous truth errors."""
    if s is None or s.size == 0:
        return pd.Series([], dtype="float64")
    return pd.to_numeric(s, errors="coerce").astype("float64").dropna()

def _arr_f64(s: pd.Series) -> np.ndarray:
    return _series_f64(s).to_numpy(dtype="float64", copy=True)

def pct_change(series: pd.Series, days: int = 30) -> float:
    s = _series_f64(series)
    if len(s) < days + 1:
        return 0.0
    last = _as_float(s.iloc[-1])
    prev = _as_float(s.iloc[-days-1])
    if prev == 0:
        return 0.0
    return (last - prev) / prev * 100.0

def swing_levels(series: pd.Series, lookback: int = 14) -> Tuple[Optional[float], Optional[float]]:
    s = _series_f64(series)
    if s.empty:
        return (None, None)
    highs = s.rolling(lookback).max()
    lows  = s.rolling(lookback).min()
    lo = highs.iloc[-1] if pd.isna(lows.iloc[-1]) else lows.iloc[-1]
    hi = highs.iloc[-1]
    lo = None if pd.isna(lo) else _as_float(lo)
    hi = None if pd.isna(hi) else _as_float(hi)
    return (lo, hi)

# -----------------------
# Assets (fonts & logos)
# -----------------------
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

def load_twd_white(target_w: int) -> Optional[Image.Image]:
    if not os.path.exists(BRAND_LOGO): return None
    try:
        im = Image.open(BRAND_LOGO).convert("RGBA")
        w,h = im.size
        r = target_w / max(1, w)
        return im.resize((int(w*r), int(h*r)), Image.Resampling.LANCZOS)
    except Exception:
        return None

# -----------------------
# Visual helpers
# -----------------------
def blue_gradient_bg(W: int, H: int) -> Image.Image:
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

    # light beams
    beams = Image.new("RGBA", (W,H), (0,0,0,0))
    d = ImageDraw.Draw(beams)
    for i, a in enumerate([70,55,40]):
        off = i*140
        d.polygon([(0, 120+off), (W, 0+off), (W, 110+off), (0, 230+off)], fill=(255,255,255,a))
    beams = beams.filter(ImageFilter.GaussianBlur(45))
    return Image.alpha_composite(bg, beams)

def draw_support_zone(d: ImageDraw.ImageDraw, x1,y1,x2,y2, opacity=40, outline=110):
    # translucent rectangle (zone), barely visible
    d.rectangle([x1,y1,x2,y2], fill=(240,248,255,opacity), outline=(240,248,255,outline), width=2)

# -----------------------
# Ticker selection
# -----------------------
def pick_tickers(n:int=6) -> List[str]:
    picks = []
    def take(pool, k):
        cands = [t for t in POOLS[pool] if t not in picks]
        rng.shuffle(cands)
        picks.extend(cands[:k])
    take("AI", 2)
    take("MAG7", 2)
    take("Semis", 1)
    others = [t for k in POOLS.keys() if k not in ("AI","MAG7","Semis") for t in POOLS[k]]
    rng.shuffle(others)
    for t in others:
        if len(picks)>=n: break
        if t not in picks: picks.append(t)
    return picks[:n]

# -----------------------
# Chart rendering (candlesticks)
# -----------------------
def render_candles(draw: ImageDraw.ImageDraw, ohlc: pd.DataFrame, box: Tuple[int,int,int,int]):
    """Draw simple candlesticks in box (x1,y1,x2,y2)."""
    x1,y1,x2,y2 = box
    data = ohlc.copy()
    for col in ["Open","High","Low","Close"]:
        data[col] = pd.to_numeric(data[col], errors="coerce").astype("float64")
    data = data.dropna()
    if data.empty:
        return

    lows  = data["Low"].to_numpy()
    highs = data["High"].to_numpy()
    closes= data["Close"].to_numpy()
    opens = data["Open"].to_numpy()

    pmin = float(np.nanmin(lows))
    pmax = float(np.nanmax(highs))
    pr   = max(1e-9, pmax-pmin)

    n = len(data)
    xs = np.linspace(x1+10, x2-10, n)
    wick_w = 2
    body_w = max(4, int((x2-x1) / max(10, n*1.8)))

    def y_from_p(p): return y2 - (float(p)-pmin)/pr*(y2-y1)

    for i in range(n):
        ox, cx, hi, lo = opens[i], closes[i], highs[i], lows[i]
        X = int(xs[i])
        y_hi = int(y_from_p(hi))
        y_lo = int(y_from_p(lo))
        draw.line([(X, y_hi), (X, y_lo)], fill=(255,255,255,180), width=wick_w)

        up = cx >= ox
        top = y_from_p(max(ox,cx))
        bot = y_from_p(min(ox,cx))
        # green up candle, red down candle (slightly desaturated for the blue bg)
        color = (92, 220, 130, 255) if up else (235, 95, 95, 255)
        draw.rectangle([X - body_w, int(top), X + body_w, int(bot)], fill=color, outline=None)

def generate_chart(ticker: str) -> Optional[str]:
    try:
        df = yf.download(ticker, period="1y", interval="1wk", progress=False, auto_adjust=False, threads=False)
        if df.empty:
            log(f"[warn] no data for {ticker}")
            return None

        # Prepare series for metrics & support zone
        close_s = _series_f64(df["Close"])
        if close_s.empty:
            log(f"[warn] empty close for {ticker}")
            return None

        sup_lo, sup_hi = swing_levels(close_s, lookback=10)

        # Canvas
        W,H = 1080, 720
        img = blue_gradient_bg(W,H)
        d   = ImageDraw.Draw(img)

        # Plot area
        margin = 40
        top_pad = 30
        bot_pad = 60
        x1,y1 = margin, top_pad + 40
        x2,y2 = W - margin, H - bot_pad

        # Support zone rectangle (subtle)
        if sup_lo is not None and sup_hi is not None and sup_hi >= sup_lo:
            # Recompute mapping bounds from actual High/Low to place zone correctly
            low = float(df["Low"].min())
            high= float(df["High"].max())
            pr  = max(1e-9, high-low)
            def y_from_p(p): return y2 - (float(p)-low)/pr*(y2-y1)
            y_top = int(y_from_p(sup_hi))
            y_bot = int(y_from_p(sup_lo))
            draw_support_zone(d, x1+6, min(y_top,y_bot), x2-6, max(y_top,y_bot), opacity=38, outline=90)

        # Candles
        render_candles(d, df[["Open","High","Low","Close"]], (x1, y1, x2, y2))

        # Logos
        lg = load_logo_color(ticker, 160)
        if lg is not None:
            img.alpha_composite(lg, (x1, 12))  # top-left
        twd = load_twd_white(160)
        if twd is not None:
            img.alpha_composite(twd, (W - twd.width - 24, H - twd.height - 18))  # bottom-right

        # Save
        out = os.path.join(CHART_DIR, f"{ticker}_chart.png")
        img.convert("RGB").save(out, "PNG")
        return out
    except Exception as e:
        log(f"[error] generate_chart({ticker}): {e}")
        return None

# -----------------------
# Captions (charts)
# -----------------------
CAP_TEMPLATES = [
    "{e} {t} — momentum building · buyers watching support · eyes on flow",
    "{e} {t} — strong 30d trend · range compressing · catalyst watch",
    "{e} {t} — pullback on radar · support nearby · patience pays",
    "{e} {t} — steady grind · dip buys active · trend intact",
    "{e} {t} — volatility cooling · setup forming · breakout level in view",
    "{e} {t} — sector read-throughs matter · rotation in play",
    "{e} {t} — watch volume footprints · bids show up near demand",
]
def chart_caption_line(ticker: str, chg30: float) -> str:
    em = SECTOR_EMOJI.get(ticker, "📈")
    tmpl = rng.choice(CAP_TEMPLATES)
    # never mention price; percent can appear implicitly via "strong 30d trend"
    # to add explicit % occasionally without price:
    maybe_pct = rng.choice([True, False, False])
    if maybe_pct:
        tmpl = tmpl.replace("30d", f"{abs(chg30):.0f}d")  # keep phrasing ambiguous
    return tmpl.format(e=em, t=ticker)

# -----------------------
# Posters (opt-in)
# -----------------------
def fetch_yahoo_headlines(tickers: List[str], max_items=40) -> List[Dict]:
    items = []
    sess = requests.Session()
    sess.headers.update({"User-Agent":"TrendWatchDesk/1.0","Accept":"application/json"})
    for t in tickers:
        try:
            r = sess.get(YF_NEWS_SEARCH, params={"q": t, "quotesCount": 0, "newsCount": 10}, timeout=10)
            if r.status_code != 200: continue
            data = r.json()
            for n in data.get("news", [])[:10]:
                title = n.get("title") or ""
                link  = n.get("link") or ""
                if not title or not link: continue
                items.append({"ticker": t, "title": title, "url": link})
        except Exception as e:
            log(f"[warn] yahoo fetch failed for {t}: {e}")
    # de-dupe by normalized title
    seen, uniq = set(), []
    for it in items:
        k = re.sub(r"[^a-z0-9 ]+","", it["title"].lower()).strip()
        if k in seen: continue
        seen.add(k); uniq.append(it)
    return uniq[:max_items]

def poster_bg(W=1080,H=1080) -> Image.Image:
    return blue_gradient_bg(W,H)

def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_w: int) -> str:
    words = text.split()
    line, out = "", []
    for w in words:
        cand = (line + " " + w).strip()
        if draw.textbbox((0,0), cand, font=font)[2] > max_w and line:
            out.append(line)
            line = w
        else:
            line = cand
    if line: out.append(line)
    return "\n".join(out)

def generate_poster(ticker: str, headline: str, subtext: str) -> Optional[str]:
    try:
        W,H = 1080,1080
        img = poster_bg(W,H)
        d   = ImageDraw.Draw(img)

        # NEWS tag
        tag_font = font_bold(44)
        tag_pad  = 12
        tag_txt  = "NEWS"
        tw,th = d.textbbox((0,0), tag_txt, font=tag_font)[2:]
        d.rounded_rectangle([40,40, 40+tw+2*tag_pad, 40+th+2*tag_pad], 14, fill=(0,36,73,210))
        d.text((40+tag_pad, 40+tag_pad), tag_txt, fill=(255,255,255,255), font=tag_font)

        # Headline
        head_font = font_bold(100)
        head = wrap_text(d, headline.upper(), head_font, W-80)
        d.multiline_text((40, 150), head, font=head_font, fill="white", spacing=10, align="left")

        # Subtext (sector/price-action context)
        sub_font = font_reg(46)
        sub_wrapped = wrap_text(d, subtext, sub_font, W-80)
        d.multiline_text((40, 420), sub_wrapped, font=sub_font, fill=(235,243,255,255), spacing=10, align="left")

        # Logos (color company top-right, TWD bottom-right)
        lg = load_logo_color(ticker, 220)
        if lg is not None:
            img.alpha_composite(lg, (W - lg.width - 40, 40))
        twd = load_twd_white(220)
        if twd is not None:
            img.alpha_composite(twd, (W - twd.width - 40, H - twd.height - 40))

        out = os.path.join(POSTER_DIR, f"{ticker}_poster_{DATESTR}.png")
        img.convert("RGB").save(out, "PNG")
        return out
    except Exception as e:
        log(f"[error] generate_poster({ticker}): {e}")
        return None

# -----------------------
# Workflows
# -----------------------
def run_daily_charts():
    tickers = pick_tickers(6)
    log(f"[info] selected tickers: {tickers}")
    generated = []
    cap_lines = []

    for t in tickers:
        out = generate_chart(t)
        if out:
            generated.append(out)
            try:
                df = yf.download(t, period="6mo", interval="1d", progress=False, auto_adjust=False, threads=False)
                chg30 = pct_change(df["Close"], 30)
                cap_lines.append(chart_caption_line(t, chg30))
            except Exception:
                pass
        else:
            log(f"[warn] chart failed for {t}")

    if cap_lines:
        try:
            with open(CAPTION_TXT, "w", encoding="utf-8") as f:
                f.write("\n".join(cap_lines))
            log(f"[info] caption file written: {CAPTION_TXT}")
        except Exception as e:
            log(f"[warn] failed to write caption file: {e}")

    print("\n==============================")
    if generated:
        print("✅ Daily charts generated:")
        for p in generated:
            print(" -", p)
    else:
        print("❌ No charts generated")
    print("==============================\n")

def run_posters():
    # Build a broad watchlist from pools
    wl = []
    for v in POOLS.values():
        wl.extend(v)
    wl = sorted(set(wl))
    news = fetch_yahoo_headlines(wl, max_items=40)
    if not news:
        print("\n⚠️ No news found → posters skipped\n")
        log("[info] no news → posters skipped")
        return
    # Take up to 2 most recent/unique
    picks = news[:2]
    made = 0
    for it in picks:
        t = it["ticker"]
        title = it["title"]
        # Compose subtext tied to price action (generic but relevant)
        sub = (f"{t} in focus as traders weigh sector read-throughs and near-term momentum. "
               "Watch guidance tone, margin commentary, and follow-through on volume.")
        out = generate_poster(t, title, sub)
        if out:
            made += 1
            # save a small caption next to poster
            em = SECTOR_EMOJI.get(t, "📈")
            cap = f"{em} {t} — {title}\nSector read-throughs; watch guidance & margins next."
            with open(os.path.splitext(out)[0] + "_caption.txt", "w", encoding="utf-8") as f:
                f.write(cap)
            log(f"[info] poster saved: {out}")

    print("\n==============================")
    print(f"Posters generated: {made}")
    print("==============================\n")

# -----------------------
# CLI
# -----------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--posters", action="store_true", help="Generate news-driven posters (optional)")
    ap.add_argument("--once", type=str, help="Generate a single ticker chart")
    args = ap.parse_args()

    try:
        if args.posters:
            log("[info] running posters")
            run_posters()
        elif args.once:
            t = args.once.upper()
            log(f"[info] quick chart for {t}")
            out = generate_chart(t)
            if out:
                print("\n✅ Chart saved:", out, "\n")
            else:
                print("\n❌ Chart failed (see run.log)\n")
        else:
            log("[info] default mode → daily charts")
            run_daily_charts()
    except Exception as e:
        log(f"[fatal] {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
