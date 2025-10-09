#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TrendWatchDesk - main.py (restored chart style)
- Candlestick charts, blue gradient, NO grid
- Subtle support-zone rectangle (translucent)
- No ticker text on chart
- Company logo (white mono) top-left; TWD white bottom-right
- Safe yfinance extractors (handles MultiIndex)
- Daily captions (single CTA at end), Posters optional
- CI back-compat: --ci (charts), --ci-posters (posters)
"""

import os, re, random, hashlib, traceback, datetime
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# -----------------------
# Paths & constants
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

# deterministic daily seed
SEED = int(hashlib.sha1(DATESTR.encode()).hexdigest(), 16) % (10**8)
rng  = random.Random(SEED)

# pools & sector map (same as before, trimmed for brevity but includes your adds)
POOLS = {
    "AI":        ["NVDA","MSFT","GOOGL","META","AMZN"],
    "MAG7":      ["AAPL","MSFT","GOOGL","META","AMZN","NVDA","TSLA"],
    "Semis":     ["NVDA","AMD","AVGO","TSM","INTC","ASML"],
    "Fintech":   ["MA","V","PYPL","SQ","SOFI"],
    "Quantum":   ["IONQ","IBM","AMZN"],
    "Robotics":  ["ISRG","FANUY","IRBT","ABB","ROK"],
    "Wildcards": ["NFLX","DIS","BABA","NIO","SHOP","PLTR"]
}
SECTOR_EMOJI = {
    "AAPL":"🍏","MSFT":"🧠","NVDA":"🤖","AMD":"🔧","TSLA":"🚗","META":"📡","GOOGL":"🔎","AMZN":"📦",
    "SHOP":"🛒","NIO":"🔌","DIS":"🎬","NFLX":"🎬","BABA":"🛍️","SOFI":"🏦","IONQ":"🧪","IBM":"💼",
    "AVGO":"📶","TSM":"🏭","INTC":"💽","ASML":"🔬","PLTR":"🛰️","MA":"💳","V":"💳","PYPL":"💸","SQ":"📱"
}
POOL_LABEL = {
    "AI": "AI megacaps","MAG7":"Mega-cap tech","Semis":"semiconductors","Fintech":"fintech",
    "Quantum":"quantum computing","Robotics":"robotics","Wildcards":"wildcards",
}
SECTOR_OF: Dict[str,str] = {}
for k, arr in POOLS.items():
    for t in arr:
        SECTOR_OF.setdefault(t, POOL_LABEL.get(k,k))

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
# Pandas-safe helpers
# -----------------------
def _series_f64(s):
    return pd.to_numeric(s, errors="coerce").astype("float64").dropna()

def extract_close(df: pd.DataFrame, ticker: Optional[str] = None) -> pd.Series:
    if df is None or df.empty:
        return pd.Series([], dtype="float64")
    if "Close" in df.columns and not isinstance(df["Close"], pd.DataFrame):
        return _series_f64(df["Close"])
    if isinstance(df.columns, pd.MultiIndex):
        if ticker and ("Close", ticker) in df.columns:
            return _series_f64(df[("Close", ticker)])
        if "Close" in df.columns:
            sub = df["Close"]
            if isinstance(sub, pd.Series): return _series_f64(sub)
            if isinstance(sub, pd.DataFrame) and not sub.empty: return _series_f64(sub.iloc[:,0])
    # fallback
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

def is_breakout(series: pd.Series, lookback: int = 20) -> bool:
    s = _series_f64(series)
    if len(s) < lookback + 1: return False
    return float(s.iloc[-1]) >= float(s.iloc[-(lookback+1):].max())

def is_breakdown(series: pd.Series, lookback: int = 20) -> bool:
    s = _series_f64(series)
    if len(s) < lookback + 1: return False
    return float(s.iloc[-1]) <= float(s.iloc[-(lookback+1):].min())

def swing_levels(series: pd.Series, lookback: int = 10) -> Tuple[Optional[float], Optional[float]]:
    s = _series_f64(series)
    if s.empty: return (None, None)
    return (float(s.rolling(lookback).min().iloc[-1]), float(s.rolling(lookback).max().iloc[-1]))

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

def to_white_mono(logo_rgba: Image.Image, alpha: int = 255) -> Image.Image:
    """Convert a color logo to solid white keeping alpha edges clean."""
    # Use the existing alpha as mask
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
        return im.resize((int(w*r), int(h*r)), Image.Resampling.LANCZOS)
    except Exception:
        return None

# -----------------------
# Visual helpers
# -----------------------
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
        off = i*140
        d.polygon([(0, 120+off), (W, 0+off), (W, 110+off), (0, 230+off)], fill=(255,255,255,a))
    beams = beams.filter(ImageFilter.GaussianBlur(45))
    return Image.alpha_composite(bg, beams)

def draw_support_zone(d: ImageDraw.ImageDraw, x1,y1,x2,y2, opacity=28, outline=72):
    d.rectangle([x1,y1,x2,y2], fill=(255,255,255,opacity), outline=(255,255,255,outline), width=2)

# -----------------------
# Selection
# -----------------------
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

# -----------------------
# Chart rendering (CANDLES; NO grid; white logos)
# -----------------------
def render_candles(draw: ImageDraw.ImageDraw, ohlc: pd.DataFrame, box: Tuple[int,int,int,int]):
    x1,y1,x2,y2 = box
    data = ohlc[["Open","High","Low","Close"]].dropna()
    if data.empty: return
    lows, highs = data["Low"].to_numpy(), data["High"].to_numpy()
    opens, closes = data["Open"].to_numpy(), data["Close"].to_numpy()
    pmin, pmax = float(np.nanmin(lows)), float(np.nanmax(highs))
    pr = max(1e-9, pmax-pmin)
    n = len(data)
    xs = np.linspace(x1+10, x2-10, n)
    body_w = max(4, int((x2-x1) / max(10, n*1.8)))
    wick_w = 2
    def y(p): return y2 - (float(p)-pmin)/pr*(y2-y1)
    for i in range(n):
        ox, cx, hi, lo = opens[i], closes[i], highs[i], lows[i]
        X = int(xs[i])
        # wick
        draw.line([(X,int(y(lo))),(X,int(y(hi)))], fill=(245,250,255,160), width=wick_w)
        # body (soft green/red tuned for blue bg)
        up = cx >= ox
        top, bot = y(max(ox,cx)), y(min(ox,cx))
        color = (90, 230, 150, 255) if up else (245, 110, 110, 255)
        draw.rectangle([X-body_w, int(top), X+body_w, int(bot)], fill=color, outline=None)

def generate_chart(ticker: str) -> Optional[str]:
    try:
        # Use 1y weekly for support zone context; 6mo daily for candles looks denser; but we’ll keep 1y weekly candles as you liked.
        df = yf.download(ticker, period="1y", interval="1wk", progress=False, auto_adjust=False, threads=False)
        if df is None or df.empty:
            log(f"[warn] no data for {ticker}")
            return None
        ohlc = extract_ohlc(df, ticker)
        if ohlc.empty:
            log(f"[warn] no ohlc for {ticker}")
            return None
        close_s = ohlc["Close"]

        # Canvas
        W,H = 1080, 720
        img  = blue_gradient_bg(W,H)
        d    = ImageDraw.Draw(img)

        # Plot area
        margin = 40
        x1,y1 = margin, margin+30
        x2,y2 = W - margin, H - margin

        # Support zone (very subtle translucent rectangle)
        lo, hi = swing_levels(close_s, 10)
        if lo is not None and hi is not None and hi >= lo:
            pmin, pmax = float(ohlc["Low"].min()), float(ohlc["High"].max())
            pr = max(1e-9, pmax-pmin)
            def y(p): return y2 - (float(p)-pmin)/pr*(y2-y1)
            y_top, y_bot = int(y(hi)), int(y(lo))
            draw_support_zone(d, x1+6, min(y_top,y_bot), x2-6, max(y_top,y_bot), opacity=24, outline=60)

        # Candles (NO grid, only candles)
        render_candles(d, ohlc, (x1, y1, x2, y2))

        # Logos: company (white mono) top-left, TWD white bottom-right
        lg_col = load_logo_color(ticker, 170)
        if lg_col is not None:
            lg_white = to_white_mono(lg_col, alpha=255)
            img.alpha_composite(lg_white, (x1, 16))
        twd = load_twd_white(160)
        if twd is not None:
            img.alpha_composite(twd, (W - twd.width - 18, H - twd.height - 14))

        out = os.path.join(CHART_DIR, f"{ticker}_chart.png")
        img.convert("RGB").save(out, "PNG")
        return out
    except Exception as e:
        log(f"[error] generate_chart({ticker}): {e}")
        return None

# -----------------------
# Captions (unchanged behavior: single CTA at end)
# -----------------------
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
    return f"{v:.1f}" if abs(v) < 10 else f"{v:.0f}"

def chart_caption_line_for(ticker: str, daily_close: pd.Series, weekly_close: pd.Series) -> Tuple[str,str]:
    e = SECTOR_EMOJI.get(ticker, "📈")
    p30 = _fmt_pct(pct_change(daily_close, 30))
    if is_breakout(daily_close, 20):
        mood = "breakout"
    elif pct_change(daily_close, 10) < 0:
        mood = "pullback"
    elif True:  # rely on weekly zone for "near_support"
        mood = "near_support" if swing_levels(weekly_close, 10)[0] is not None else "trend"
    line = random.choice(CHART_TEMPLATES[mood]).format(e=e, t=ticker, p=p30)
    return (re.sub(r"\s+"," ", line).strip(), mood)

def build_summary_cta(mood_counts: Dict[str,int], tickers: List[str]) -> str:
    mood = "trend"
    if mood_counts:
        mood = max(mood_counts.items(), key=lambda kv: kv[1])[0] or "trend"
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
    return random.choice(prompts.get(mood, prompts["trend"]))

# -----------------------
# Posters (kept as before)
# -----------------------
def poster_bg(W=1080,H=1080): return blue_gradient_bg(W,H)

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
        W,H=1080,1080
        img=poster_bg(W,H); d=ImageDraw.Draw(img)
        # tag
        tag_font=font_bold(44); pad=12; txt="NEWS"
        tw,th=d.textbbox((0,0),txt,font=tag_font)[2:]
        d.rounded_rectangle([40,40,40+tw+2*pad,40+th+2*pad],14,fill=(0,36,73,210))
        d.text((40+pad,40+pad),txt,fill="white",font=tag_font)
        # headline
        hfont=font_bold(100)
        head=wrap_text(d, headline.upper(), hfont, W-80)
        d.multiline_text((40,150), head, font=hfont, fill="white", spacing=10, align="left")
        # sub
        sfont=font_reg(46)
        subw=wrap_text(d, subtext, sfont, W-80)
        d.multiline_text((40,420), subw, font=sfont, fill=(235,243,255,255), spacing=10, align="left")
        # logos (color on right for posters, per your preference)
        lg = load_logo_color_safe(ticker, 220)
        if lg is not None:
            img.alpha_composite(lg, (W - lg.width - 40, 40))
        twd=load_twd_white(220)
        if twd is not None:
            img.alpha_composite(twd, (W - twd.width - 40, H - twd.height - 40))
        out=os.path.join(POSTER_DIR, f"{ticker}_poster_{DATESTR}.png")
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
    elif pct_change(d_close, 10) > 0: pa="buy-the-dip flows still showing up"
    else: pa="consolidation looks orderly"
    core = f"{e} {ticker} — still in focus as sentiment shifts."
    tail = f"30d: {p30:+.1f}% • 5d: {p5:+.1f}% · {pa}."
    return core+"\n"+tail

# -----------------------
# Workflows
# -----------------------
def run_daily_charts():
    tickers = pick_tickers(6)
    log(f"[info] selected tickers: {tickers}")
    generated = []
    cap_lines = []
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

    seen, uniq = set(), []
    for it in items:
        k = re.sub(r"[^a-z0-9 ]+","", it["title"].lower()).strip()
        if k in seen: continue
        seen.add(k); uniq.append(it)

    if not uniq:
        print("\n⚠️ No news found → posters skipped\n")
        log("[info] no news → posters skipped")
        return

    picks = uniq[:2]
    made = 0
    for it in picks:
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

# -----------------------
# CLI with CI back-compat
# -----------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--posters", action="store_true", help="Generate news-driven posters (optional)")
    ap.add_argument("--once", type=str, help="Generate a single ticker chart")
    ap.add_argument("--ci", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--ci-posters", action="store_true", help=argparse.SUPPRESS)
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
            print("\n✅ Chart saved:", out, "\n" if out else "\n❌ Chart failed (see run.log)\n")
        else:
            log("[info] default mode → daily charts")
            run_daily_charts()
    except Exception as e:
        log(f"[fatal] {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
