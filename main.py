#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TrendWatchDesk - main.py (stable, IG-ready, CI back-compat)
Default: daily chart run
Optional: --posters to generate news-driven posters
Back-compat: --ci (charts), --ci-posters (posters)

Charts
- 1080x720, 1y weekly candlesticks
- Blue gradient background, NO grid
- Subtle translucent support zone rectangle (not solid)
- Company logo (color) top-left, TWD logo (white) bottom-right
- NO ticker text on the image

Daily Captions
- Casual tone, sector emojis, ONLY 30d percent (no price)
- High variety, not repetitive
- ONE contextual summary CTA at the very end (mood + sectors)

Posters (optional, isolated for stability)
- 1080x1080, blue gradient + beams
- Headline (Grift-Bold) + subtext (Grift-Regular)
- Company logo (color) top-right, TWD (white) bottom-right
- Caption tied to headline intent + recent price action (30d/5d)
"""

import os, re, math, random, hashlib, datetime, traceback
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

# Pools (includes SOFI, Robotics, Quantum)
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

# Sector emojis
SECTOR_EMOJI = {
    "AAPL":"🍏","MSFT":"🧠","NVDA":"🤖","AMD":"🔧","TSLA":"🚗","META":"📡","GOOGL":"🔎","AMZN":"📦",
    "SPY":"📊","QQQ":"📈","GLD":"🪙","SOFI":"🏦","IONQ":"🧪","IBM":"💼","ISRG":"🩺","ROK":"🏭",
    "AVGO":"📶","TSM":"🏭","INTC":"💽","ASML":"🔬","NFLX":"🎬","DIS":"🏰","BABA":"🛍️","NIO":"🔌",
    "SHOP":"🛒","PLTR":"🛰️","MA":"💳","V":"💳","PYPL":"💸","SQ":"📱"
}

# Friendly names for pool summary in CTA
POOL_LABEL = {
    "AI": "AI megacaps",
    "MAG7": "Mega-cap tech",
    "Semis": "semiconductors",
    "Healthcare": "healthcare",
    "Fintech": "fintech",
    "Quantum": "quantum computing",
    "Robotics": "robotics",
    "Wildcards": "wildcards",
}

# Reverse map: ticker -> sector label (first pool hit wins; good enough for summary)
SECTOR_OF: Dict[str, str] = {}
for pool, arr in POOLS.items():
    label = POOL_LABEL.get(pool, pool)
    for t in arr:
        SECTOR_OF.setdefault(t, label)

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
    if s is None or len(s) == 0:
        return pd.Series([], dtype="float64")
    return pd.to_numeric(s, errors="coerce").astype("float64").dropna()

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
    lo = lows.iloc[-1]
    hi = highs.iloc[-1]
    lo = None if pd.isna(lo) else _as_float(lo)
    hi = None if pd.isna(hi) else _as_float(hi)
    return (lo, hi)

# Price action flags
def is_breakout(series: pd.Series, lookback: int = 20) -> bool:
    s = _series_f64(series)
    if len(s) < lookback + 1: return False
    return float(s.iloc[-1]) >= float(s.iloc[-(lookback+1):].max())

def is_breakdown(series: pd.Series, lookback: int = 20) -> bool:
    s = _series_f64(series)
    if len(s) < lookback + 1: return False
    return float(s.iloc[-1]) <= float(s.iloc[-(lookback+1):].min())

def near_support_flag(wk_close: pd.Series, lookback: int = 10, tol: float = 0.6) -> bool:
    s = _series_f64(wk_close)
    lo, hi = swing_levels(s, lookback)
    if lo is None or hi is None or hi <= lo: return False
    mid = 0.5 * (lo + hi)
    rngp = max(1e-9, hi - lo)
    last = float(s.iloc[-1])
    return abs(last - mid) <= tol * rngp

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

    # subtle beams
    beams = Image.new("RGBA", (W,H), (0,0,0,0))
    d = ImageDraw.Draw(beams)
    for i, a in enumerate([70,55,40]):
        off = i*140
        d.polygon([(0, 120+off), (W, 0+off), (W, 110+off), (0, 230+off)], fill=(255,255,255,a))
    beams = beams.filter(ImageFilter.GaussianBlur(45))
    return Image.alpha_composite(bg, beams)

def draw_support_zone(d: ImageDraw.ImageDraw, x1,y1,x2,y2, opacity=34, outline=88):
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
    x1,y1,x2,y2 = box
    data = ohlc.copy()
    for col in ["Open","High","Low","Close"]:
        data[col] = pd.to_numeric(data[col], errors="coerce").astype("float64")
    data = data.dropna()
    if data.empty: return

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
        color = (92, 220, 130, 255) if up else (235, 95, 95, 255)
        draw.rectangle([X - body_w, int(top), X + body_w, int(bot)], fill=color, outline=None)

def generate_chart(ticker: str) -> Optional[str]:
    try:
        df = yf.download(ticker, period="1y", interval="1wk", progress=False, auto_adjust=False, threads=False)
        if df.empty:
            log(f"[warn] no data for {ticker}")
            return None

        # Canvas
        W,H = 1080, 720
        img = blue_gradient_bg(W,H)
        d   = ImageDraw.Draw(img)

        # Plot area
        margin = 40
        top_pad = 30
        bot_pad = 40
        x1,y1 = margin, top_pad + 40
        x2,y2 = W - margin, H - bot_pad

        # Support zone (subtle)
        close_s = _series_f64(df["Close"])
        if not close_s.empty:
            sup_lo, sup_hi = swing_levels(close_s, lookback=10)
            if sup_lo is not None and sup_hi is not None and sup_hi >= sup_lo:
                low  = float(pd.to_numeric(df["Low"], errors="coerce").min())
                high = float(pd.to_numeric(df["High"], errors="coerce").max())
                pr   = max(1e-9, high-low)
                def y_from_p(p): return y2 - (float(p)-low)/pr*(y2-y1)
                y_top = int(y_from_p(sup_hi))
                y_bot = int(y_from_p(sup_lo))
                draw_support_zone(d, x1+6, min(y_top,y_bot), x2-6, max(y_top,y_bot), opacity=34, outline=88)

        # Candles
        render_candles(d, df[["Open","High","Low","Close"]], (x1, y1, x2, y2))

        # Logos
        lg = load_logo_color(ticker, 160)
        if lg is not None:
            img.alpha_composite(lg, (x1, 10))  # color logo, top-left
        twd = load_twd_white(160)
        if twd is not None:
            img.alpha_composite(twd, (W - twd.width - 20, H - twd.height - 16))  # white logo, bottom-right

        out = os.path.join(CHART_DIR, f"{ticker}_chart.png")
        img.convert("RGB").save(out, "PNG")
        return out
    except Exception as e:
        log(f"[error] generate_chart({ticker}): {e}")
        return None

# -----------------------
# Daily Captions (natural, 30d%, single summary CTA at end)
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

def _fmt_pct(val: float) -> str:
    return f"{val:.1f}" if abs(val) < 10 else f"{val:.0f}"

def chart_caption_line_for(ticker: str, daily_close: pd.Series, weekly_close: pd.Series) -> Tuple[str, str]:
    """Returns (line, mood) where mood is one of breakout/near_support/pullback/trend."""
    e = SECTOR_EMOJI.get(ticker, "📈")
    p30 = _fmt_pct(pct_change(daily_close, 30))
    if is_breakout(daily_close, 20):
        mood = "breakout"
    elif near_support_flag(weekly_close, 10, 0.6):
        mood = "near_support"
    elif pct_change(daily_close, 10) < 0:
        mood = "pullback"
    else:
        mood = "trend"
    line = random.choice(CHART_TEMPLATES[mood]).format(e=e, t=ticker, p=p30)
    return (re.sub(r"\s+"," ", line).strip(), mood)

def build_summary_cta(mood_counts: Dict[str,int], tickers: List[str]) -> str:
    """One-line summary CTA at the end: sector overview + a context question."""
    mood = "trend"
    if mood_counts:
        mood = max(mood_counts.items(), key=lambda kv: kv[1])[0] or "trend"

    sectors, seen = [], set()
    for t in tickers:
        s = SECTOR_OF.get(t)
        if s and s not in seen:
            seen.add(s); sectors.append(s)

    if not sectors:
        sector_phrase = "the tape"
    elif len(sectors) == 1:
        sector_phrase = sectors[0]
    else:
        sector_phrase = f"{sectors[0]} & {sectors[1]}"

    prompts = {
        "breakout": [
            f"{sector_phrase} showing breakouts — how far can momentum carry from here?",
            f"Breakout day for {sector_phrase}. Do you see follow-through next?",
            f"{sector_phrase} leading — continuation or fade from here?",
        ],
        "near_support": [
            f"{sector_phrase} holding the base — does this look like a durable floor?",
            f"Bounces near support in {sector_phrase}. Would you add on dips?",
            f"{sector_phrase} defending the zone — base building or just a pause?",
        ],
        "pullback": [
            f"Pullbacks across {sector_phrase} — healthy reset or trend change?",
            f"{sector_phrase} cooling off — are you buying this dip or waiting?",
            f"Red day in {sector_phrase}. Is this the reset it needed?",
        ],
        "trend": [
            f"{sector_phrase} still trending — is there more room to run?",
            f"Uptrends intact in {sector_phrase}. Ride it or trim here?",
            f"{sector_phrase} steady today — what’s your plan into next week?",
        ],
    }
    return random.choice(prompts.get(mood, prompts["trend"]))

# -----------------------
# Posters (opt-in)
# -----------------------
HEADLINE_TAGS = [
    ("deal",       r"\b(deal|partnership|alliance|collaborat|tie[- ]?up|expands?\s+with|teams?\s+up)\b"),
    ("guidance",   r"\b(guidance|outlook|forecast|raises|lifts|cuts|lowers)\b"),
    ("earnings",   r"\b(earnings|results|eps|beat|miss)\b"),
    ("product",    r"\b(chip|platform|feature|launch|rollout|roadmap|ai|model)\b"),
    ("legal",      r"\b(lawsuit|antitrust|probe|investigation|settlement)\b"),
    ("macro",      r"\b(rates|inflation|tariff|regulation|ban|export controls?)\b"),
]

POSTER_TEMPLATES = {
    "deal": [
        "{e} {t} — new deal making waves, traders watching the upside.",
        "{e} {t} — partnership headlines, sentiment leaning positive.",
    ],
    "guidance": [
        "{e} {t} — guidance shift draws attention, market weighing the outlook.",
        "{e} {t} — revised forecasts spark a fresh look at margins.",
    ],
    "earnings": [
        "{e} {t} — earnings reaction sets the tone this week.",
        "{e} {t} — numbers out, traders watching the follow-through.",
    ],
    "product": [
        "{e} {t} — product news keeps the sector moving.",
        "{e} {t} — launch chatter adding to momentum.",
    ],
    "legal": [
        "{e} {t} — legal overhang keeps risk on the table.",
    ],
    "macro": [
        "{e} {t} — macro backdrop feeding into the story.",
    ],
    "default": [
        "{e} {t} — still in focus as sentiment shifts.",
    ],
}

def tag_headline(headline: str) -> Optional[str]:
    h = headline.lower()
    for label, pat in HEADLINE_TAGS:
        if re.search(pat, h): return label
    return None

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

def generate_poster_image(ticker: str, headline: str, subtext: str) -> Optional[str]:
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

        # Subtext
        sub_font = font_reg(46)
        sub_wrapped = wrap_text(d, subtext, sub_font, W-80)
        d.multiline_text((40, 420), sub_wrapped, font=sub_font, fill=(235,243,255,255), spacing=10, align="left")

        # Logos
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
        log(f"[error] generate_poster_image({ticker}): {e}")
        return None

def build_poster_caption(ticker: str, headline: str, d_close: pd.Series) -> str:
    e = SECTOR_EMOJI.get(ticker, "📈")
    p5  = _fmt_pct(pct_change(d_close, 5))
    p30 = _fmt_pct(pct_change(d_close, 30))
    angle = tag_headline(headline) or "default"

    # price action tail
    if is_breakout(d_close, 20):
        pa = "breakout on the chart"
    elif is_breakdown(d_close, 20):
        pa = "breakdown risk in view"
    elif pct_change(d_close, 10) > 0:
        pa = "buy-the-dip flows still showing up"
    else:
        pa = "consolidation looks orderly"

    core = random.choice(POSTER_TEMPLATES.get(angle, POSTER_TEMPLATES["default"])).format(e=e, t=ticker)
    tail = f"30d: +{p30}% • 5d: +{p5}% · {pa}."
    return core + "\n" + tail

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
                line, mood = chart_caption_line_for(t, ddf["Close"], wdf["Close"])
                mood_counts[mood] = mood_counts.get(mood,0)+1
                cap_lines.append(line)
            except Exception:
                pass
        else:
            log(f"[warn] chart failed for {t}")

    # Single contextual summary CTA at the end
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
        for p in generated:
            print(" -", p)
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
            data = r.json()
            for n in data.get("news", [])[:10]:
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
            dclose = yf.download(t, period="6mo", interval="1d", progress=False, auto_adjust=False, threads=False)["Close"]
            cap = build_poster_caption(t, title, dclose)
            with open(os.path.splitext(out)[0] + "_caption.txt", "w", encoding="utf-8") as f:
                f.write(cap)
            log(f"[info] poster saved: {out}")

    print("\n==============================")
    print(f"Posters generated: {made}")
    print("==============================\n")

# -----------------------
# CLI (with CI back-compat)
# -----------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    # New flags
    ap.add_argument("--posters", action="store_true", help="Generate news-driven posters (optional)")
    ap.add_argument("--once", type=str, help="Generate a single ticker chart")
    # Legacy flags for CI back-compat
    ap.add_argument("--ci", action="store_true", help=argparse.SUPPRESS)          # legacy: run daily charts
    ap.add_argument("--ci-posters", action="store_true", help=argparse.SUPPRESS)  # legacy: run posters
    args = ap.parse_args()

    try:
        # Legacy mappings first
        if args.ci:
            log("[info] legacy --ci → daily charts")
            run_daily_charts()
        elif args.ci_posters:
            log("[info] legacy --ci-posters → posters")
            run_posters()
        # New flags
        elif args.posters:
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
