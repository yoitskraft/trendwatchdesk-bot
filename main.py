#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TrendWatchDesk - main.py
Single source of truth aligned with OPERATIONS_GUIDE.md

Features:
- Weekly charts (6 tickers per run) with support zone, logos, Grift fonts
- Posters (breaking news) with IG-native gradient layout, Grift fonts, wrapped subtext
- Captions (daily + poster) with natural, news-driven tone and emojis
- Poster captions ALWAYS complement poster copy (no duplication)
- Watchlist + weighted pools; deterministic selection by date
- Yahoo Finance polling + simple clustering; graceful fallbacks
- Outputs: output/charts/, output/posters/, output/caption_YYYYMMDD.txt, run.log
"""

import os, sys, io, re, math, json, time, random, hashlib, traceback, datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# =========================
# ---- Global Configs -----
# =========================
ASSETS_DIR        = "assets"
LOGO_DIR          = os.path.join(ASSETS_DIR, "logos")
FONT_DIR          = os.path.join(ASSETS_DIR, "fonts")
BRAND_LOGO        = os.path.join(ASSETS_DIR, "brand_logo.png")
FONT_BOLD_PATH    = os.path.join(FONT_DIR, "Grift-Bold.ttf")
FONT_REG_PATH     = os.path.join(FONT_DIR, "Grift-Regular.ttf")

OUTPUT_DIR        = "output"
CHART_DIR         = os.path.join(OUTPUT_DIR, "charts")
POSTER_DIR        = os.path.join(OUTPUT_DIR, "posters")
CAPTION_TXT       = os.path.join(OUTPUT_DIR, f"caption_{datetime.date.today().strftime('%Y%m%d')}.txt")
RUN_LOG           = os.path.join(OUTPUT_DIR, "run.log")

# Create dirs
for d in (OUTPUT_DIR, CHART_DIR, POSTER_DIR):
    os.makedirs(d, exist_ok=True)

# Deterministic seed per day (stable selections)
TODAY      = datetime.date.today()
DATESTAMP  = TODAY.strftime("%Y%m%d")
SEED       = int(hashlib.sha1(DATESTAMP.encode()).hexdigest(), 16) % (10**8)
rng        = random.Random(SEED)

# Watchlist/pools (align with Ops Guide)
WATCHLIST = ["AAPL","MSFT","NVDA","AMD","TSLA","SPY","QQQ","GLD","AMZN","META","GOOGL"]

POOLS = {
    "AI": ["NVDA","MSFT","GOOGL","META","AMZN"],
    "MAG7": ["AAPL","MSFT","GOOGL","META","AMZN","NVDA","TSLA"],
    "Semis": ["NVDA","AMD","AVGO","TSM","INTC","ASML"],
    "Healthcare": ["UNH","JNJ","PFE","MRK","LLY"],
    "Fintech": ["MA","V","PYPL","SQ"],
    "Quantum": ["IONQ","IBM","AMZN"],
    "Wildcards": ["NFLX","DIS","BABA","NIO","SHOP","PLTR"]
}

# HTTP session
SESS = requests.Session()
SESS.headers.update({
    "User-Agent": "TrendWatchDesk/1.0 (+github actions)",
    "Accept": "*/*",
    "Accept-Encoding": "identity"
})

# Emojis by sector-ish feel (loose)
SECTOR_EMOJI = {
    "NVDA":"🤖","AMD":"🔧","TSLA":"🚗","AAPL":"🍏","MSFT":"🧠",
    "META":"📡","GOOGL":"🔎","AMZN":"📦","SPY":"📊","QQQ":"📈","GLD":"🪙"
}

# =========================
# ---- Logging helpers ----
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
# ---- Fonts (Grift) ------
# =========================
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        # Fallback to default if font not found
        return ImageFont.load_default()

def font_bold(size: int) -> ImageFont.FreeTypeFont:
    return _font(FONT_BOLD_PATH, size)

def font_reg(size: int) -> ImageFont.FreeTypeFont:
    return _font(FONT_REG_PATH, size)

# =========================
# ---- Chart Generator ----
# =========================
def pick_tickers(n: int = 6) -> List[str]:
    # Weighted pick: pull more from AI/MAG7/Semis; fill with others
    picks = set()
    def grab(pool, k):
        cands = [t for t in POOLS[pool] if t not in picks]
        rng.shuffle(cands)
        picks.update(cands[:k])
    grab("AI", 2)
    grab("MAG7", 2)
    grab("Semis", 1)
    # Fill remainder from others
    flat_others = [t for k,v in POOLS.items() if k not in ("AI","MAG7","Semis") for t in v]
    rng.shuffle(flat_others)
    for t in flat_others:
        if len(picks) >= n: break
        picks.add(t)
    return list(picks)[:n]

def swing_levels(series: pd.Series, lookback: int = 14) -> Tuple[Optional[float], Optional[float]]:
    """Simple swing high/low detection for support zone."""
    if series is None or series.empty: return (None, None)
    highs = series.rolling(lookback).max()
    lows  = series.rolling(lookback).min()
    return float(lows.iloc[-1]) if not math.isnan(lows.iloc[-1]) else None, \
           float(highs.iloc[-1]) if not math.isnan(highs.iloc[-1]) else None

def pct_change(series: pd.Series, days: int = 30) -> float:
    try:
        if len(series) < days+1: return 0.0
        return float((series.iloc[-1] - series.iloc[-days-1]) / series.iloc[-days-1] * 100.0)
    except Exception:
        return 0.0

def load_logo(ticker: str, target_w: int) -> Optional[Image.Image]:
    path = os.path.join(LOGO_DIR, f"{ticker}.png")
    if not os.path.exists(path):
        return None
    try:
        img = Image.open(path).convert("RGBA")
        w, h = img.size
        ratio = target_w / max(1, w)
        new = img.resize((int(w*ratio), int(h*ratio)), Image.Resampling.LANCZOS)
        return new
    except Exception:
        return None

def twd_logo(target_w: int) -> Optional[Image.Image]:
    if not os.path.exists(BRAND_LOGO): return None
    try:
        logo = Image.open(BRAND_LOGO).convert("RGBA")
        # tint to white using alpha only (so it blends on gradients)
        r,g,b,a = logo.split()
        white = Image.new("RGBA", logo.size, (255,255,255,0))
        white.putalpha(a)
        w, h = white.size
        ratio = target_w / max(1, w)
        new = white.resize((int(w*ratio), int(h*ratio)), Image.Resampling.LANCZOS)
        return new
    except Exception:
        return None

def draw_support_box(draw: ImageDraw.ImageDraw, chart_box: Tuple[int,int,int,int], low: float, high: float, y_from_price) -> None:
    if low is None or high is None: return
    x1,y1,x2,y2 = chart_box
    y_low  = y_from_price(high)   # translucent band around zone mid
    y_high = y_from_price(low)
    box = [x1+2, min(y_low, y_high), x2-2, max(y_low, y_high)]
    draw.rectangle(box, fill=(40,120,255,48), outline=(40,120,255,160), width=2)

def generate_chart(ticker: str) -> Optional[str]:
    """Weekly close chart with support zone + logos + meta text."""
    try:
        df = yf.download(ticker, period="1y", interval="1wk", progress=False, auto_adjust=False, threads=False)
        if df.empty:
            log(f"[warn] No data for {ticker}")
            return None
        close = df["Close"].dropna()
        last = float(close.iloc[-1])
        chg30 = pct_change(close, days=30)
        chg1d = 0.0
        sup_low, sup_high = swing_levels(close, lookback=10)

        # Canvas
        W,H = 1080, 720
        img = Image.new("RGBA", (W,H), (255,255,255,255))
        draw = ImageDraw.Draw(img)

        # Areas
        margin = 40
        header_h = 150
        footer_h = 70
        plot = [margin+30, margin+header_h, W-margin-30, H-margin-footer_h]

        # Title block
        f_ticker = font_bold(76)
        f_meta   = font_reg(38)
        ticker_text = ticker
        draw.text((margin+30, margin+30), ticker_text, fill=(0,0,0,255), font=f_ticker)

        meta = f"{last:,.2f} • {chg30:+.2f}% (30d)"
        draw.text((margin+30, margin+100), meta, fill=(30,30,30,255), font=f_meta)

        # Plot close (simple line for clean look)
        x1,y1,x2,y2 = plot
        xs = np.linspace(x1, x2, num=len(close))
        minp, maxp = float(close.min()), float(close.max())
        prange = max(1e-8, maxp - minp)

        def y_from_price(p):
            # invert (higher price = lower y)
            return y2 - (p - minp) / prange * (y2 - y1)

        pts = [(int(xs[i]), int(y_from_price(close.iloc[i]))) for i in range(len(close))]
        for i in range(1, len(pts)):
            draw.line([pts[i-1], pts[i]], fill=(0,0,0,255), width=3)

        # subtle grid
        for gy in np.linspace(y1, y2, 5):
            draw.line([(x1, gy),(x2, gy)], fill=(0,0,0,30), width=1)

        # Support zone
        draw_support_box(draw, plot, sup_low, sup_high, y_from_price)

        # Logos
        logo = load_logo(ticker, target_w=140)
        if logo:
            img.alpha_composite(logo, (W - logo.width - 26, 24))
        twd = twd_logo(target_w=160)
        if twd:
            img.alpha_composite(twd, (W - twd.width - 26, H - twd.height - 24))

        # Subheading
        f_sub = font_reg(32)
        draw.text((x1, y1-56), "Weekly chart • key support zone", fill=(40,40,40,255), font=f_sub)

        out = os.path.join(CHART_DIR, f"{ticker}_chart.png")
        img.convert("RGB").save(out, "PNG")
        return out
    except Exception as e:
        log(f"[error] generate_chart({ticker}): {e}")
        return None

# =========================
# ---- Caption Engine -----
# =========================
def caption_daily(ticker: str, last: float, chg30: float, near_support: bool) -> str:
    """For charts (daily workflow) — natural, non-repetitive."""
    emj = SECTOR_EMOJI.get(ticker, "📈")
    cues = []
    if chg30 >= 8: cues.append("momentum building 🔥")
    if chg30 <= -8: cues.append("recent pullback on the radar ⚠️")
    if near_support: cues.append("buyers defending support 🛡️")
    if not cues:
        cues = rng.sample([
            "steady drift with eyes on catalysts",
            "range tightening as traders wait for a trigger",
            "quiet grind higher while liquidity improves"
        ], k=1)
    cue = rng.choice(["; ".join(cues), " · ".join(cues)])

    ctas = [
        "Save for later 📌",
        "Your take below 👇",
        "Share with a friend 🔄"
    ]
    cta = rng.choice(ctas)

    body = f"{emj} {ticker} at {last:,.2f} — {chg30:+.2f}% (30d). {cue}. {cta}"
    return body

def caption_poster(ticker: str, poster_headline: str) -> str:
    """
    For posters — MUST complement poster body.
    Structure: Hook → Added context → Forward angle → CTA
    """
    hook = f"{SECTOR_EMOJI.get(ticker, '📈')} {ticker} — still in the spotlight"
    context = (
        "Beyond the headline, investors are weighing sector read-throughs, "
        "peer reactions, and what this means for near-term demand."
    )
    fwd = (
        "Next up: guidance and earnings commentary — the market wants detail "
        "on margins and runway."
    )
    ctas = [
        "What’s your take? Drop a comment 👇",
        "Save this for later 📌",
        "Share with someone tracking the story 🔄"
    ]
    return f"{hook}\n{context}\n{fwd}\n\n{rng.choice(ctas)}"

# =========================
# ---- Poster Engine ------
# =========================
def poster_background(W=1080, H=1080) -> Image.Image:
    """Blue gradient + diagonal beams."""
    base = Image.new("RGB", (W, H), "#0d3a66")
    grad = Image.new("RGB", (W, H))
    for y in range(H):
        t = y / H
        r = int(10 + (20-10)*t)
        g = int(58 + (130-58)*t)
        b = int(102 + (220-102)*t)
        for x in range(W):
            grad.putpixel((x, y), (r, g, b))
    bg = Image.blend(base, grad, 0.9)

    beams = Image.new("RGBA", (W, H), (0,0,0,0))
    d = ImageDraw.Draw(beams)
    for i, alpha in enumerate([80, 60, 40]):
        off = i * 140
        d.polygon([(0, 140+off), (W, 0+off), (W, 120+off), (0, 260+off)], fill=(255,255,255,alpha))
    beams = beams.filter(ImageFilter.GaussianBlur(45))
    return Image.alpha_composite(bg.convert("RGBA"), beams)

def wrap_text_by_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width_px: int) -> str:
    words = text.split()
    lines, line = [], ""
    for w in words:
        test = f"{line}{w} "
        tw = draw.textbbox((0,0), test, font=font)[2]
        if tw > max_width_px and line:
            lines.append(line.rstrip())
            line = w + " "
        else:
            line = test
    if line:
        lines.append(line.rstrip())
    return "\n".join(lines)

def draw_news_tag(draw, x=40, y=40):
    tag = "NEWS"
    f = font_bold(42)
    pad = 14
    tw, th = draw.textbbox((0,0), tag, font=f)[2:]
    rect = (x, y, x + tw + pad*2, y + th + pad*2)
    draw.rounded_rectangle(rect, radius=12, fill=(0,36,73,200))
    draw.text((x+pad, y+pad), tag, font=f, fill=(255,255,255,255))

def generate_poster(ticker: str, headline_lines: List[str], subtext_lines: List[str]) -> Optional[str]:
    """
    IG-native poster with:
    - Headline (Grift-Bold, left-aligned)
    - Subtext (Grift-Regular, 3–4 lines), wrapped to safe margins
    - Ticker logo top-right; TWD bottom-right
    """
    try:
        W,H = 1080,1080
        img = poster_background(W,H)
        draw = ImageDraw.Draw(img)

        draw_news_tag(draw, 40, 40)

        # Headline
        f_head = font_bold(108)
        head = "\n".join(headline_lines)
        draw.multiline_text((40, 160), head, font=f_head, fill=(255,255,255,255), spacing=10, align="left")

        # Subtext (wrapped)
        f_sub = font_reg(48)
        sub = " ".join(subtext_lines)
        wrapped = wrap_text_by_width(draw, sub, f_sub, max_width_px=W-80)
        draw.multiline_text((40, 420), wrapped, font=f_sub, fill=(235,243,255,255), spacing=10, align="left")

        # Ticker logo top-right (fallback badge if missing)
        logo = load_logo(ticker, target_w=220)
        if logo is None:
            # badge fallback using Grift-Bold text
            badge_w, badge_h = 220, 110
            badge = Image.new("RGBA", (badge_w, badge_h), (255,255,255,35))
            bd = ImageDraw.Draw(badge)
            bd.rounded_rectangle([0,0,badge_w,badge_h], radius=20, outline=(255,255,255,120), width=2, fill=(255,255,255,28))
            bf = font_bold(64)
            tw = bd.textbbox((0,0), ticker, font=bf)[2]
            bd.text(((badge_w-tw)//2, (badge_h-64)//2 - 2), ticker, font=bf, fill=(255,255,255,255))
            img.alpha_composite(badge, (W - badge_w - 40, 40))
        else:
            img.alpha_composite(logo, (W - logo.width - 40, 40))

        # TWD logo bottom-right (white tinted)
        twd = twd_logo(target_w=220)
        if twd:
            # subtle shadow
            shadow = Image.new("RGBA", twd.size, (0,0,0,0))
            sd = ImageDraw.Draw(shadow)
            sd.bitmap((0,0), twd.split()[3], fill=(0,0,0,80))
            shadow = shadow.filter(ImageFilter.GaussianBlur(6))
            pos = (W - twd.width - 40, H - twd.height - 40)
            img.alpha_composite(shadow, pos)
            img.alpha_composite(twd, pos)

        out = os.path.join(POSTER_DIR, f"{ticker}_poster_{DATESTAMP}.png")
        img.convert("RGB").save(out, "PNG")
        return out
    except Exception as e:
        log(f"[error] generate_poster({ticker}): {e}")
        return None

# =========================
# ---- News Polling -------
# =========================
YF_NEWS_ENDPOINT = "https://query1.finance.yahoo.com/v1/finance/search"

def fetch_yahoo_headlines(tickers: List[str], max_items: int = 40) -> List[Dict]:
    """
    Lightweight news pull (Yahoo). If it fails, return empty and skip posters.
    """
    items = []
    for t in tickers:
        try:
            # Best-effort: Yahoo Finance search endpoint (subject to change)
            params = {"q": t, "quotesCount": 0, "newsCount": 10}
            r = SESS.get(YF_NEWS_ENDPOINT, params=params, timeout=10)
            if r.status_code != 200:
                continue
            data = r.json()
            news = data.get("news", [])[:10]
            for n in news:
                title = n.get("title") or ""
                link  = n.get("link") or ""
                pub   = n.get("providerPublishTime")
                if not title or not link: 
                    continue
                items.append({"ticker": t, "title": title, "url": link, "ts": pub})
        except Exception as e:
            log(f"[warn] yahoo fetch for {t} failed: {e}")
    # de-dupe by title
    seen = set()
    uniq = []
    for it in items:
        key = it["title"].strip().lower()
        if key in seen: 
            continue
        seen.add(key)
        uniq.append(it)
    return uniq[:max_items]

def cluster_by_popularity(items: List[Dict]) -> List[Dict]:
    """
    Simple heuristic: pick items whose titles mention broader themes or recur across multiple tickers.
    """
    if not items: return []
    # Count occurrences by normalized title segments
    counts = {}
    for it in items:
        key = re.sub(r"[^a-z0-9 ]+","", it["title"].lower()).strip()
        counts[key] = counts.get(key, 0) + 1
    # Keep items with count >= 2, else fallback to top few
    pops = [it for it in items if counts.get(re.sub(r"[^a-z0-9 ]+","", it["title"].lower()).strip(),0) >= 2]
    if not pops:
        pops = items[:6]
    return pops

# =========================
# ---- Workflows ----------
# =========================
def run_daily_charts():
    """Mon/Wed/Fri workflow equivalent: generate 6 charts + a caption file."""
    tickers = pick_tickers(6)
    log(f"[info] selected tickers: {tickers}")
    cap_lines = []
    for t in tickers:
        path = generate_chart(t)
        if path:
            # Build daily caption line
            try:
                df = yf.download(t, period="6mo", interval="1d", progress=False, auto_adjust=False, threads=False)
                close = df["Close"].dropna()
                last = float(close.iloc[-1])
                chg30 = pct_change(close, days=30)
                # approximate near support using weekly as proxy
                wk = yf.download(t, period="1y", interval="1wk", progress=False, auto_adjust=False, threads=False)["Close"].dropna()
                sup_low, sup_high = swing_levels(wk, 10)
                near = False
                if sup_low is not None and sup_high is not None:
                    mid = 0.5*(sup_low+sup_high)
                    rng = (sup_high - sup_low) + 1e-8
                    near = abs(last - mid) <= 0.6 * rng
                cap_lines.append(caption_daily(t, last, chg30, near))
            except Exception:
                pass
    if cap_lines:
        try:
            with open(CAPTION_TXT, "w", encoding="utf-8") as f:
                f.write("\n".join(cap_lines))
            log(f"[info] caption file written: {CAPTION_TXT}")
        except Exception as e:
            log(f"[warn] failed to write caption file: {e}")

def run_posters():
    """Every 3 hours workflow equivalent: generate a poster for popular headline(s)."""
    news = fetch_yahoo_headlines(WATCHLIST, max_items=40)
    if not news:
        log("[info] no news items fetched; skipping posters")
        return
    popular = cluster_by_popularity(news)
    rng.shuffle(popular)
    for item in popular[:1]:  # one strong item per run
        tkr = item["ticker"]
        title = item["title"].strip()

        # Headline → ALL CAPS, broken into 1–2 lines max
        # simple split heuristic
        words = title.upper().split()
        if len(words) > 6:
            head_lines = [" ".join(words[:6]), " ".join(words[6:12])]
        else:
            head_lines = [" ".join(words)]

        # Subtext (3–4 lines total when wrapped): we build a richer paragraph
        sub_lines = [
            f"This headline keeps {tkr} in focus as investors weigh the read-through for peers and demand.",
            "The move could shift sentiment across the sector while big funds assess near-term margins.",
            "Analysts will watch guidance and updates on runway into the next print."
        ]

        out = generate_poster(tkr, head_lines, sub_lines)
        if out:
            cap = caption_poster(tkr, title)
            # save companion caption to help the post scheduler
            poster_caption = os.path.splitext(out)[0] + "_caption.txt"
            try:
                with open(poster_caption, "w", encoding="utf-8") as f:
                    f.write(cap)
            except Exception:
                pass
            log(f"[info] poster saved: {out}")
        else:
            log("[warn] poster generation failed")

# =========================
# ---- CLI Entrypoint -----
# =========================
def main():
    """
    Modes:
      --daily      Generate 6 charts + caption file
      --posters    Generate breaking-news poster(s)
      --both       Run both (daily first, then posters)
      --once TKR   Generate a single chart quickly
      --poster-demo TKR "HEADLINE"   Generate one poster locally with default subtext
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily", action="store_true")
    ap.add_argument("--posters", action="store_true")
    ap.add_argument("--both", action="store_true")
    ap.add_argument("--once", type=str, help="single ticker chart")
    ap.add_argument("--poster-demo", nargs="+", help='usage: --poster-demo TKR "HEADLINE..."')
    args = ap.parse_args()

    try:
        if args.daily:
            log("[info] running daily charts")
            run_daily_charts()
        elif args.posters:
            log("[info] running posters")
            run_posters()
        elif args.both:
            log("[info] running daily charts + posters")
            run_daily_charts()
            run_posters()
        elif args.once:
            t = args.once.upper()
            log(f"[info] quick chart for {t}")
            out = generate_chart(t)
            if out:
                log(f"[info] saved: {out}")
            else:
                log("[warn] chart failed")
        elif args.poster_demo:
            if len(args.poster_demo) < 2:
                print("usage: --poster-demo TKR \"HEADLINE...\"")
                return
            tkr = args.poster_demo[0].upper()
            head = " ".join(args.poster_demo[1:]).upper()
            words = head.split()
            if len(words) > 6:
                head_lines = [" ".join(words[:6]), " ".join(words[6:12])]
            else:
                head_lines = [head]
            sub = [
                f"{tkr} stays in the spotlight as investors parse sector read-throughs.",
                "Institutional desks will watch guidance and margin commentary.",
                "Momentum hinges on execution into the next print."
            ]
            out = generate_poster(tkr, head_lines, sub)
            if out:
                cap = caption_poster(tkr, head)
                sc = os.path.splitext(out)[0] + "_caption.txt"
                with open(sc, "w", encoding="utf-8") as f:
                    f.write(cap)
                log(f"[info] poster saved: {out}")
            else:
                log("[warn] poster failed")
        else:
            # default daily behavior (for CI convenience)
            log("[info] default mode → daily charts")
            run_daily_charts()
    except Exception as e:
        log(f"[fatal] {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
