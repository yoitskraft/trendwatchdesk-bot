#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TrendWatchDesk - main.py
Aligned with OPERATIONS_GUIDE.md (single source of truth)

Features
- Weekly charts (6 tickers per run) with support zone, logos, Grift fonts
- Captions: natural, news-driven, emojis; poster captions complement posters (no duplication)
- Posters: IG-native gradient + beams; Grift-Bold headline + Grift-Regular subtext (3–4 lines), word-wrapped
- News polling (Yahoo Finance lightweight) + simple clustering, safe fallback
- Deterministic daily seed for stable selections
- Outputs: output/charts/, output/posters/, output/caption_YYYYMMDD.txt, output/run.log
- CLI:
    --daily              Generate charts + a daily caption file (like Mon/Wed/Fri workflow)
    --posters            Generate breaking news posters (news-driven only)
    --both               Run daily then posters
    --once TKR           One-off chart for a single ticker
    --poster-demo TKR "HEADLINE..."   Force-generate a poster for testing (kept for parity)
    --poster-mockup [TKR "HEADLINE..."]  New: easy local poster test; defaults to AAPL if args omitted
"""

import os, re, math, random, hashlib, traceback, datetime, json
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

for d in (OUTPUT_DIR, CHART_DIR, POSTER_DIR):
    os.makedirs(d, exist_ok=True)

TODAY      = datetime.date.today()
DATESTAMP  = TODAY.strftime("%Y%m%d")
SEED       = int(hashlib.sha1(DATESTAMP.encode()).hexdigest(), 16) % (10**8)
rng        = random.Random(SEED)

# Watchlist per Ops Guide
WATCHLIST = ["AAPL","MSFT","NVDA","AMD","TSLA","SPY","QQQ","GLD","AMZN","META","GOOGL"]

POOLS = {
    "AI":        ["NVDA","MSFT","GOOGL","META","AMZN"],
    "MAG7":      ["AAPL","MSFT","GOOGL","META","AMZN","NVDA","TSLA"],
    "Semis":     ["NVDA","AMD","AVGO","TSM","INTC","ASML"],
    "Healthcare":["UNH","JNJ","PFE","MRK","LLY"],
    "Fintech":   ["MA","V","PYPL","SQ"],
    "Quantum":   ["IONQ","IBM","AMZN"],
    "Wildcards": ["NFLX","DIS","BABA","NIO","SHOP","PLTR"]
}

# Lightweight Yahoo Finance search (subject to change upstream)
YF_NEWS_ENDPOINT = "https://query1.finance.yahoo.com/v1/finance/search"

SESS = requests.Session()
SESS.headers.update({
    "User-Agent": "TrendWatchDesk/1.0 (+github actions)",
    "Accept": "application/json",
    "Accept-Encoding": "identity"
})

SECTOR_EMOJI = {
    "AAPL":"🍏","MSFT":"🧠","NVDA":"🤖","AMD":"🔧","TSLA":"🚗",
    "META":"📡","GOOGL":"🔎","AMZN":"📦","SPY":"📊","QQQ":"📈","GLD":"🪙"
}

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
# ---- Fonts / Logos ------
# =========================
def _font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

def font_bold(size: int): return _font(FONT_BOLD_PATH, size)
def font_reg(size: int):  return _font(FONT_REG_PATH, size)

def load_logo(ticker: str, target_w: int) -> Optional[Image.Image]:
    path = os.path.join(LOGO_DIR, f"{ticker}.png")
    if not os.path.exists(path): return None
    try:
        img = Image.open(path).convert("RGBA")
        w, h = img.size
        ratio = target_w / max(1, w)
        return img.resize((int(w*ratio), int(h*ratio)), Image.Resampling.LANCZOS)
    except Exception:
        return None

def twd_logo(target_w: int) -> Optional[Image.Image]:
    if not os.path.exists(BRAND_LOGO): return None
    try:
        logo = Image.open(BRAND_LOGO).convert("RGBA")
        r,g,b,a = logo.split()
        white = Image.new("RGBA", logo.size, (255,255,255,0)); white.putalpha(a)
        w,h = white.size
        ratio = target_w / max(1, w)
        return white.resize((int(w*ratio), int(h*ratio)), Image.Resampling.LANCZOS)
    except Exception:
        return None

# =========================
# ---- Chart Generator ----
# =========================
def pick_tickers(n: int = 6) -> List[str]:
    picks = set()
    def grab(pool, k):
        cands = [t for t in POOLS[pool] if t not in picks]
        rng.shuffle(cands)
        picks.update(cands[:k])
    grab("AI", 2)
    grab("MAG7", 2)
    grab("Semis", 1)
    # fill remaining from others
    others = [t for k,v in POOLS.items() if k not in ("AI","MAG7","Semis") for t in v]
    rng.shuffle(others)
    for t in others:
        if len(picks) >= n: break
        picks.add(t)
    return list(picks)[:n]

def swing_levels(series: pd.Series, lookback: int = 14) -> Tuple[Optional[float], Optional[float]]:
    if series is None or series.empty: return (None, None)
    highs = series.rolling(lookback).max()
    lows  = series.rolling(lookback).min()
    lo = lows.iloc[-1]; hi = highs.iloc[-1]
    return (float(lo) if not math.isnan(lo) else None,
            float(hi) if not math.isnan(hi) else None)

def pct_change(series: pd.Series, days: int = 30) -> float:
    try:
        if len(series) < days+1: return 0.0
        return float((series.iloc[-1] - series.iloc[-days-1]) / series.iloc[-days-1] * 100.0)
    except Exception:
        return 0.0

def generate_chart(ticker: str) -> Optional[str]:
    """Weekly ‘clean’ chart with support zone, logos, and text."""
    try:
        df = yf.download(ticker, period="1y", interval="1wk", progress=False, auto_adjust=False, threads=False)
        if df.empty:
            log(f"[warn] No data for {ticker}")
            return None
        close = df["Close"].dropna()
        last  = float(close.iloc[-1])
        chg30 = pct_change(close, 30)
        sup_low, sup_high = swing_levels(close, lookback=10)

        # Canvas
        W,H = 1080, 720
        img = Image.new("RGBA", (W,H), (255,255,255,255))
        d   = ImageDraw.Draw(img)

        # Regions
        margin = 40
        header_h = 150
        footer_h = 70
        x1,y1,x2,y2 = margin+30, margin+header_h, W-margin-30, H-margin-footer_h

        # Titles
        f_ticker = font_bold(76)
        f_meta   = font_reg(38)
        d.text((margin+30, margin+30), ticker, fill=(0,0,0,255), font=f_ticker)
        d.text((margin+30, margin+100), f"{last:,.2f} • {chg30:+.2f}% (30d)", fill=(30,30,30,255), font=f_meta)

        # Plot
        xs = np.linspace(x1, x2, num=len(close))
        minp, maxp = float(close.min()), float(close.max())
        prange = max(1e-8, maxp - minp)
        def y_from_price(p): return y2 - (p - minp) / prange * (y2 - y1)

        pts = [(int(xs[i]), int(y_from_price(close.iloc[i]))) for i in range(len(close))]
        for i in range(1, len(pts)):
            d.line([pts[i-1], pts[i]], fill=(0,0,0,255), width=3)

        # Subtle grid
        for gy in np.linspace(y1, y2, 5):
            d.line([(x1, gy),(x2, gy)], fill=(0,0,0,30), width=1)

        # Support zone (translucent)
        if sup_low is not None and sup_high is not None:
            y_lo = y_from_price(sup_high)
            y_hi = y_from_price(sup_low)
            band = [x1+2, min(y_lo, y_hi), x2-2, max(y_lo, y_hi)]
            d.rectangle(band, fill=(40,120,255,48), outline=(40,120,255,160), width=2)

        # Logos
        lg = load_logo(ticker, 140)
        if lg: img.alpha_composite(lg, (W - lg.width - 26, 24))
        twd = twd_logo(160)
        if twd: img.alpha_composite(twd, (W - twd.width - 26, H - twd.height - 24))

        # Subheading
        f_sub = font_reg(32)
        d.text((x1, y1-56), "Weekly chart • key support zone", fill=(40,40,40,255), font=f_sub)

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
    emj = SECTOR_EMOJI.get(ticker, "📈")
    cues = []
    if chg30 >= 8: cues.append("momentum building 🔥")
    if chg30 <= -8: cues.append("recent pullback on the radar ⚠️")
    if near_support: cues.append("buyers defending support 🛡️")
    if not cues:
        cues = ["range tightening as traders wait for a trigger"]

    cta = rng.choice(["Save for later 📌","Your take below 👇","Share with a friend 🔄"])
    return f"{emj} {ticker} at {last:,.2f} — {chg30:+.2f}% (30d). {' · '.join(cues)}. {cta}"

def caption_poster(ticker: str, poster_headline: str) -> str:
    """Poster captions MUST complement the poster text (no duplication)."""
    hook = f"{SECTOR_EMOJI.get(ticker, '📈')} {ticker} — still in the spotlight"
    context = ("Beyond the headline, investors are weighing sector read-throughs, "
               "peer reactions, and what this means for near-term demand.")
    fwd = "Next up: guidance and earnings commentary — the market wants detail on margins and runway."
    cta = rng.choice(["What’s your take? Drop a comment 👇", "Save this for later 📌", "Share with a friend 🔄"])
    return f"{hook}\n{context}\n{fwd}\n\n{cta}"

# =========================
# ---- Poster Engine ------
# =========================
def poster_background(W=1080, H=1080) -> Image.Image:
    """Blue gradient + diagonal beams for IG-native look."""
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

def draw_news_tag(draw: ImageDraw.ImageDraw, x=40, y=40):
    f = font_bold(42); pad = 14
    tw, th = draw.textbbox((0,0), "NEWS", font=f)[2:]
    rect = (x, y, x + tw + pad*2, y + th + pad*2)
    draw.rounded_rectangle(rect, 12, fill=(0,36,73,200))
    draw.text((x+pad, y+pad), "NEWS", font=f, fill=(255,255,255,255))

def generate_poster(ticker: str, headline_lines: List[str], subtext_lines: List[str]) -> Optional[str]:
    """IG-native poster with left-aligned headline + extended subtext (wrapped), logos in correct corners."""
    try:
        W,H = 1080,1080
        img = poster_background(W,H)
        d   = ImageDraw.Draw(img)

        draw_news_tag(d)

        # Headline
        d.multiline_text((40,160), "\n".join(headline_lines), font=font_bold(108),
                         fill=(255,255,255,255), spacing=10, align="left")

        # Subtext (3–4 short lines when wrapped)
        sub = " ".join(subtext_lines)
        sub_wrapped = wrap_text_by_width(d, sub, font_reg(48), W-80)
        d.multiline_text((40,420), sub_wrapped, font=font_reg(48),
                         fill=(235,243,255,255), spacing=10, align="left")

        # Ticker logo (top-right) — fallback rounded badge
        logo = load_logo(ticker, 220)
        if logo is not None:
            img.alpha_composite(logo, (W - logo.width - 40, 40))
        else:
            badge_w, badge_h = 220, 110
            badge = Image.new("RGBA", (badge_w, badge_h), (255,255,255,35))
            bd = ImageDraw.Draw(badge)
            bd.rounded_rectangle([0,0,badge_w, badge_h], 20, outline=(255,255,255,120),
                                 width=2, fill=(255,255,255,28))
            bf = font_bold(64)
            tw = bd.textbbox((0,0), ticker, font=bf)[2]
            bd.text(((badge_w-tw)//2, (badge_h-64)//2-2), ticker, font=bf, fill="white")
            img.alpha_composite(badge, (W - badge_w - 40, 40))

        # TrendWatchDesk logo (bottom-right), tinted white + subtle glow
        twd = twd_logo(220)
        if twd:
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
def fetch_yahoo_headlines(tickers: List[str], max_items: int = 40) -> List[Dict]:
    """Best-effort Yahoo Finance news pull. If it fails, return empty (no posters)."""
    items = []
    for t in tickers:
        try:
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
    seen = set(); uniq = []
    for it in items:
        key = it["title"].strip().lower()
        if key in seen: continue
        seen.add(key); uniq.append(it)
    return uniq[:max_items]

def cluster_by_popularity(items: List[Dict]) -> List[Dict]:
    """Keep items that recur by normalized title, else top few."""
    if not items: return []
    counts = {}
    def norm(s: str) -> str: return re.sub(r"[^a-z0-9 ]+","", s.lower()).strip()
    for it in items:
        k = norm(it["title"])
        counts[k] = counts.get(k, 0) + 1
    pops = [it for it in items if counts.get(norm(it["title"]), 0) >= 2]
    if not pops:
        pops = items[:6]
    return pops

# =========================
# ---- Workflows ----------
# =========================
def run_daily_charts():
    """Generate 6 charts + daily captions file (Mon/Wed/Fri in CI)."""
    tickers = pick_tickers(6)
    log(f"[info] selected tickers: {tickers}")
    cap_lines = []
    for t in tickers:
        path = generate_chart(t)
        if path:
            try:
                df = yf.download(t, period="6mo", interval="1d", progress=False, auto_adjust=False, threads=False)
                close = df["Close"].dropna()
                last = float(close.iloc[-1]) if not close.empty else 0.0
                chg30 = pct_change(close, 30)
                # approx near-support check using weekly
                wk = yf.download(t, period="1y", interval="1wk", progress=False, auto_adjust=False, threads=False)["Close"].dropna()
                sup_low, sup_high = swing_levels(wk, 10)
                near = False
                if sup_low is not None and sup_high is not None and last:
                    mid = 0.5*(sup_low+sup_high)
                    rngp = max(1e-8, (sup_high - sup_low))
                    near = abs(last - mid) <= 0.6 * rngp
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
    """Generate news-driven poster(s). If no news, skip (no artificial fallbacks)."""
    news = fetch_yahoo_headlines(WATCHLIST, max_items=40)
    if not news:
        log("[info] No news found → Poster skipped")
        return
    popular = cluster_by_popularity(news)
    rng.shuffle(popular)
    # One good poster per run
    for item in popular[:1]:
        tkr = item["ticker"]
        title = item["title"].strip()
        # Headline to 1–2 lines
        words = title.upper().split()
        headline_lines = [" ".join(words[:6]), " ".join(words[6:12])] if len(words) > 6 else [" ".join(words)]
        # Extended subtext (richer paragraph → wrapped later)
        sub_lines = [
            f"This headline keeps {tkr} in focus as investors weigh read-through for peers and demand.",
            "The move could shift sentiment across the sector while big funds assess near-term margins.",
            "Analysts will watch guidance and runway into the next print."
        ]
        out = generate_poster(tkr, headline_lines, sub_lines)
        if out:
            cap = caption_poster(tkr, title)
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
# ---- Demos / Mockups ----
# =========================
def run_poster_demo(args: List[str]):
    """Legacy demo: --poster-demo TKR "HEADLINE..." (kept for parity)."""
    if not args or len(args) < 2:
        print('usage: --poster-demo TKR "HEADLINE..."')
        return
    tkr = args[0].upper()
    head = " ".join(args[1:]).upper()
    words = head.split()
    headline_lines = [" ".join(words[:6]), " ".join(words[6:12])] if len(words) > 6 else [head]
    sub = [
        f"{tkr} stays in the spotlight as investors parse sector read-throughs.",
        "Institutional desks will watch guidance and margin commentary.",
        "Momentum hinges on execution into the next print."
    ]
    out = generate_poster(tkr, headline_lines, sub)
    if out:
        cap = caption_poster(tkr, head)
        sc = os.path.splitext(out)[0] + "_caption.txt"
        with open(sc, "w", encoding="utf-8") as f:
            f.write(cap)
        log(f"[info] poster saved: {out}")
    else:
        log("[warn] poster failed")

def run_poster_mockup(args: Optional[List[str]] = None):
    """
    New: --poster-mockup [TKR "HEADLINE..."]
    If no args → default AAPL demo.
    """
    if args and len(args) >= 2:
        tkr = args[0].upper()
        headline = " ".join(args[1:])
    else:
        tkr = "AAPL"
        headline = "APPLE HITS RECORD HIGH"

    words = headline.upper().split()
    headline_lines = [" ".join(words[:6]), " ".join(words[6:12])] if len(words) > 6 else [headline.upper()]

    sub_lines = [
        f"{tkr} stays in focus as investors parse demand signals across the sector.",
        "Analysts highlight broader implications for peers and market sentiment.",
        "Guidance and margin commentary will be key going forward."
    ]

    out = generate_poster(tkr, headline_lines, sub_lines)
    if out:
        cap = caption_poster(tkr, headline)
        capfile = os.path.splitext(out)[0] + "_caption.txt"
        with open(capfile, "w", encoding="utf-8") as f:
            f.write(cap)
        log(f"[mockup] Poster saved: {out}")
    else:
        log("[mockup] Poster generation failed")

# =========================
# ---- CLI ----------------
# =========================
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily", action="store_true", help="Generate 6 charts + daily caption file")
    ap.add_argument("--posters", action="store_true", help="Generate news-driven posters")
    ap.add_argument("--both", action="store_true", help="Run daily then posters")
    ap.add_argument("--once", type=str, help="Generate a single chart for one ticker")
    ap.add_argument("--poster-demo", nargs="*", help='Legacy: --poster-demo TKR "HEADLINE..."')
    ap.add_argument("--poster-mockup", nargs="*", help='New: --poster-mockup [TKR "HEADLINE..."]')
    args = ap.parse_args()

    try:
        if args.daily:
            log("[info] running daily charts")
            run_daily_charts()
        elif args.posters:
            log("[info] running posters (news-driven)")
            run_posters()
        elif args.both:
            log("[info] running daily charts + posters")
            run_daily_charts()
            run_posters()
        elif args.once:
            t = args.once.upper()
            log(f"[info] quick chart for {t}")
            out = generate_chart(t)
            if out: log(f"[info] saved: {out}")
            else:   log("[warn] chart failed")
        elif args.poster_demo:
            run_poster_demo(args.poster_demo)
        elif args.poster_mockup is not None:
            run_poster_mockup(args.poster_mockup)
        else:
            # default to daily for convenience in local testing
            log("[info] default mode → daily charts")
            run_daily_charts()
    except Exception as e:
        log(f"[fatal] {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
