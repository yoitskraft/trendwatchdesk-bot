#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TrendWatchDesk – Stable CI Version (charts + posters + captions)
Design + behavior aligned to your earlier main.py (pools, weekly charts, captions).
"""

import os, re, random, hashlib, traceback, datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ========= Config / Paths =========
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
CAPTION_TXT  = os.path.join(OUTPUT_DIR, f"caption_{datetime.date.today().strftime('%Y%m%d')}.txt")

for d in (OUTPUT_DIR, CHART_DIR, POSTER_DIR):
    os.makedirs(d, exist_ok=True)

TODAY     = datetime.date.today()
DATESTAMP = TODAY.strftime("%Y%m%d")
SEED      = int(hashlib.sha1(DATESTAMP.encode()).hexdigest(), 16) % (10**8)
rng       = random.Random(SEED)

# ========= Watchlist & Pools (as before) =========
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

# ========= Yahoo Finance =========
YF_NEWS_ENDPOINT = "https://query1.finance.yahoo.com/v1/finance/search"
SESS = requests.Session()
SESS.headers.update({"User-Agent": "TrendWatchDesk/1.0", "Accept": "application/json"})

# ========= Theme (charts) =========
CLR_BG         = (255,255,255,255)
CLR_GRID       = (0,0,0,28)
CLR_LINE       = (20,20,20,255)
CLR_META       = (35,35,35,255)
CLR_SUPPORT    = (40,120,255,48)
CLR_SUPPORT_O  = (40,120,255,160)
CLR_LAST_DOT   = (20,20,20,255)

# ========= Emojis for chart captions (same style you wanted) =========
SECTOR_EMOJI = {
    "AAPL":"🍏","MSFT":"🧠","NVDA":"🤖","AMD":"🔧","TSLA":"🚗",
    "META":"📡","GOOGL":"🔎","AMZN":"📦","SPY":"📊","QQQ":"📈","GLD":"🪙"
}

# ========= Env knobs (optional) =========
UI_SCALE   = float(os.getenv("TWD_UI_SCALE",   "1.0"))
TEXT_SCALE = float(os.getenv("TWD_TEXT_SCALE", "1.0"))
TLOGO_SCALE= float(os.getenv("TWD_TLOGO_SCALE","1.0"))
DEBUG      = os.getenv("TWD_DEBUG","0") == "1"

# ========= Logging =========
def log(msg: str):
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(RUN_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

# ========= Helpers =========
def _to_1d_float_array(x) -> np.ndarray:
    """Return strict 1-D float64 array with NaNs removed (handles DF/Series/ndarray/list)."""
    if x is None: return np.array([], dtype="float64")
    if isinstance(x, pd.DataFrame):
        num = x.select_dtypes(include=[np.number])
        x = (num.iloc[:,0] if not num.empty else x.iloc[:,0])
    if isinstance(x, pd.Series): arr = x.to_numpy()
    elif isinstance(x, np.ndarray): arr = x
    else: arr = np.array(x)
    arr = pd.to_numeric(arr, errors="coerce")
    if isinstance(arr, pd.Series): arr = arr.to_numpy()
    arr = np.asarray(arr, dtype="float64").ravel()
    arr = arr[~np.isnan(arr)]
    return arr

def _font(path: str, size: int):
    try: return ImageFont.truetype(path, size)
    except Exception: return ImageFont.load_default()

def font_bold(size: int): return _font(FONT_BOLD, int(size*TEXT_SCALE))
def font_reg(size: int):  return _font(FONT_REG,  int(size*TEXT_SCALE))

def recolor_to_white(img: Image.Image) -> Image.Image:
    img = img.convert("RGBA")
    r,g,b,a = img.split()
    white = Image.new("RGBA", img.size, (255,255,255,255))
    white.putalpha(a)
    return white

def load_logo_white(ticker: str, target_w: int) -> Optional[Image.Image]:
    path = os.path.join(LOGO_DIR, f"{ticker}.png")
    if not os.path.exists(path): return None
    img = Image.open(path).convert("RGBA")
    w, h = img.size
    ratio = (target_w*TLOGO_SCALE) / max(1, w)
    img = img.resize((int(w*ratio), int(h*ratio)), Image.Resampling.LANCZOS)
    return recolor_to_white(img)

def twd_logo_white(target_w: int) -> Optional[Image.Image]:
    if not os.path.exists(BRAND_LOGO): return None
    img = Image.open(BRAND_LOGO).convert("RGBA")
    w, h = img.size
    ratio = (target_w*TLOGO_SCALE) / max(1, w)
    img = img.resize((int(w*ratio), int(h*ratio)), Image.Resampling.LANCZOS)
    return recolor_to_white(img)

def measure_text(draw, text, font):
    bbox = draw.textbbox((0,0), text, font=font)
    return bbox[2]-bbox[0], bbox[3]-bbox[1]

def wrap_to_width(draw, text, font, max_w):
    words = text.split()
    lines, line = [], ""
    for w in words:
        test = (line + " " + w).strip()
        tw,_ = measure_text(draw, test, font)
        if tw > max_w and line:
            lines.append(line); line = w
        else:
            line = test
    if line: lines.append(line)
    return lines

def fit_headline(draw, text, font_path, start_size, max_w, max_lines):
    size = start_size
    while size >= 56:
        f = ImageFont.truetype(font_path, int(size*TEXT_SCALE))
        lines = wrap_to_width(draw, text, f, max_w)
        if len(lines) <= max_lines:
            return lines, f
        size -= 4
    f = ImageFont.truetype(font_path, int(56*TEXT_SCALE))
    return wrap_to_width(draw, text, f, max_w)[:max_lines], f

# ========= Ticker selection (same logic) =========
def pick_tickers(n: int = 6) -> List[str]:
    picks = set()
    def grab(pool, k):
        cands = [t for t in POOLS[pool] if t not in picks]
        rng.shuffle(cands)
        picks.update(cands[:k])
    grab("AI", 2)
    grab("MAG7", 2)
    grab("Semis", 1)
    others = [t for k,v in POOLS.items() if k not in ("AI","MAG7","Semis") for t in v]
    rng.shuffle(others)
    for t in others:
        if len(picks) >= n: break
        picks.add(t)
    return list(picks)[:n]

# ========= Charts =========
def generate_chart(tkr: str) -> Optional[str]:
    """
    Weekly chart:
      - Subtle grid + support zone band
      - Big ticker; subheader; price/30d chip (at last price line)
      - White company logo (top-right) and white TWD logo (bottom-right)
    """
    try:
        df = yf.download(tkr, period="1y", interval="1wk", progress=False, auto_adjust=False, threads=False)
        if df.empty:
            log(f"[warn] {tkr}: no data"); return None

        arr = _to_1d_float_array(df.get("Close"))
        if arr.size < 2:
            log(f"[warn] {tkr}: not enough Close points"); return None

        last = float(arr[-1])
        chg30 = 0.0
        if arr.size > 31 and arr[-31] != 0:
            chg30 = (last - float(arr[-31])) / float(arr[-31]) * 100.0

        s = pd.Series(arr)
        hi = s.rolling(10).max().iloc[-1]
        lo = s.rolling(10).min().iloc[-1]
        sup_lo = None if pd.isna(lo) else float(lo)
        sup_hi = None if pd.isna(hi) else float(hi)

        # Canvas
        W,H = int(1080*UI_SCALE), int(720*UI_SCALE)
        margin = int(40*UI_SCALE)
        header_h = int(150*UI_SCALE)
        footer_h = int(72*UI_SCALE)
        x1,y1,x2,y2 = margin+30, margin+header_h, W-margin-30, H-margin-footer_h

        img = Image.new("RGBA", (W,H), CLR_BG); d = ImageDraw.Draw(img)

        # Header
        d.text((margin+30, margin+30), tkr, fill=(0,0,0,255), font=font_bold(76))
        d.text((margin+30, margin+100), "Weekly chart • key support zone", fill=CLR_META, font=font_reg(38))

        # Grid
        for gy in np.linspace(y1, y2, 5):
            d.line([(x1, gy),(x2, gy)], fill=CLR_GRID, width=1)
        for gx in np.linspace(x1, x2, 6):
            d.line([(gx, y1),(gx, y2)], fill=CLR_GRID, width=1)

        # Price mapping
        n = int(arr.size)
        xs = np.linspace(x1, x2, num=n)
        minp,maxp = float(np.nanmin(arr)), float(np.nanmax(arr))
        prng = max(1e-8, maxp-minp)
        def y_from(p): return y2 - ((float(p)-minp)/prng)*(y2-y1)

        # Support zone
        if (sup_lo is not None) and (sup_hi is not None):
            y_lo = y_from(sup_hi); y_hi = y_from(sup_lo)
            d.rectangle([x1+2, min(y_lo,y_hi), x2-2, max(y_lo,y_hi)],
                        fill=CLR_SUPPORT, outline=CLR_SUPPORT_O, width=2)

        # Price line
        pts = [(int(xs[i]), int(y_from(arr[i]))) for i in range(n)]
        for i in range(1, n):
            d.line([pts[i-1], pts[i]], fill=CLR_LINE, width=4)

        # Last price dot & chip (at the last y)
        lx, ly = pts[-1]
        d.ellipse([lx-5, ly-5, lx+5, ly+5], fill=CLR_LAST_DOT)

        label = f"{last:,.2f}  ({chg30:+.2f}% 30d)"
        f_last = font_reg(34)
        tw, th = measure_text(d, label, f_last)
        chip_x1, chip_y1 = x2 - tw - 18, ly - th - 10
        chip_x2, chip_y2 = x2, ly + 10
        d.rounded_rectangle([chip_x1, chip_y1, chip_x2, chip_y2], radius=10,
                            fill=(255,255,255,230), outline=(0,0,0,25))
        d.text((chip_x1+10, ly - th//2), label, font=f_last, fill=(15,15,15,255))

        # Logos (white)
        lg  = load_logo_white(tkr, 140)
        twd = twd_logo_white(160)
        if lg is not None:
            img.alpha_composite(lg, (W - lg.width - 26, 24))
        if twd is not None:
            img.alpha_composite(twd, (W - twd.width - 26, H - twd.height - 24))

        out = os.path.join(CHART_DIR, f"{tkr}_chart.png")
        img.convert("RGB").save(out, "PNG")
        return out

    except Exception as e:
        # extra diagnostics
        try:
            shape = None if df is None else getattr(df.get("Close"), "shape", None)
        except Exception:
            shape = None
        log(f"[error] generate_chart({tkr}): {e} (Close shape={shape})")
        return None

# ========= Chart Captions (as you wanted) =========
def caption_daily(ticker: str, last: float, chg30: float, near_support: bool) -> str:
    emj = SECTOR_EMOJI.get(ticker, "📈")
    cues = []
    if chg30 >= 8: cues.append("momentum building 🔥")
    if chg30 <= -8: cues.append("recent pullback on the radar ⚠️")
    if near_support: cues.append("buyers defending support 🛡️")
    if not cues: cues = ["range tightening as traders wait for a trigger"]
    cta = rng.choice(["Save for later 📌","Your take below 👇","Share with a friend 🔄"])
    return f"{emj} {ticker} at {last:,.2f} — {chg30:+.2f}% (30d). {' · '.join(cues)}. {cta}"

# ========= Posters =========
def poster_background(W=1080,H=1080):
    base=Image.new("RGB",(W,H),"#0d3a66")
    grad=Image.new("RGB",(W,H))
    for y in range(H):
        t=y/H; r=int(10+(20-10)*t); g=int(58+(130-58)*t); b=int(102+(220-102)*t)
        for x in range(W): grad.putpixel((x,y),(r,g,b))
    beams=Image.new("RGBA",(W,H),(0,0,0,0)); d=ImageDraw.Draw(beams)
    for i,a in enumerate([80,60,40]):
        off=i*140; d.polygon([(0,140+off),(W,0+off),(W,120+off),(0,260+off)],fill=(255,255,255,a))
    return Image.alpha_composite(base.convert("RGBA"),beams.filter(ImageFilter.GaussianBlur(45)))

def generate_poster(ticker: str, headline: str, subtext_lines: List[str]) -> Optional[str]:
    """
    IG poster:
      - NEWS tag TL
      - Small white ticker logo TR (no overlap with text)
      - 2-line auto-fit headline (Grift-Bold)
      - 3–4 wrapped subtext lines (Grift-Regular), clipped above TWD logo
      - White TWD logo BR
    """
    try:
        W,H = 1080,1080
        PAD, GAP = 44, 22
        img = poster_background(W,H); d = ImageDraw.Draw(img)

        # White logos
        tlogo = load_logo_white(ticker, 180)   # dialed down from 220 to prevent dominance
        twd   = twd_logo_white(200)

        # NEWS tag
        tag_font = font_bold(42)
        tag_text = "NEWS"
        tw_tag, th_tag = measure_text(d, tag_text, tag_font)
        tag_rect = [PAD, PAD, PAD + tw_tag + 28, PAD + th_tag + 20]
        d.rounded_rectangle(tag_rect, radius=12, fill=(0,36,73,210))
        d.text((PAD+14, PAD+10), tag_text, font=tag_font, fill="white")

        # Place logos
        right_x = W - PAD
        top_used = PAD + th_tag + 20 + GAP
        bottom_reserved = PAD
        tlogo_pos = None
        if tlogo is not None:
            tlogo_pos = (right_x - tlogo.width, PAD)
            img.alpha_composite(tlogo, tlogo_pos)
            top_used = max(top_used, PAD + tlogo.height + GAP)
        if twd is not None:
            twd_pos = (right_x - twd.width, H - PAD - twd.height)
            img.alpha_composite(twd, twd_pos)
            bottom_reserved = max(bottom_reserved, twd.height + GAP)

        # Headline (max 2 lines, auto-fit; left aligned; avoid right logo area)
        left = PAD
        right = W - PAD
        if tlogo_pos is not None:
            right = min(right, tlogo_pos[0] - GAP)
        head_max_w = right - left
        h_lines, hfont = fit_headline(d, headline.upper(), FONT_BOLD, 112, head_max_w, 2)

        y = top_used + 14
        for l in h_lines:
            d.text((left, y), l, font=hfont, fill="white")
            _, lh = measure_text(d, l, hfont)
            y += lh + 8

        # Subtext (wrap; stop before TWD area)
        sub_font = font_reg(48)
        sub_bottom_limit = H - PAD - bottom_reserved
        sub_y = y + 14
        for para in subtext_lines:
            for l in wrap_to_width(d, para, sub_font, W - 2*PAD):
                _, lh = measure_text(d, l, sub_font)
                if sub_y + lh > sub_bottom_limit: break
                d.text((left, sub_y), l, font=sub_font, fill=(235,243,255,255))
                sub_y += lh + 10
            sub_y += 10
            if sub_y >= sub_bottom_limit: break

        out = os.path.join(POSTER_DIR, f"{ticker}_poster_{DATESTAMP}.png")
        img.convert("RGB").save(out, "PNG")
        return out
    except Exception as e:
        log(f"[error] generate_poster({ticker}): {e}")
        return None

# ========= Poster Captions (no emojis; not a copy of poster text) =========
def caption_poster(ticker: str, headline: str) -> str:
    lines = [
        f"{ticker} stays in focus after: {headline}",
        "Investors are weighing sector read-throughs, peer reactions, and positioning.",
        "Watch guidance, margins and the demand outlook through the next print.",
        rng.choice(["What’s your take? Drop a comment.", "Save this for later.", "Share with a friend."])
    ]
    return "\n".join(lines)

# ========= Yahoo News =========
def fetch_yahoo_headlines(tickers: List[str], max_items: int = 40) -> List[Dict]:
    items=[]
    for t in tickers:
        try:
            r = SESS.get(YF_NEWS_ENDPOINT, params={"q": t, "quotesCount": 0, "newsCount": 8}, timeout=10)
            if r.status_code != 200: continue
            for n in r.json().get("news", [])[:8]:
                title = n.get("title")
                if title: items.append({"ticker": t, "title": title})
        except Exception as e:
            log(f"[warn] yahoo fetch {t}: {e}")
    # dedupe by normalized title
    seen=set(); uniq=[]
    for it in items:
        key = re.sub(r"[^a-z0-9 ]+","", it["title"].lower()).strip()
        if key in seen: continue
        seen.add(key); uniq.append(it)
    return uniq[:max_items]

# ========= Workflows =========
def run_daily_charts() -> int:
    tickers = pick_tickers(6)
    log(f"[info] selected tickers: {tickers}")
    generated = []
    cap_lines = []

    for t in tickers:
        p = generate_chart(t)
        if p:
            generated.append(p)
            try:
                d1 = yf.download(t, period="6mo", interval="1d", progress=False, auto_adjust=False, threads=False)
                arr = _to_1d_float_array(d1.get("Close"))
                last = float(arr[-1]) if arr.size else 0.0
                chg30 = 0.0
                if arr.size > 31 and arr[-31] != 0:
                    chg30 = (last - float(arr[-31])) / float(arr[-31]) * 100.0
                wk = yf.download(t, period="1y", interval="1wk", progress=False, auto_adjust=False, threads=False)
                warr = _to_1d_float_array(wk.get("Close"))
                near = False
                if warr.size > 10:
                    s = pd.Series(warr)
                    hi = s.rolling(10).max().iloc[-1]
                    lo = s.rolling(10).min().iloc[-1]
                    if not pd.isna(hi) and not pd.isna(lo):
                        sup_lo, sup_hi = float(lo), float(hi)
                        mid = 0.5*(sup_lo+sup_hi)
                        rngp = max(1e-8, sup_hi - sup_lo)
                        near = abs(last - mid) <= 0.6*rngp
                cap_lines.append(caption_daily(t, last, chg30, near))
            except Exception:
                pass

    # daily caption file
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
        for p in generated: print(" -", p)
    else:
        print("❌ No charts generated")
    print("Caption file:", CAPTION_TXT if cap_lines else "(none)")
    print("==============================\n")
    return len(generated)

def run_posters() -> int:
    news = fetch_yahoo_headlines(WATCHLIST, max_items=40)
    if not news:
        print("\n⚠️ No news found → Poster skipped\n")
        return 0

    # take the first 2 unique for the day (stable by seeded RNG)
    rng.shuffle(news)
    chosen = news[:2]

    generated = []
    for item in chosen:
        tkr = item["ticker"]
        title = item["title"].strip()
        sub_lines = [
            f"{tkr} remains in focus as investors parse demand signals and peer read-throughs.",
            "Positioning may shift as large funds gauge margins, guidance and runway.",
            "Watch how this plays into sector sentiment near-term."
        ]
        out = generate_poster(tkr, title, sub_lines)
        if out:
            generated.append(out)
            # poster caption (no emojis, not a copy of poster)
            captext = caption_poster(tkr, title)
            capfile = os.path.splitext(out)[0] + "_caption.txt"
            try:
                with open(capfile, "w", encoding="utf-8") as f:
                    f.write(captext)
            except Exception as e:
                log(f"[warn] could not save poster caption: {e}")

    print("\n==============================")
    if generated:
        print(f"✅ Posters generated: {len(generated)}")
        for p in generated: print(" -", p)
    else:
        print("❌ No posters generated")
    print("==============================\n")
    return len(generated)

# ========= CLI =========
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ci", action="store_true", help="Charts gate: exit 2 if none generated")
    ap.add_argument("--ci-posters", action="store_true", help="Posters gate: exit 2 if none generated")
    ap.add_argument("--daily", action="store_true", help="Generate charts + caption file (no exit gating)")
    ap.add_argument("--posters", action="store_true", help="Generate posters (no exit gating)")
    ap.add_argument("--both", action="store_true", help="Run charts then posters")
    ap.add_argument("--once", type=str, help="Generate a single chart for one ticker")
    args = ap.parse_args()

    try:
        if args.ci:
            count = run_daily_charts()
            raise SystemExit(0 if count>0 else 2)
        elif args.ci_posters:
            count = run_posters()
            raise SystemExit(0 if count>0 else 2)
        elif args.daily:
            run_daily_charts()
        elif args.posters:
            run_posters()
        elif args.both:
            run_daily_charts()
            run_posters()
        elif args.once:
            t = args.once.upper()
            p = generate_chart(t)
            if p: print("\n✅ Chart saved:", p, "\n")
            else: print("\n❌ Chart failed (see run.log)\n")
        else:
            # default to daily charts to mirror your earlier behavior
            run_daily_charts()
    except Exception as e:
        log(f"[fatal] {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
