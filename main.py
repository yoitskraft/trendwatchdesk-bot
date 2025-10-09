#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TrendWatchDesk – main.py
Final version: Weekly candlestick charts (blue brand bg, white logos, no grid, faint support, bottom-left info)
+ IG posters (auto-fit, white logos) + Yahoo headlines. CI safe.

Outputs:
  charts  → output/charts/{TICKER}_chart.png
  caption → output/caption_YYYYMMDD.txt (daily charts)
  posters → output/posters/{TICKER}_poster_YYYYMMDD.png (+ _caption.txt)
"""

import os, re, random, hashlib, traceback, datetime
from typing import List, Dict, Optional

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

WATCHLIST = ["AAPL","MSFT","NVDA","AMD","TSLA","SPY","QQQ","GLD","AMZN","META","GOOGL"]

# Yahoo Finance search
YF_NEWS_ENDPOINT = "https://query1.finance.yahoo.com/v1/finance/search"
SESS = requests.Session()
SESS.headers.update({
    "User-Agent": "TrendWatchDesk/1.0 (+github actions)",
    "Accept": "application/json",
    "Accept-Encoding": "identity"
})

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
# ---- Helpers ------------
# =========================
def _to_1d_float_array(x) -> np.ndarray:
    """Force a 1-D float64 numpy array."""
    if x is None:
        return np.array([], dtype="float64")
    if isinstance(x, pd.DataFrame):
        x = x.select_dtypes(include=[np.number]).iloc[:, 0] if not x.empty else []
    if isinstance(x, pd.Series):
        arr = x.to_numpy()
    else:
        arr = np.array(x)
    arr = pd.to_numeric(arr, errors="coerce")
    if isinstance(arr, pd.Series):
        arr = arr.to_numpy()
    arr = np.asarray(arr, dtype="float64").ravel()
    arr = arr[~np.isnan(arr)]
    return arr

def _font(path: str, size: int):
    try: return ImageFont.truetype(path, size)
    except Exception: return ImageFont.load_default()

def font_bold(size: int): return _font(FONT_BOLD_PATH, size)
def font_reg(size: int):  return _font(FONT_REG_PATH, size)

def load_logo_white(ticker: str, target_w: int) -> Optional[Image.Image]:
    path = os.path.join(LOGO_DIR, f"{ticker}.png")
    if not os.path.exists(path): return None
    try:
        img = Image.open(path).convert("RGBA")
        # Recolor to white
        r,g,b,a = img.split()
        white = Image.new("RGBA", img.size, (255,255,255,255))
        white.putalpha(a)
        w,h = img.size
        ratio = target_w / max(1, w)
        return white.resize((int(w*ratio), int(h*ratio)), Image.Resampling.LANCZOS)
    except Exception:
        return None

def twd_logo_white(target_w: int) -> Optional[Image.Image]:
    if not os.path.exists(BRAND_LOGO): return None
    try:
        logo = Image.open(BRAND_LOGO).convert("RGBA")
        r,g,b,a = logo.split()
        white = Image.new("RGBA", logo.size, (255,255,255,255))
        white.putalpha(a)
        w,h = logo.size
        ratio = target_w / max(1, w)
        return white.resize((int(w*ratio), int(h*ratio)), Image.Resampling.LANCZOS)
    except Exception:
        return None

def chart_background(W=1080, H=720) -> Image.Image:
    base = Image.new("RGB", (W, H), "#0d3a66")
    grad = Image.new("RGB", (W, H))
    for y in range(H):
        t = y / H
        r = int(10 + (20 - 10) * t)
        g = int(58 + (130 - 58) * t)
        b = int(102 + (220 - 102) * t)
        for x in range(W):
            grad.putpixel((x, y), (r, g, b))
    beams = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(beams)
    for i, alpha in enumerate([60, 40, 25]):
        off = i * 120
        d.polygon([(0, 100+off), (W, 0+off), (W, 100+off), (0, 200+off)], fill=(255,255,255,alpha))
    beams = beams.filter(ImageFilter.GaussianBlur(45))
    return Image.alpha_composite(base.convert("RGBA"), beams)

def poster_background(W=1080, H=1080) -> Image.Image:
    return chart_background(W, H)

# =========================
# ---- Chart Generator ----
# =========================
def generate_chart(tkr: str) -> Optional[str]:
    try:
        df = yf.download(tkr, period="1y", interval="1wk",
                         progress=False, auto_adjust=False, threads=False)
        if df.empty:
            log(f"[warn] {tkr}: no data"); return None

        o = _to_1d_float_array(df.get("Open"))
        h = _to_1d_float_array(df.get("High"))
        l = _to_1d_float_array(df.get("Low"))
        c = _to_1d_float_array(df.get("Close"))
        if c.size < 2:
            log(f"[warn] {tkr}: not enough points"); return None

        last = float(c[-1])
        chg30 = 0.0
        if c.size > 31 and c[-31] != 0:
            chg30 = (last - float(c[-31])) / float(c[-31]) * 100.0

        # Canvas
        W,H = 1080,720
        margin = 40
        header_h = 140
        footer_h = 60
        x1,y1,x2,y2 = margin+30, margin+header_h, W-margin-30, H-margin-footer_h
        img = chart_background(W,H); d = ImageDraw.Draw(img)

        # Price range mapping
        minp,maxp = float(np.nanmin(l)), float(np.nanmax(h))
        prng = max(1e-8,maxp-minp)
        def y_from(p): return y2 - ((float(p)-minp)/prng)*(y2-y1)

        # Support zone (very faint)
        s = pd.Series(c)
        hi = s.rolling(10).max().iloc[-1]
        lo = s.rolling(10).min().iloc[-1]
        sup_lo = None if pd.isna(lo) else float(lo)
        sup_hi = None if pd.isna(hi) else float(hi)
        if (sup_lo is not None) and (sup_hi is not None):
            y_lo, y_hi = y_from(sup_hi), y_from(sup_lo)
            d.rectangle([x1+2, min(y_lo,y_hi), x2-2, max(y_lo,y_hi)],
                        fill=(255,255,255,20), outline=(255,255,255,60), width=1)

        # Candlesticks
        xs = np.linspace(x1, x2, num=len(c))
        bar_w = max(3,int((x2-x1)/len(c)*0.5))
        for i in range(len(c)):
            cx = int(xs[i])
            op,cl,hi_,lo_ = float(o[i]),float(c[i]),float(h[i]),float(l[i])
            col = (60,255,120,255) if cl>=op else (255,80,80,255)  # green/red
            d.line([(cx,y_from(lo_)), (cx,y_from(hi_))], fill=col, width=2)  # wick
            y_op,y_cl = y_from(op),y_from(cl)
            top,bot = min(y_op,y_cl),max(y_op,y_cl)
            d.rectangle([cx-bar_w, top, cx+bar_w, bot], fill=col, outline=col)

        # Bottom-left info (white)
        label = f"{last:,.2f}  ({chg30:+.2f}% 30d)"
        f_last = font_reg(36)
        d.text((x1, H - footer_h + 10), label, font=f_last, fill="white")

        # Logos (WHITE) — top-left company, bottom-right TWD
        lg  = load_logo_white(tkr, 140)
        twd = twd_logo_white(160)
        if lg:  img.alpha_composite(lg, (margin+10, 24))
        if twd: img.alpha_composite(twd,(W-twd.width-26,H-twd.height-24))

        out = os.path.join(CHART_DIR, f"{tkr}_chart.png")
        img.convert("RGB").save(out,"PNG")
        return out
    except Exception as e:
        log(f"[error] generate_chart({tkr}): {e}")
        return None

# =========================
# ---- Caption Engine -----
# =========================
def caption_daily(ticker: str, last: float, chg30: float, near_support: bool) -> str:
    cues = []
    if chg30 >= 8: cues.append("momentum building")
    if chg30 <= -8: cues.append("recent pullback on the radar")
    if near_support: cues.append("buyers defending support")
    if not cues: cues = ["range tightening as traders wait for a trigger"]
    return f"{ticker} at {last:,.2f} — {chg30:+.2f}% (30d). {' · '.join(cues)}."

# =========================
# ---- Poster Engine ------
# =========================
def wrap_to_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_w: int) -> list:
    words = text.split(); lines=[]; line=""
    for w in words:
        test = (line+" "+w).strip()
        tw = draw.textbbox((0,0), test, font=font)[2]
        if tw>max_w and line:
            lines.append(line); line=w
        else:
            line=test
    if line: lines.append(line)
    return lines

def fit_headline(draw, text, font_path, start_size, max_w, max_lines):
    size=start_size
    while size>=56:
        f=ImageFont.truetype(font_path, size)
        lines=wrap_to_width(draw, text, f, max_w)
        if len(lines)<=max_lines: return lines, f
        size-=4
    f=ImageFont.truetype(font_path, 56)
    return wrap_to_width(draw, text, f, max_w)[:max_lines], f

def poster_background(W=1080, H=1080) -> Image.Image:
    base = Image.new("RGB", (W, H), "#0d3a66")
    grad = Image.new("RGB", (W, H))
    for y in range(H):
        t = y / H
        r = int(10 + (20 - 10) * t)
        g = int(58 + (130 - 58) * t)
        b = int(102 + (220 - 102) * t)
        for x in range(W):
            grad.putpixel((x, y), (r, g, b))
    beams = Image.new("RGBA", (W, H), (0,0,0,0))
    d = ImageDraw.Draw(beams)
    for i, alpha in enumerate([80, 60, 40]):
        off = i * 140
        d.polygon([(0, 140+off), (W, 0+off), (W, 120+off), (0, 260+off)], fill=(255,255,255,alpha))
    beams = beams.filter(ImageFilter.GaussianBlur(45))
    return Image.alpha_composite(base.convert("RGBA"), beams)

def caption_poster(ticker: str, headline: str) -> str:
    lines = [
        f"{ticker} stays in focus after: {headline}",
        "Investors are weighing sector read-throughs, peer reactions, and positioning.",
        "Watch guidance, margins and the demand outlook into the next print.",
        rng.choice(["What’s your take? Drop a comment.", "Save this for later.", "Share with a friend."])
    ]
    return "\n".join(lines)

def generate_poster(ticker: str, headline: str, subtext_lines: List[str]) -> Optional[str]:
    try:
        W,H = 1080,1080
        PAD, GAP = 44, 22
        img = poster_background(W,H); d = ImageDraw.Draw(img)

        # Logos (white)
        tlogo = load_logo_white(ticker, 180)
        twd   = twd_logo_white(200)

        # NEWS tag
        tag_font=font_bold(42); tag_text="NEWS"
        tw_tag, th_tag = d.textbbox((0,0), tag_text, font=tag_font)[2:]
        tag_rect = [PAD, PAD, PAD + tw_tag + 28, PAD + th_tag + 20]
        d.rounded_rectangle(tag_rect, radius=12, fill=(0,36,73,210))
        d.text((PAD+14, PAD+10), tag_text, font=tag_font, fill="white")

        # Place logos
        right_x = W - PAD
        top_used = PAD + th_tag + 20 + GAP
        bottom_reserved = PAD
        tlogo_pos = None
        if tlogo is not None:
            tlogo_pos = (PAD, PAD)  # top-left to mirror charts
            img.alpha_composite(tlogo, tlogo_pos)
            top_used = max(top_used, PAD + tlogo.height + GAP)
        if twd is not None:
            twd_pos = (right_x - twd.width, H - PAD - twd.height)
            img.alpha_composite(twd, twd_pos)
            bottom_reserved = max(bottom_reserved, twd.height + GAP)

        # Headline (max 2 lines, auto-fit)
        left = PAD
        right = W - PAD
        if tlogo_pos is not None:
            left = PAD + (tlogo.width + GAP)
        head_max_w = right - left
        h_lines, hfont = fit_headline(d, headline.upper(), FONT_BOLD_PATH, 112, head_max_w, 2)

        y = top_used + 14
        for l in h_lines:
            d.text((left, y), l, font=hfont, fill="white")
            _, lh = d.textbbox((0,0), l, font=hfont)[2:]
            y += lh + 8

        # Subtext (wrap; stop before TWD)
        sub_font = font_reg(48)
        sub_bottom_limit = H - PAD - bottom_reserved
        sub_y = y + 14
        for para in subtext_lines:
            for l in wrap_to_width(d, para, sub_font, W - left - PAD):
                _, lh = d.textbbox((0,0), l, font=sub_font)[2:]
                if sub_y + lh > sub_bottom_limit: break
                d.text((left, sub_y), l, font=sub_font, fill=(235,243,255,255))
                sub_y += lh + 10
            sub_y += 10
            if sub_y >= sub_bottom_limit: break

        out = os.path.join(POSTER_DIR, f"{ticker}_poster_{DATESTAMP}.png")
        img.convert("RGB").save(out, "PNG")
        # caption for poster
        capfile = os.path.splitext(out)[0] + "_caption.txt"
        try:
            with open(capfile, "w", encoding="utf-8") as f:
                f.write(caption_poster(ticker, headline))
        except Exception as e:
            log(f"[warn] could not save poster caption: {e}")
        return out
    except Exception as e:
        log(f"[error] generate_poster({ticker}): {e}")
        return None

# =========================
# ---- News (Yahoo) -------
# =========================
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

# =========================
# ---- Workflows ----------
# =========================
def pick_tickers(n: int = 6) -> List[str]:
    # deterministic by seed, balanced pools
    pools = {
        "AI":        ["NVDA","MSFT","GOOGL","META","AMZN"],
        "MAG7":      ["AAPL","MSFT","GOOGL","META","AMZN","NVDA","TSLA"],
        "Semis":     ["NVDA","AMD","AVGO","TSM","INTC","ASML"],
    }
    picks=set()
    def grab(pool,k):
        c=[t for t in pools[pool] if t not in picks]
        rng.shuffle(c); picks.update(c[:k])
    grab("AI",2); grab("MAG7",2); grab("Semis",1)
    others=[t for t in WATCHLIST if t not in picks]
    rng.shuffle(others)
    for t in others:
        if len(picks)>=n: break
        picks.add(t)
    return list(picks)[:n]

def run_daily_charts() -> int:
    tickers = pick_tickers(6)
    log(f"[info] selected tickers: {tickers}")
    generated = []
    cap_lines = []

    for t in tickers:
        path = generate_chart(t)
        if path:
            generated.append(path)
            try:
                df = yf.download(t, period="6mo", interval="1d", progress=False, auto_adjust=False, threads=False)
                close = _to_1d_float_array(df.get("Close"))
                last = float(close[-1]) if close.size else 0.0
                chg30 = 0.0
                if close.size > 31 and close[-31] != 0:
                    chg30 = (last - float(close[-31])) / float(close[-31]) * 100.0
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
    rng.shuffle(news)
    chosen = news[:2]

    generated = []
    for item in chosen:
        tkr = item["ticker"]; title = item["title"].strip()
        sub_lines = [
            f"{tkr} remains in focus as investors parse demand signals and peer read-throughs.",
            "Positioning may shift as large funds gauge margins, guidance and runway.",
            "Watch how this plays into sector sentiment near-term."
        ]
        out = generate_poster(tkr, title, sub_lines)
        if out: generated.append(out)

    print("\n==============================")
    if generated:
        print(f"✅ Posters generated: {len(generated)}")
        for p in generated: print(" -", p)
    else:
        print("❌ No posters generated")
    print("==============================\n")
    return len(generated)

# =========================
# ---- CLI ----------------
# =========================
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily", action="store_true", help="Generate daily charts + caption file")
    ap.add_argument("--posters", action="store_true", help="Generate news-driven posters")
    ap.add_argument("--both", action="store_true", help="Run charts then posters")
    ap.add_argument("--ci", action="store_true", help="CI gate for charts (exit 2 if none)")
    ap.add_argument("--ci-posters", action="store_true", help="CI gate for posters (exit 2 if none)")
    ap.add_argument("--once", type=str, help="One-off chart for a single ticker (e.g., --once AAPL)")
    args = ap.parse_args()

    try:
        if args.daily:
            run_daily_charts()
        elif args.posters:
            run_posters()
        elif args.both:
            run_daily_charts(); run_posters()
        elif args.ci:
            count = run_daily_charts()
            raise SystemExit(0 if count > 0 else 2)
        elif args.ci_posters:
            count = run_posters()
            raise SystemExit(0 if count > 0 else 2)
        elif args.once:
            t = args.once.upper()
            p = generate_chart(t)
            if p: print("\n✅ Chart saved:", p, "\n")
            else: print("\n❌ Chart failed (see run.log)\n")
        else:
            # default to daily charts
            run_daily_charts()
    except Exception as e:
        log(f"[fatal] {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
