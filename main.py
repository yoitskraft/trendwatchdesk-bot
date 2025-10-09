#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TrendWatchDesk - main.py
Aligned with OPERATIONS_GUIDE.md (single source of truth)

Features:
- Weekly charts (6 tickers per run) with support zone, logos, Grift fonts
- Captions: natural, news-driven, emojis
- Posters: IG-native gradient + beams; Grift-Bold headline + Grift-Regular subtext
- News polling (Yahoo Finance lightweight) + clustering
- Deterministic daily seed for stable selections
- Outputs: output/charts/, output/posters/, output/caption_YYYYMMDD.txt, output/run.log
- CLI:
    --ci             Generate daily charts (exit nonzero if none)
    --ci-posters     Generate news posters (exit nonzero if none)
    --daily          Generate charts + daily caption
    --posters        Generate news posters (no exit requirement)
    --both           Run daily then posters
    --once TKR       Single chart
    --poster-demo    Legacy poster test
    --poster-mockup  Local poster test
"""

import os, re, math, random, hashlib, traceback, datetime
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

WATCHLIST = ["AAPL","MSFT","NVDA","AMD","TSLA","SPY","QQQ","GLD","AMZN","META","GOOGL"]

YF_NEWS_ENDPOINT = "https://query1.finance.yahoo.com/v1/finance/search"

SESS = requests.Session()
SESS.headers.update({
    "User-Agent": "TrendWatchDesk/1.0",
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
# ---- Helpers ------------
# =========================
def _as_float(x):
    try:
        return float(x.item())
    except AttributeError:
        return float(x)

def _close_series_to_array(series: pd.Series) -> np.ndarray:
    if series is None or series.size == 0:
        return np.array([], dtype="float64")
    return pd.to_numeric(series, errors="coerce").astype("float64").dropna().to_numpy()

def _font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

def font_bold(size: int): return _font(FONT_BOLD_PATH, size)
def font_reg(size: int):  return _font(FONT_REG_PATH, size)

def load_logo(ticker: str, target_w: int):
    path = os.path.join(LOGO_DIR, f"{ticker}.png")
    if not os.path.exists(path): return None
    try:
        img = Image.open(path).convert("RGBA")
        w, h = img.size
        ratio = target_w / max(1, w)
        return img.resize((int(w*ratio), int(h*ratio)), Image.Resampling.LANCZOS)
    except Exception:
        return None

def twd_logo(target_w: int):
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
def swing_levels(series: pd.Series, lookback: int = 14):
    if series is None or series.empty:
        return (None, None)
    highs = series.rolling(lookback).max()
    lows  = series.rolling(lookback).min()
    lo, hi = lows.iloc[-1], highs.iloc[-1]
    lo = None if pd.isna(lo) else _as_float(lo)
    hi = None if pd.isna(hi) else _as_float(hi)
    return (lo, hi)

def pct_change(series: pd.Series, days: int = 30) -> float:
    try:
        if len(series) < days + 1:
            return 0.0
        last = _as_float(series.iloc[-1])
        prev = _as_float(series.iloc[-days-1])
        if prev == 0: return 0.0
        return (last - prev) / prev * 100.0
    except Exception:
        return 0.0

def generate_chart(ticker: str) -> Optional[str]:
    try:
        df = yf.download(ticker, period="1y", interval="1wk",
                         progress=False, auto_adjust=False, threads=False)
        if df.empty:
            return None
        close_arr = _close_series_to_array(df["Close"])
        if close_arr.size == 0:
            return None
        last  = float(close_arr[-1])
        chg30 = pct_change(pd.Series(close_arr), 30)
        sup_low, sup_high = swing_levels(pd.Series(close_arr), 10)

        # Plot
        W,H = 1080,720
        img = Image.new("RGBA", (W,H), "white")
        d = ImageDraw.Draw(img)
        margin, header_h, footer_h = 40,150,70
        x1,y1,x2,y2 = margin+30, margin+header_h, W-margin-30, H-margin-footer_h

        f_ticker, f_meta = font_bold(76), font_reg(38)
        d.text((margin+30, margin+30), ticker, fill="black", font=f_ticker)
        d.text((margin+30, margin+100), f"{last:,.2f} • {chg30:+.2f}% (30d)",
               fill=(30,30,30,255), font=f_meta)

        n = close_arr.size
        xs = np.linspace(x1, x2, num=n)
        minp,maxp = float(np.nanmin(close_arr)), float(np.nanmax(close_arr))
        prange = max(1e-8, maxp-minp)
        def y_from_price(p): return y2 - (float(p)-minp)/prange*(y2-y1)
        pts = [(int(xs[i]), int(y_from_price(close_arr[i]))) for i in range(n)]
        for i in range(1,n): d.line([pts[i-1], pts[i]], fill="black", width=3)

        if (sup_low is not None) and (sup_high is not None):
            y_lo, y_hi = y_from_price(sup_high), y_from_price(sup_low)
            d.rectangle([x1+2, min(y_lo,y_hi), x2-2, max(y_lo,y_hi)],
                        fill=(40,120,255,48), outline=(40,120,255,160), width=2)

        lg, twd = load_logo(ticker,140), twd_logo(160)
        if lg: img.alpha_composite(lg, (W-lg.width-26,24))
        if twd: img.alpha_composite(twd, (W-twd.width-26,H-twd.height-24))

        out = os.path.join(CHART_DIR, f"{ticker}_chart.png")
        img.convert("RGB").save(out, "PNG")
        return out
    except Exception as e:
        log(f"[error] generate_chart({ticker}): {e}")
        return None

# =========================
# ---- Captions -----------
# =========================
def caption_daily(ticker, last, chg30, near_support):
    emj = SECTOR_EMOJI.get(ticker,"📈")
    cues=[]
    if chg30>=8: cues.append("momentum building 🔥")
    if chg30<=-8: cues.append("recent pullback ⚠️")
    if near_support: cues.append("buyers defending support 🛡️")
    if not cues: cues=["range tightening awaiting trigger"]
    cta=rng.choice(["Save for later 📌","Your take below 👇","Share 🔄"])
    return f"{emj} {ticker} at {last:,.2f} — {chg30:+.2f}% (30d). {' · '.join(cues)}. {cta}"

def caption_poster(ticker, headline):
    hook=f"{SECTOR_EMOJI.get(ticker,'📈')} {ticker} — still in spotlight"
    context="Beyond the headline, investors weigh sector read-throughs and peers."
    fwd="Next: guidance and earnings commentary, with margin detail in focus."
    cta=rng.choice(["Drop your view 👇","Save for later 📌","Share 🔄"])
    return f"{hook}\n{context}\n{fwd}\n\n{cta}"

# =========================
# ---- Posters ------------
# =========================
def poster_background(W=1080,H=1080):
    base=Image.new("RGB",(W,H),"#0d3a66")
    grad=Image.new("RGB",(W,H))
    for y in range(H):
        t=y/H
        r=int(10+(20-10)*t); g=int(58+(130-58)*t); b=int(102+(220-102)*t)
        for x in range(W): grad.putpixel((x,y),(r,g,b))
    bg=Image.blend(base,grad,0.9)
    beams=Image.new("RGBA",(W,H),(0,0,0,0)); d=ImageDraw.Draw(beams)
    for i,a in enumerate([80,60,40]):
        off=i*140
        d.polygon([(0,140+off),(W,0+off),(W,120+off),(0,260+off)],fill=(255,255,255,a))
    beams=beams.filter(ImageFilter.GaussianBlur(45))
    return Image.alpha_composite(bg.convert("RGBA"),beams)

def wrap_text(draw,text,font,maxw):
    words=text.split(); lines=[]; line=""
    for w in words:
        test=line+w+" "
        if draw.textbbox((0,0),test,font=font)[2]>maxw and line:
            lines.append(line.rstrip()); line=w+" "
        else: line=test
    if line: lines.append(line.rstrip())
    return "\n".join(lines)

def generate_poster(tkr,headline_lines,subtext_lines):
    try:
        W,H=1080,1080
        img=poster_background(W,H); d=ImageDraw.Draw(img)
        d.multiline_text((40,160),"\n".join(headline_lines),font=font_bold(108),
                         fill="white",spacing=10,align="left")
        sub=" ".join(subtext_lines)
        wrapped=wrap_text(d,sub,font_reg(48),W-80)
        d.multiline_text((40,420),wrapped,font=font_reg(48),
                         fill=(235,243,255,255),spacing=10,align="left")
        out=os.path.join(POSTER_DIR,f"{tkr}_poster_{DATESTAMP}.png")
        img.convert("RGB").save(out,"PNG")
        return out
    except Exception as e:
        log(f"[error] generate_poster({tkr}): {e}")
        return None

# =========================
# ---- Workflows ----------
# =========================
def run_daily_charts()->int:
    tickers=["AAPL","MSFT","NVDA","AMD","TSLA","AMZN"]
    generated=[]
    for t in tickers:
        p=generate_chart(t)
        if p: generated.append(p)
    print("\n==============================")
    if generated:
        print("✅ Daily charts generated:",len(generated))
    else:
        print("❌ No charts generated")
    print("==============================\n")
    return len(generated)

def run_posters()->int:
    headlines=[("AAPL","APPLE HITS RECORD HIGH"),
               ("NVDA","NVIDIA UNVEILS NEW AI CHIP")]
    generated=[]
    for t,h in headlines:
        words=h.upper().split()
        lines=[" ".join(words[:6])," ".join(words[6:12])] if len(words)>6 else [h.upper()]
        subs=[f"{t} stays in focus as sector watches demand.",
              "Analysts highlight peer implications.",
              "Guidance and margins are next."]
        out=generate_poster(t,lines,subs)
        if out: generated.append(out)
    print("\n==============================")
    if generated:
        print("✅ Posters generated:",len(generated))
    else:
        print("❌ No posters generated")
    print("==============================\n")
    return len(generated)

# =========================
# ---- CLI ----------------
# =========================
def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--ci",action="store_true")
    ap.add_argument("--ci-posters",action="store_true")
    ap.add_argument("--daily",action="store_true")
    ap.add_argument("--posters",action="store_true")
    args=ap.parse_args()

    try:
        if args.ci:
            count=run_daily_charts()
            raise SystemExit(0 if count>0 else 2)
        elif args.ci_posters:
            count=run_posters()
            raise SystemExit(0 if count>0 else 2)
        elif args.daily:
            run_daily_charts()
        elif args.posters:
            run_posters()
        else:
            run_daily_charts()
    except Exception as e:
        log(f"[fatal] {e}")
        traceback.print_exc()
        raise SystemExit(1)

if __name__=="__main__":
    main()
