#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TrendWatchDesk - main.py
Stabilized version:
- Safe yfinance extractors
- Candlestick charts, support zones
- Colored logos
- Posters with right-aligned logos
- Captions with varied CTAs (single block at end)
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
ASSETS_DIR   = "assets"
LOGO_DIR     = os.path.join(ASSETS_DIR, "logos")
FONT_DIR     = os.path.join(ASSETS_DIR, "fonts")
BRAND_LOGO   = os.path.join(ASSETS_DIR, "brand_logo.png")
FONT_BOLD    = os.path.join(FONT_DIR, "Grift-Bold.ttf")
FONT_REG     = os.path.join(FONT_DIR, "Grift-Regular.ttf")

OUTPUT_DIR   = "output"
CHART_DIR    = os.path.join(OUTPUT_DIR, "charts")
POSTER_DIR   = os.path.join(OUTPUT_DIR, "posters")
CAPTION_TXT  = os.path.join(OUTPUT_DIR, f"caption_{datetime.date.today().strftime('%Y%m%d')}.txt")
RUN_LOG      = os.path.join(OUTPUT_DIR, "run.log")

for d in (OUTPUT_DIR, CHART_DIR, POSTER_DIR):
    os.makedirs(d, exist_ok=True)

TODAY     = datetime.date.today()
DATESTAMP = TODAY.strftime("%Y%m%d")
SEED      = int(hashlib.sha1(DATESTAMP.encode()).hexdigest(), 16) % (10**8)
rng       = random.Random(SEED)

WATCHLIST = ["AAPL","MSFT","NVDA","AMD","TSLA","AMZN","META","GOOGL","NIO","SHOP","DIS","NFLX","BABA","JPM","SOFI"]

SESS = requests.Session()
SESS.headers.update({"User-Agent": "TrendWatchDesk/1.0","Accept": "application/json"})

SECTOR_EMOJI = {
    "AAPL":"🍏","MSFT":"🧠","NVDA":"🤖","AMD":"🔧","TSLA":"🚗",
    "META":"📡","GOOGL":"🔎","AMZN":"📦","SHOP":"🛍️","NIO":"⚡",
    "DIS":"🎬","NFLX":"📺","BABA":"🐉","JPM":"🏦","SOFI":"💳"
}

# =========================
# ---- Logging ------------
# =========================
def log(msg: str):
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    line = f"[{ts}] {msg}"
    print(line)
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# =========================
# ---- Pandas Safe --------
# =========================
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
            if isinstance(sub, pd.DataFrame): return _series_f64(sub.iloc[:,0])
    return _series_f64(df.iloc[:,0])

def extract_ohlc(df: pd.DataFrame, ticker: Optional[str] = None) -> pd.DataFrame:
    out = {}
    for name in ["Open","High","Low","Close"]:
        ser = None
        if name in df.columns and not isinstance(df[name], pd.DataFrame):
            ser = df[name]
        elif isinstance(df.columns, pd.MultiIndex):
            if ticker and (name,ticker) in df.columns:
                ser = df[(name,ticker)]
            elif name in df.columns:
                sub = df[name]
                if isinstance(sub, pd.Series): ser=sub
                elif isinstance(sub,pd.DataFrame): ser=sub.iloc[:,0]
        if ser is not None:
            out[name] = pd.to_numeric(ser,errors="coerce").astype("float64")
    return pd.DataFrame(out).dropna()

def pct_change(s: pd.Series, days=30) -> float:
    s = _series_f64(s)
    if len(s)<days+1: return 0.0
    prev = float(s.iloc[-days-1]); last = float(s.iloc[-1])
    if prev==0: return 0.0
    return (last-prev)/prev*100.0

# =========================
# ---- Fonts / Logos ------
# =========================
def _font(path,size): 
    try: return ImageFont.truetype(path,size)
    except: return ImageFont.load_default()
def fbold(s): return _font(FONT_BOLD,s)
def freg(s):  return _font(FONT_REG,s)

def load_logo(ticker,tw):
    path=os.path.join(LOGO_DIR,f"{ticker}.png")
    if not os.path.exists(path): return None
    img=Image.open(path).convert("RGBA")
    w,h=img.size; r=tw/max(1,w)
    return img.resize((int(w*r),int(h*r)),Image.Resampling.LANCZOS)

# =========================
# ---- Candlestick Chart --
# =========================
def render_candles(d,ohlc,rect):
    x1,y1,x2,y2=rect; n=len(ohlc)
    if n==0: return
    minp,maxp=float(ohlc["Low"].min()),float(ohlc["High"].max())
    pr=max(1e-9,maxp-minp); xs=np.linspace(x1,x2,n)
    def y(p): return y2-(p-minp)/pr*(y2-y1)
    for i,row in enumerate(ohlc.itertuples()):
        cx=int(xs[i]); o,h,l,c=row.Open,row.High,row.Low,row.Close
        d.line([(cx,int(y(l))),(cx,int(y(h)))],fill="black",width=2)
        top,bottom=y(max(o,c)),y(min(o,c))
        color=(0,200,0,255) if c>=o else (200,0,0,255)
        d.rectangle([cx-4,top,cx+4,bottom],fill=color,outline=color)

def draw_support_zone(d,x1,y1,x2,y2,opacity=40,outline=120):
    d.rectangle([x1,y1,x2,y2],fill=(255,255,255,opacity),outline=(255,255,255,outline),width=2)

def generate_chart(ticker:str)->Optional[str]:
    try:
        df=yf.download(ticker,period="6mo",interval="1d",progress=False)
        ohlc=extract_ohlc(df,ticker)
        if ohlc.empty: return None
        close=ohlc["Close"]; last=float(close.iloc[-1])
        chg30=pct_change(close,30)

        W,H=1080,720
        img=Image.new("RGBA",(W,H),(10,30,80,255))
        d=ImageDraw.Draw(img)
        margin=60;x1,y1,x2,y2=margin,margin,W-margin,H-margin

        render_candles(d,ohlc,(x1,y1,x2,y2))
        sup_lo,sup_hi=(close.rolling(10).min().iloc[-1],close.rolling(10).max().iloc[-1])
        if not pd.isna(sup_lo) and not pd.isna(sup_hi):
            minp,maxp=float(ohlc["Low"].min()),float(ohlc["High"].max())
            pr=max(1e-9,maxp-minp)
            def y(p): return y2-(p-minp)/pr*(y2-y1)
            draw_support_zone(d,x1+6,int(y(sup_hi)),x2-6,int(y(sup_lo)))

        logo=load_logo(ticker,160)
        if logo is not None: img.alpha_composite(logo,(40,40))

        f=freg(34)
        d.text((40,H-60),f"{SECTOR_EMOJI.get(ticker,'📈')} {ticker}  {chg30:+.2f}% (30d)",fill="white",font=f)

        out=os.path.join(CHART_DIR,f"{ticker}_chart.png")
        img.convert("RGB").save(out,"PNG")
        return out
    except Exception as e:
        log(f"[error] generate_chart({ticker}): {e}")
        return None

# =========================
# ---- Captions -----------
# =========================
def caption_line(ticker,last,chg30)->str:
    emj=SECTOR_EMOJI.get(ticker,"📈")
    if chg30>15: cue="strong breakout 🚀"
    elif chg30>5: cue="momentum building 🔥"
    elif chg30<-8: cue="under pressure ⚠️"
    else: cue="traders watching closely 👀"
    return f"{emj} {ticker} — {chg30:+.2f}% (30d), {cue}"

def caption_cta_block()->str:
    opts=[
        "💬 How high can this go?",
        "🤔 Is this the bottom?",
        "📌 Save and come back later",
        "👇 What’s your take?"
    ]
    rng.shuffle(opts)
    return "\n\n"+"\n".join(opts)

# =========================
# ---- Posters ------------
# =========================
def poster_background(W=1080,H=1080):
    base=Image.new("RGB",(W,H),"#0d3a66")
    grad=Image.new("RGB",(W,H))
    for y in range(H):
        t=y/H;r=int(20*t+10);g=int(80*t+58);b=int(120*t+102)
        for x in range(W): grad.putpixel((x,y),(r,g,b))
    return Image.blend(base,grad,0.9).convert("RGBA")

def generate_poster(ticker,title):
    W,H=1080,1080
    img=poster_background(W,H); d=ImageDraw.Draw(img)
    fH=fbold(92); fS=freg(44)
    d.text((50,150),title.upper(),font=fH,fill="white")
    d.text((50,400),f"{ticker} in focus. Sector mood shifting.",font=fS,fill="white")

    logo=load_logo(ticker,220)
    if logo is not None: img.alpha_composite(logo,(W-logo.width-40,40))

    out=os.path.join(POSTER_DIR,f"{ticker}_poster_{DATESTAMP}.png")
    img.convert("RGB").save(out,"PNG"); return out

# =========================
# ---- Workflows ----------
# =========================
def run_daily_charts():
    tickers=rng.sample(WATCHLIST,6)
    log(f"[info] selected tickers: {tickers}")
    cap_lines=[]
    for t in tickers:
        p=generate_chart(t)
        if p:
            df=yf.download(t,period="6mo",interval="1d",progress=False)
            close=extract_close(df,t)
            chg30=pct_change(close,30)
            cap_lines.append(caption_line(t,float(close.iloc[-1]),chg30))
    if cap_lines:
        with open(CAPTION_TXT,"w") as f: f.write("\n".join(cap_lines)+caption_cta_block())
        log(f"[info] caption file written: {CAPTION_TXT}")

def run_posters():
    for t in rng.sample(WATCHLIST,2):
        out=generate_poster(t,"Hot news flows here")
        log(f"[info] poster saved: {out}")

# =========================
# ---- CLI ----------------
# =========================
def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--ci",action="store_true")
    ap.add_argument("--ci-posters",action="store_true")
    args=ap.parse_args()
    if args.ci: run_daily_charts()
    elif args.ci_posters: run_posters()
    else: run_daily_charts()

if __name__=="__main__":
    main()
