#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TrendWatchDesk - main.py
Stable CI version: Charts + Posters + Yahoo Finance news
"""

import os, re, random, hashlib, traceback, datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# =========================
# ---- Configs ------------
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
RUN_LOG      = os.path.join(OUTPUT_DIR, "run.log")

for d in (OUTPUT_DIR, CHART_DIR, POSTER_DIR):
    os.makedirs(d, exist_ok=True)

TODAY     = datetime.date.today()
DATESTAMP = TODAY.strftime("%Y%m%d")
SEED      = int(hashlib.sha1(DATESTAMP.encode()).hexdigest(), 16) % (10**8)
rng       = random.Random(SEED)

WATCHLIST = ["AAPL","MSFT","NVDA","AMD","TSLA","AMZN","META","GOOGL"]

YF_NEWS_ENDPOINT = "https://query1.finance.yahoo.com/v1/finance/search"
SESS = requests.Session()
SESS.headers.update({"User-Agent": "TrendWatchDesk/1.0"})

SECTOR_EMOJI = {
    "AAPL":"🍏","MSFT":"🧠","NVDA":"🤖","AMD":"🔧","TSLA":"🚗",
    "META":"📡","GOOGL":"🔎","AMZN":"📦"
}

# =========================
# ---- Logging ------------
# =========================
def log(msg: str):
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    line=f"[{ts}] {msg}"
    print(line)
    try:
        with open(RUN_LOG,"a",encoding="utf-8") as f: f.write(line+"\n")
    except: pass

# =========================
# ---- Helpers ------------
# =========================
def _as_float(x):
    try: return float(x.item())
    except AttributeError: return float(x)

def _close_series_to_array(series: pd.Series) -> np.ndarray:
    if series is None or series.size==0: return np.array([],dtype="float64")
    arr=pd.to_numeric(series,errors="coerce").dropna().to_numpy(dtype="float64")
    return arr.reshape(-1)

def _font(path,size):
    try: return ImageFont.truetype(path,size)
    except: return ImageFont.load_default()
def font_bold(size): return _font(FONT_BOLD,size)
def font_reg(size):  return _font(FONT_REG,size)

def load_logo(ticker:str,w:int)->Optional[Image.Image]:
    p=os.path.join(LOGO_DIR,f"{ticker}.png")
    if not os.path.exists(p): return None
    img=Image.open(p).convert("RGBA")
    ratio=w/max(1,img.width)
    return img.resize((int(img.width*ratio),int(img.height*ratio)),Image.Resampling.LANCZOS)

def twd_logo(w:int)->Optional[Image.Image]:
    if not os.path.exists(BRAND_LOGO): return None
    logo=Image.open(BRAND_LOGO).convert("RGBA")
    r,g,b,a=logo.split()
    base=Image.new("RGBA",logo.size,(255,255,255,0)); base.putalpha(a)
    ratio=w/max(1,logo.width)
    return base.resize((int(logo.width*ratio),int(logo.height*ratio)),Image.Resampling.LANCZOS)

# =========================
# ---- Charts -------------
# =========================
def swing_levels(series: pd.Series, lookback=14):
    if series is None or series.empty: return (None,None)
    hi,lo=series.rolling(lookback).max().iloc[-1], series.rolling(lookback).min().iloc[-1]
    return (None if pd.isna(lo) else _as_float(lo),
            None if pd.isna(hi) else _as_float(hi))

def pct_change(series: pd.Series, days=30)->float:
    if len(series)<days+1: return 0.0
    last,prev=_as_float(series.iloc[-1]),_as_float(series.iloc[-days-1])
    return 0.0 if prev==0 else (last-prev)/prev*100.0

def generate_chart(tkr:str)->Optional[str]:
    try:
        df=yf.download(tkr,period="1y",interval="1wk",progress=False,auto_adjust=False,threads=False)
        if df.empty: return None
        arr=_close_series_to_array(df["Close"])
        if arr.size==0: return None
        last=float(arr[-1]); chg30=pct_change(pd.Series(arr),30)
        sup_lo,sup_hi=swing_levels(pd.Series(arr),10)

        W,H=1080,720
        img=Image.new("RGBA",(W,H),"white"); d=ImageDraw.Draw(img)
        d.text((50,40),tkr,font=font_bold(76),fill="black")
        d.text((50,120),f"{last:,.2f} • {chg30:+.2f}% (30d)",font=font_reg(38),fill="gray")

        xs=np.linspace(100,W-100,num=arr.size)
        minp,maxp=np.nanmin(arr),np.nanmax(arr); pr=max(1e-8,maxp-minp)
        def y(p): return (H-100)-(p-minp)/pr*(H-200)
        pts=[(int(xs[i]),int(y(arr[i]))) for i in range(arr.size)]
        for i in range(1,len(pts)): d.line([pts[i-1],pts[i]],fill="black",width=3)

        if sup_lo and sup_hi:
            y_lo,y_hi=y(sup_hi),y(sup_lo)
            d.rectangle([100,y_lo,W-100,y_hi],fill=(40,120,255,48),outline=(40,120,255,160))

        lg=load_logo(tkr,140)
        if lg: img.alpha_composite(lg,(W-lg.width-30,30))
        twd=twd_logo(160)
        if twd: img.alpha_composite(twd,(W-twd.width-30,H-twd.height-30))

        out=os.path.join(CHART_DIR,f"{tkr}_chart.png")
        img.convert("RGB").save(out,"PNG")
        return out
    except Exception as e:
        log(f"[error] generate_chart({tkr}): {e}")
        return None

# =========================
# ---- Posters ------------
# =========================
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

def wrap_text(draw,text,font,max_w):
    words=text.split(); lines=[]; line=""
    for w in words:
        test=f"{line}{w} "; tw=draw.textbbox((0,0),test,font=font)[2]
        if tw>max_w and line: lines.append(line.rstrip()); line=w+" "
        else: line=test
    if line: lines.append(line.rstrip())
    return "\n".join(lines)

def generate_poster(tkr:str,headline:str,sub_lines:List[str])->Optional[str]:
    try:
        W,H=1080,1080; img=poster_background(W,H); d=ImageDraw.Draw(img)
        d.text((40,40),"NEWS",font=font_bold(42),fill="white")
        d.multiline_text((40,160),wrap_text(d,headline.upper(),font_bold(108),W-80),
                         font=font_bold(108),fill="white",spacing=10)
        sub="\n".join([wrap_text(d,s,font_reg(48),W-80) for s in sub_lines])
        d.multiline_text((40,420),sub,font=font_reg(48),fill=(235,243,255),spacing=10)
        lg=load_logo(tkr,220)
        if lg: img.alpha_composite(lg,(W-lg.width-40,40))
        twd=twd_logo(220)
        if twd: img.alpha_composite(twd,(W-twd.width-40,H-twd.height-40))
        out=os.path.join(POSTER_DIR,f"{tkr}_poster_{DATESTAMP}.png")
        img.convert("RGB").save(out,"PNG")
        return out
    except Exception as e:
        log(f"[error] generate_poster({tkr}): {e}")
        return None

# =========================
# ---- Yahoo News ---------
# =========================
def fetch_yahoo_headlines(tickers: List[str], max_items=20) -> List[Dict]:
    items=[]
    for t in tickers:
        try:
            r=SESS.get(YF_NEWS_ENDPOINT,params={"q":t,"newsCount":10},timeout=10)
            if r.status_code!=200: continue
            for n in r.json().get("news",[])[:10]:
                if n.get("title"):
                    items.append({"ticker":t,"title":n["title"]})
        except Exception as e:
            log(f"[warn] yahoo fetch {t}: {e}")
    seen=set(); uniq=[]
    for it in items:
        k=it["title"].strip().lower()
        if k in seen: continue
        seen.add(k); uniq.append(it)
    return uniq[:max_items]

# =========================
# ---- Workflows ----------
# =========================
def run_daily_charts()->int:
    tickers=["AAPL","MSFT","NVDA","AMD","TSLA","AMZN"]
    generated=[generate_chart(t) for t in tickers if generate_chart(t)]
    if generated: print("==============================\n✅ Charts generated:",len(generated),"\n==============================")
    else: print("==============================\n❌ No charts generated\n==============================")
    return len(generated)

def run_posters()->int:
    news=fetch_yahoo_headlines(WATCHLIST,20)
    if not news: return 0
    generated=[]
    for item in news[:2]:
        tkr=item["ticker"]; title=item["title"]
        sub=[f"{tkr} stays in focus as investors parse demand signals.",
             "Analysts highlight sector implications.",
             "Guidance and margins remain key."]
        p=generate_poster(tkr,title,sub)
        if p: generated.append(p)
    if generated: print("==============================\n✅ Posters generated:",len(generated),"\n==============================")
    else: print("==============================\n❌ No posters generated\n==============================")
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
            c=run_daily_charts(); raise SystemExit(0 if c>0 else 2)
        elif args.ci_posters:
            c=run_posters(); raise SystemExit(0 if c>0 else 2)
        elif args.daily: run_daily_charts()
        elif args.posters: run_posters()
        else: run_daily_charts()
    except Exception as e:
        log(f"[fatal] {e}"); traceback.print_exc()

if __name__=="__main__":
    main()
