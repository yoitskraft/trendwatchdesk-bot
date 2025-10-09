#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TrendWatchDesk - Stable CI Version
Charts + Posters + Captions
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
CAPTION_TXT  = os.path.join(OUTPUT_DIR, f"caption_{datetime.date.today().strftime('%Y%m%d')}.txt")

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
def _to_1d_float_array(x) -> np.ndarray:
    if x is None: return np.array([],dtype="float64")
    if isinstance(x, pd.DataFrame):
        x = x.select_dtypes(include=[np.number]).iloc[:,0] if not x.empty else x.iloc[:,0]
    if isinstance(x, pd.Series): arr = x.to_numpy()
    elif isinstance(x, np.ndarray): arr = x
    else: arr = np.array(x)
    arr = pd.to_numeric(arr, errors="coerce")
    if isinstance(arr, pd.Series): arr = arr.to_numpy()
    arr = np.asarray(arr, dtype="float64").ravel()
    arr = arr[~np.isnan(arr)]
    return arr

def _font(path,size): 
    try: return ImageFont.truetype(path,size)
    except: return ImageFont.load_default()
def font_bold(size): return _font(FONT_BOLD,size)
def font_reg(size):  return _font(FONT_REG,size)

def recolor_to_white(img: Image.Image) -> Image.Image:
    img = img.convert("RGBA")
    r,g,b,a = img.split()
    white = Image.new("RGBA", img.size, (255,255,255,255))
    white.putalpha(a)
    return white

def load_logo_white(ticker:str,w:int)->Optional[Image.Image]:
    p=os.path.join(LOGO_DIR,f"{ticker}.png")
    if not os.path.exists(p): return None
    img=Image.open(p).convert("RGBA")
    ratio=w/max(1,img.width)
    img=img.resize((int(img.width*ratio),int(img.height*ratio)),Image.Resampling.LANCZOS)
    return recolor_to_white(img)

def twd_logo_white(w:int)->Optional[Image.Image]:
    if not os.path.exists(BRAND_LOGO): return None
    img=Image.open(BRAND_LOGO).convert("RGBA")
    ratio=w/max(1,img.width)
    img=img.resize((int(img.width*ratio),int(img.height*ratio)),Image.Resampling.LANCZOS)
    return recolor_to_white(img)

def measure_text(draw,text,font): 
    bbox=draw.textbbox((0,0),text,font=font)
    return bbox[2]-bbox[0], bbox[3]-bbox[1]

def wrap_to_width(draw,text,font,max_w):
    words=text.split(); lines=[]; line=""
    for w in words:
        test=(line+" "+w).strip()
        tw,_=measure_text(draw,test,font)
        if tw>max_w and line: lines.append(line); line=w
        else: line=test
    if line: lines.append(line)
    return lines

def fit_headline(draw,text,font_path,start_size,max_w,max_lines):
    size=start_size
    while size>=56:
        f=ImageFont.truetype(font_path,size)
        lines=wrap_to_width(draw,text,f,max_w)
        if len(lines)<=max_lines: return lines,f
        size-=4
    f=ImageFont.truetype(font_path,56)
    return wrap_to_width(draw,text,f,max_w)[:max_lines],f

# =========================
# ---- Charts -------------
# =========================
def generate_chart(tkr:str)->Optional[str]:
    try:
        df=yf.download(tkr,period="1y",interval="1wk",progress=False,auto_adjust=False,threads=False)
        if df.empty: return None
        arr=_to_1d_float_array(df["Close"])
        if arr.size<2: return None
        last=float(arr[-1]); chg30=(last-float(arr[-31]))/float(arr[-31])*100 if arr.size>31 else 0.0
        s=pd.Series(arr)
        hi,lo=s.rolling(10).max().iloc[-1], s.rolling(10).min().iloc[-1]
        sup_lo=None if pd.isna(lo) else float(lo); sup_hi=None if pd.isna(hi) else float(hi)

        W,H=1080,720
        img=Image.new("RGBA",(W,H),"white"); d=ImageDraw.Draw(img)
        d.text((50,40),tkr,font=font_bold(76),fill="black")
        d.text((50,120),f"{last:,.2f} • {chg30:+.2f}% (30d)",font=font_reg(38),fill="gray")

        xs=np.linspace(100,W-100,num=arr.size)
        minp,maxp=np.nanmin(arr),np.nanmax(arr); pr=max(1e-8,maxp-minp)
        def y(p): return (H-100)-((p-minp)/pr*(H-200))
        pts=[(int(xs[i]),int(y(arr[i]))) for i in range(arr.size)]
        for i in range(1,len(pts)): d.line([pts[i-1],pts[i]],fill="black",width=3)

        if sup_lo and sup_hi:
            d.rectangle([100,int(y(sup_hi)),W-100,int(y(sup_lo))],fill=(40,120,255,48),outline=(40,120,255,160))

        lg=load_logo_white(tkr,140)
        if lg: img.alpha_composite(lg,(W-lg.width-30,30))
        twd=twd_logo_white(160)
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

def generate_poster(tkr:str,headline:str,subtext_lines:List[str])->Optional[str]:
    try:
        W,H=1080,1080; PAD=44; img=poster_background(W,H); d=ImageDraw.Draw(img)
        # NEWS tag
        tag_font=font_bold(42); d.text((PAD,PAD),"NEWS",font=tag_font,fill="white")
        # Logos
        lg=load_logo_white(tkr,220); twd=twd_logo_white(220)
        if lg: img.alpha_composite(lg,(W-lg.width-PAD,PAD))
        if twd: img.alpha_composite(twd,(W-twd.width-PAD,H-twd.height-PAD))
        # Headline
        lines,hfont=fit_headline(d,headline.upper(),FONT_BOLD,112,W-2*PAD,2)
        y=PAD+80
        for l in lines:
            d.text((PAD,y),l,font=hfont,fill="white"); y+=measure_text(d,l,hfont)[1]+10
        # Subtext
        subf=font_reg(48); sub_y=y+20
        for para in subtext_lines:
            for l in wrap_to_width(d,para,subf,W-2*PAD):
                if sub_y+60>H-PAD-220: break
                d.text((PAD,sub_y),l,font=subf,fill=(235,243,255)); sub_y+=55
        out=os.path.join(POSTER_DIR,f"{tkr}_poster_{DATESTAMP}.png")
        img.convert("RGB").save(out,"PNG"); return out
    except Exception as e:
        log(f"[error] generate_poster({tkr}): {e}"); return None

# =========================
# ---- News ---------------
# =========================
def fetch_yahoo_headlines(tickers: List[str], max_items: int = 20) -> List[Dict]:
    items=[]
    for t in tickers:
        try:
            r=SESS.get(YF_NEWS_ENDPOINT,params={"q":t,"newsCount":5},timeout=10)
            if r.status_code!=200: continue
            for n in r.json().get("news",[])[:5]:
                if "title" in n: items.append({"ticker":t,"title":n["title"]})
        except Exception as e: log(f"[warn] fetch fail {t}: {e}")
    seen=set(); uniq=[]
    for it in items:
        if it["title"].lower() not in seen:
            seen.add(it["title"].lower()); uniq.append(it)
    return uniq[:max_items]

# =========================
# ---- Workflows ----------
# =========================
def run_daily_charts():
    tickers=["AAPL","MSFT","NVDA","AMD","TSLA","AMZN"]
    generated=[generate_chart(t) for t in tickers if generate_chart(t)]
    print("==============================")
    if generated: print(f"✅ Charts generated: {len(generated)}")
    else: print("❌ No charts generated")
    print("==============================")
    return len(generated)

def run_posters():
    news=fetch_yahoo_headlines(WATCHLIST)
    if not news: return 0
    count=0
    for item in news[:2]:
        sub=[f"{item['ticker']} stays in focus as investors parse signals.",
             "Analysts highlight sector implications.",
             "Guidance and margins remain key."]
        if generate_poster(item["ticker"],item["title"],sub): count+=1
    print("==============================")
    if count: print(f"✅ Posters generated: {count}")
    else: print("❌ No posters generated")
    print("==============================")
    return count

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
