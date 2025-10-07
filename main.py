#!/usr/bin/env python3
import os, random, datetime, traceback
import numpy as np
import pandas as pd
import yfinance as yf
from PIL import Image, ImageDraw, ImageFont
import requests

# ------------------ Config ------------------
OUTPUT_DIR = os.path.abspath("output")
TODAY = datetime.date.today()
DATESTR = TODAY.strftime("%Y%m%d")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "").strip()
BRAND_LOGO_PATH = os.getenv("BRAND_LOGO_PATH", "assets/brand_logo.png")

DEFAULT_TF = (os.getenv("TWD_TF", "D") or "D").upper()

# 4H support detection
FOURH_LOOKBACK_DAYS = int(os.getenv("TWD_4H_LOOKBACK_DAYS", "120"))
SWING_WINDOW = int(os.getenv("TWD_SWING_WINDOW", "3"))
ATR_LEN = int(os.getenv("TWD_ATR_LEN", "14"))
ZONE_PCT_TOL = float(os.getenv("TWD_ZONE_PCT_TOL", "0.004"))

# ------------------ Company pool ------------------
COMPANY_QUERY = {
    "META": "Meta Platforms", "AMD": "Advanced Micro Devices", "GOOG": "Google Alphabet",
    "GOOGL": "Alphabet", "AAPL": "Apple", "MSFT": "Microsoft", "TSM": "Taiwan Semiconductor",
    "TSLA": "Tesla", "JNJ": "Johnson & Johnson", "MA": "Mastercard", "V": "Visa",
    "NVDA": "NVIDIA", "AMZN": "Amazon", "SNOW": "Snowflake", "SQ": "Block Inc",
    "PYPL": "PayPal", "UNH": "UnitedHealth"
}

SESS = requests.Session()
SESS.headers.update({"User-Agent": "TWD/1.0"})

# ------------------ News fetcher ------------------
def news_headline_for(ticker):
    name = COMPANY_QUERY.get(ticker, ticker)
    if NEWSAPI_KEY:
        try:
            r = SESS.get(
                "https://newsapi.org/v2/everything",
                params={"q": f'"{name}" OR {ticker}', "language": "en",
                        "sortBy": "publishedAt", "pageSize": 1},
                headers={"X-Api-Key": NEWSAPI_KEY}, timeout=8
            )
            if r.ok:
                d = r.json()
                if d.get("articles"):
                    title = d["articles"][0].get("title") or ""
                    src = d["articles"][0].get("source", {}).get("name", "")
                    if title:
                        return f"{title} ({src})" if src else title
        except Exception:
            pass
    try:
        items = getattr(yf.Ticker(ticker), "news", []) or []
        if items:
            t = items[0].get("title") or ""
            p = items[0].get("publisher") or ""
            if t:
                return f"{t} ({p})" if p else t
    except Exception:
        pass
    return None

# ------------------ Ticker chooser ------------------
def choose_tickers_somehow():
    pool = list(COMPANY_QUERY.keys())
    rnd = random.Random(DATESTR)
    k = min(6, len(pool))
    return rnd.sample(pool, k)

# ------------------ OHLC helpers ------------------
def _find_col(df: pd.DataFrame, name: str):
    if df is None or df.empty:
        return None
    if name in df.columns:
        ser = df[name]
        if isinstance(ser, pd.DataFrame):
            ser = ser.iloc[:, 0]
        return pd.to_numeric(ser, errors="coerce")
    if isinstance(df.columns, pd.MultiIndex):
        if name in df.columns.get_level_values(-1):
            sub = df.xs(name, axis=1, level=-1)
            ser = sub.iloc[:, 0] if isinstance(sub, pd.DataFrame) else sub
            return pd.to_numeric(ser, errors="coerce")
    return None

def _get_ohlc_df(df: pd.DataFrame):
    if df is None or df.empty: return None
    o = _find_col(df, "Open")
    h = _find_col(df, "High")
    l = _find_col(df, "Low")
    c = _find_col(df, "Close") or _find_col(df, "Adj Close")
    if c is None: return None
    idx = c.index
    def _align(x): return x.reindex(idx) if x is not None else None
    o, h, l = _align(o), _align(h), _align(l)
    out = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c}).dropna()
    return out if not out.empty else None

# ------------------ Technical helpers ------------------
def atr(df: pd.DataFrame, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def swing_points(df: pd.DataFrame, w=3):
    highs, lows = [], []
    h, l = df["High"], df["Low"]
    for i in range(w, len(df)-w):
        if h.iloc[i] >= max(h.iloc[i-w:i]) and h.iloc[i] >= max(h.iloc[i+1:i+1+w]):
            highs.append((i, float(h.iloc[i])))
        if l.iloc[i] <= min(l.iloc[i-w:i]) and l.iloc[i] <= min(l.iloc[i+1:i+1+w]):
            lows.append((i, float(l.iloc[i])))
    return highs, lows

def pick_support_level_from_4h(df4h, trend_bullish, w, pct_tol, atr_len):
    if df4h is None or df4h.empty: return (None, None)
    highs, lows = swing_points(df4h, w)
    last_px = float(df4h["Close"].iloc[-1])
    atrv = float(atr(df4h, atr_len).iloc[-1]) if len(df4h) > 1 else 0.0
    tol_abs = max(atrv, last_px * pct_tol)
    if trend_bullish:
        candidates = [(i, v) for i, v in highs if v <= last_px]
        if not candidates: return (None, None)
        _, level = sorted(candidates, key=lambda t: abs(last_px - t[1]))[0]
    else:
        if not lows: return (None, None)
        _, level = max(lows, key=lambda t: t[0])
    return (level - tol_abs, level + tol_abs)

# ------------------ Data fetcher ------------------
def fetch_one(ticker):
    tf = DEFAULT_TF
    df_d = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=True)
    ohlc_d = _get_ohlc_df(df_d)
    if ohlc_d is None: return None

    last = float(ohlc_d["Close"].iloc[-1])
    base_val = float(ohlc_d["Close"].iloc[-30]) if len(ohlc_d) > 30 else float(ohlc_d["Close"].iloc[0])
    chg30 = 100.0 * (last - base_val) / base_val

    # 4H support
    df_60 = yf.Ticker(ticker).history(period="6mo", interval="60m", auto_adjust=True)
    sup_low, sup_high = None, None
    if not df_60.empty:
        df_4h = df_60.resample("4H").agg({"Open":"first","High":"max","Low":"min","Close":"last"}).dropna()
        trend_bullish = chg30 > 0
        sup_low, sup_high = pick_support_level_from_4h(df_4h, trend_bullish, SWING_WINDOW, ZONE_PCT_TOL, ATR_LEN)

    df_render = ohlc_d.tail(260)
    return (df_render, last, chg30, sup_low, sup_high, tf)

# ------------------ Renderer ------------------
def render_single_post(out_path, ticker, payload):
    (df, last, chg30, sup_low, sup_high, tf_tag) = payload
    W,H=1080,1080
    base=Image.new("RGBA",(W,H),(255,255,255,255))
    draw=ImageDraw.Draw(base)

    def _font(size, bold=False):
        try:
            return ImageFont.truetype("assets/fonts/Roboto-Bold.ttf" if bold else "assets/fonts/Roboto-Regular.ttf", size)
        except: return ImageFont.load_default()

    f_ticker=_font(80,True); f_price=_font(40,True); f_delta=_font(36,True); f_sub=_font(28)

    # Header
    y=50;x=60
    draw.text((x,y),ticker,fill=(0,0,0),font=f_ticker); y+=90
    draw.text((x,y),f"{last:,.2f} USD",fill=(40,40,40),font=f_price); y+=50
    col=(22,163,74) if chg30>=0 else (239,68,68)
    draw.text((x,y),f"{chg30:+.2f}% past 30d",fill=col,font=f_delta); y+=50
    draw.text((x,y),"Daily chart • last ~1y",fill=(120,120,120),font=f_sub)

    # Chart
    chart=[150,200,W-100,H-150]; cx1,cy1,cx2,cy2=chart
    df2=df[["Open","High","Low","Close"]].dropna()
    ymin,ymax=df2["Low"].min(),df2["High"].max()
    def sx(i): return cx1+(i/len(df2))*(cx2-cx1)
    def sy(v): return cy2-((v-ymin)/(ymax-ymin))*(cy2-cy1)

    # Support zone
    if sup_low and sup_high:
        sup_rect=[cx1,sy(sup_high),cx2,sy(sup_low)]
        ImageDraw.Draw(base).rectangle(sup_rect,fill=(120,162,255,50),outline=(120,162,255,120))

    # Candles
    for i,row in enumerate(df2.itertuples(index=False)):
        O,Hh,Ll,C=row
        x=sx(i);col=(22,163,74) if C>=O else (239,68,68)
        draw.line([(x,sy(Hh)),(x,sy(Ll))],fill=(120,120,120),width=1)
        draw.rectangle([x-2,sy(max(O,C)),x+2,sy(min(O,C))],fill=col)

    # Brand logo
    if os.path.exists(BRAND_LOGO_PATH):
        try:
            logo=Image.open(BRAND_LOGO_PATH).convert("RGBA")
            logo=logo.resize((100,100))
            base.alpha_composite(logo,(W-160,H-160))
        except: pass

    out=base.convert("RGB")
    os.makedirs(os.path.dirname(out_path),exist_ok=True)
    out.save(out_path,quality=95)

# ------------------ Captions ------------------
def plain_english_line(ticker, headline, payload, seed=None):
    _,_,chg30,_,_,_=payload
    if not headline: headline="Quiet on the news front."
    cue="bullish trend 🔎" if chg30>0 else "bearish tone 👀" if chg30<0 else "neutral tone 🎯"
    return f"📈 {ticker} — {headline} — {cue}"[:280]

CTA_POOL=["Save for later 📌 · Comment 💬 · Swipe ➡️"]

# ------------------ Main ------------------
def main():
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    tickers=choose_tickers_somehow()
    print("[info] selected tickers:",tickers)
    captions=[]
    for t in tickers:
        try:
            payload=fetch_one(t)
            if not payload: continue
            out_path=os.path.join(OUTPUT_DIR,f"twd_{t}_{DATESTR}.png")
            render_single_post(out_path,t,payload)
            headline=news_headline_for(t)
            captions.append(plain_english_line(t,headline,payload))
        except Exception as e:
            print("Error",t,e);traceback.print_exc()
    if captions:
        capfile=os.path.join(OUTPUT_DIR,f"caption_{DATESTR}.txt")
        with open(capfile,"w") as f:
            f.write("\n\n".join(captions))
            f.write("\n\n"+random.choice(CTA_POOL))

if __name__=="__main__":
    main()
