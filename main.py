#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TrendWatchDesk – main.py
- Charts: weekly candlesticks, blue gradient bg, no grid, subtle feathered support zone (rectangle), white logos
- Posters: blue bg, company logo top-right, TWD bottom-right, auto-fit headline, wrapped subtext
- Captions (charts): sector-based emojis, show % (1D/5D/30D), NO price, high variety
- News: Yahoo Finance headlines for posters
- Outputs:
    charts  → output/charts/{TICKER}_chart.png
    caption → output/caption_YYYYMMDD.txt
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
# ---- Paths & Config -----
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

# =========================
# ---- Universe/Pools -----
# =========================
WATCHLIST = [
    # MAG7 / Big Tech
    "AAPL","MSFT","GOOGL","META","AMZN","NVDA","TSLA","NFLX",
    # AI / Semis / Infra
    "AMD","AVGO","TSM","ASML","ARM","SMCI","INTC","MU","TER",
    # Cloud / SaaS / Data / Security
    "CRM","NOW","SNOW","DDOG","MDB","PLTR","PANW","CRWD","ZS",
    # Fintech / Payments / Crypto rails
    "V","MA","PYPL","SQ","COIN","HOOD","SOFI",
    # Energy / Commodities
    "XOM","CVX","SLB","OXY","COP","GLD","SLV","USO","DBC","UNG",
    # Defense / Aerospace
    "LMT","RTX","NOC","GD","BA",
    # Health / Biotech / Medtech
    "LLY","REGN","MRK","PFE","JNJ","ISRG",
    # Retail / E-comm
    "SHOP","COST","WMT",
    # Robotics
    "BOTZ","ROBO","IRBT","FANUY","ISRG","TER",
    # Quantum
    "IONQ","RGTI","QBTS","IBM",
    # Broad ETFs you mentioned before
    "SPY","QQQ","DIA","IWM",
]

POOLS = {
    "AI_Semis":    ["NVDA","AMD","AVGO","TSM","ASML","ARM","SMCI","INTC","MU","TER"],
    "Cloud":       ["MSFT","GOOGL","AMZN","CRM","NOW","SNOW","DDOG","MDB","PLTR"],
    "Security":    ["PANW","CRWD","ZS"],
    "Fintech":     ["V","MA","PYPL","SQ","COIN","HOOD","SOFI"],
    "Energy":      ["XOM","CVX","SLB","OXY","COP","USO","DBC","UNG"],
    "Defense":     ["LMT","RTX","NOC","GD","BA"],
    "Biotech":     ["LLY","REGN","MRK","PFE","JNJ","ISRG"],
    "Retail":      ["SHOP","COST","WMT","AMZN"],
    "BigTech":     ["AAPL","META","NVDA","MSFT","GOOGL","AMZN","NFLX","TSLA"],
    "Commodities": ["GLD","SLV","USO","DBC","UNG"],
    "Robotics":    ["BOTZ","ROBO","IRBT","FANUY","ISRG","TER"],
    "Quantum":     ["IONQ","RGTI","QBTS","IBM"],
    "Wildcards":   ["PLTR","SHOP","SMCI","ARM","SOFI","IONQ","RGTI","QBTS"],
}

COMMODITY_ETFS = {"GLD":"Gold", "SLV":"Silver", "USO":"Crude Oil", "UNG":"Nat Gas", "DBC":"Commodities"}

# =========================
# ---- Yahoo News ----------
# =========================
YF_NEWS_ENDPOINT = "https://query1.finance.yahoo.com/v1/finance/search"
SESS = requests.Session()
SESS.headers.update({"User-Agent": "TrendWatchDesk/1.0", "Accept": "application/json"})

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
    """Strict 1-D float64 with NaNs removed; safe for DF/Series/ndarray/list."""
    if x is None: return np.array([], dtype="float64")
    if isinstance(x, pd.DataFrame):
        num = x.select_dtypes(include=[np.number])
        if num.empty and not x.empty:
            x = x.iloc[:,0]
        else:
            x = num.iloc[:,0] if not num.empty else pd.Series([], dtype="float64")
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
def font_bold(size: int): return _font(FONT_BOLD, size)
def font_reg(size: int):  return _font(FONT_REG,  size)

def recolor_to_white(img: Image.Image) -> Image.Image:
    img = img.convert("RGBA")
    *_, a = img.split()
    white = Image.new("RGBA", img.size, (255,255,255,255))
    white.putalpha(a)
    return white

def load_logo_white(ticker: str, target_w: int) -> Optional[Image.Image]:
    path = os.path.join(LOGO_DIR, f"{ticker}.png")
    if not os.path.exists(path): return None
    img = Image.open(path).convert("RGBA")
    ratio = target_w / max(1, img.width)
    img = img.resize((int(img.width*ratio), int(img.height*ratio)), Image.Resampling.LANCZOS)
    return recolor_to_white(img)

def twd_logo_white(target_w: int) -> Optional[Image.Image]:
    if not os.path.exists(BRAND_LOGO): return None
    img = Image.open(BRAND_LOGO).convert("RGBA")
    ratio = target_w / max(1, img.width)
    img = img.resize((int(img.width*ratio), int(img.height*ratio)), Image.Resampling.LANCZOS)
    return recolor_to_white(img)

def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont):
    bbox = draw.textbbox((0,0), text, font=font)
    return bbox[2]-bbox[0], bbox[3]-bbox[1]

def wrap_to_width(draw, text, font, max_w):
    words=text.split(); lines=[]; line=""
    for w in words:
        test=(line+" "+w).strip()
        tw,_=measure_text(draw,test,font)
        if tw>max_w and line: lines.append(line); line=w
        else: line=test
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
# ---- Sector mapping -----
# =========================
TICKER_SECTOR = {
    "NVDA":"Semis","AMD":"Semis","AVGO":"Semis","TSM":"Semis","ASML":"Semis","ARM":"Semis","SMCI":"Semis","INTC":"Semis","MU":"Semis","TER":"Robotics",
    "MSFT":"Cloud","GOOGL":"Cloud","AMZN":"Cloud","CRM":"Cloud","NOW":"Cloud","SNOW":"Cloud","DDOG":"Cloud","MDB":"Cloud","PLTR":"Cloud",
    "PANW":"Security","CRWD":"Security","ZS":"Security",
    "V":"Fintech","MA":"Fintech","PYPL":"Fintech","SQ":"Fintech","COIN":"Fintech","HOOD":"Fintech","SOFI":"Fintech",
    "XOM":"Energy","CVX":"Energy","SLB":"Energy","OXY":"Energy","COP":"Energy",
    "GLD":"Commodities","SLV":"Commodities","USO":"Commodities","DBC":"Commodities","UNG":"Commodities",
    "LMT":"Defense","RTX":"Defense","NOC":"Defense","GD":"Defense","BA":"Defense",
    "LLY":"Biotech","REGN":"Biotech","MRK":"Biotech","PFE":"Biotech","JNJ":"Biotech","ISRG":"Robotics",
    "SHOP":"Retail","COST":"Retail","WMT":"Retail",
    "AAPL":"BigTech","META":"BigTech","NFLX":"BigTech","TSLA":"Auto",
    "BOTZ":"Robotics","ROBO":"Robotics","IRBT":"Robotics","FANUY":"Robotics",
    "IONQ":"Quantum","RGTI":"Quantum","QBTS":"Quantum","IBM":"Quantum",
    "SPY":"Commodities","QQQ":"BigTech","DIA":"Commodities","IWM":"Commodities",
}

EMO_MAP = {
    "Semis":["🧠","⚡️","🚀"], "Cloud":["☁️","🖥️","🛰️"], "Security":["🛡️","🔒","🕵️"],
    "Fintech":["💳","🏦","📲"], "Energy":["⛽️","🛢️","⚙️"], "Commodities":["🪙","📦","🏭"],
    "Defense":["🛡️","✈️","🛰️"], "Biotech":["🧪","🧬","🩺"], "Retail":["🛍️","🛒","🏷️"],
    "BigTech":["📡","📱","💾"], "Auto":["🚗","🔋","🛠️"], "Robotics":["🤖","🦾","🔧"], "Quantum":["🔮","🧲","🧠"]
}

def sector_for(tkr: str) -> str:
    return TICKER_SECTOR.get(tkr.upper(), "BigTech")

# =========================
# ---- Ticker Selection ---
# =========================
def pick_tickers(n: int = 6) -> List[str]:
    picks = set()
    def grab(pool, k):
        if pool not in POOLS: return
        cands = [t for t in POOLS[pool] if t not in picks]
        rng.shuffle(cands); picks.update(cands[:k])
    for pool, k in (("AI_Semis",2), ("Cloud",1), ("Fintech",1), ("Robotics",1)):
        grab(pool, k)
    others = [t for t in WATCHLIST if t not in picks]
    rng.shuffle(others)
    for t in others:
        if len(picks) >= n: break
        picks.add(t)
    return list(picks)[:n]

# =========================
# ---- Charts -------------
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

        W,H = 1080,720
        margin = 40
        header_h = 140
        footer_h = 60
        x1,y1,x2,y2 = margin+30, margin+header_h, W-margin-30, H-margin-footer_h
        img = chart_background(W,H); d = ImageDraw.Draw(img)

        minp,maxp = float(np.nanmin(l)), float(np.nanmax(h))
        prng = max(1e-8,maxp-minp)
        def y_from(p): return y2 - ((float(p)-minp)/prng)*(y2-y1)

        # Feathered support rectangle
        s = pd.Series(c)
        hi = s.rolling(10).max().iloc[-1]
        lo = s.rolling(10).min().iloc[-1]
        sup_lo = None if pd.isna(lo) else float(lo)
        sup_hi = None if pd.isna(hi) else float(hi)
        if (sup_lo is not None) and (sup_hi is not None):
            y_lo = y_from(sup_hi)
            y_hi = y_from(sup_lo)
            top = min(y_lo, y_hi)
            bot = max(y_lo, y_hi)
            overlay = Image.new("RGBA", img.size, (0,0,0,0))
            od = ImageDraw.Draw(overlay)
            od.rectangle([x1, top, x2, bot],
                         fill=(255,255,255,28),
                         outline=(255,255,255,80), width=2)
            overlay = overlay.filter(ImageFilter.GaussianBlur(1.0))
            img = Image.alpha_composite(img, overlay)
            d = ImageDraw.Draw(img)

        # Candles
        xs = np.linspace(x1, x2, num=len(c))
        bar_w = max(3,int((x2-x1)/len(c)*0.5))
        for i in range(len(c)):
            cx = int(xs[i])
            op,cl,hi_,lo_ = float(o[i]),float(c[i]),float(h[i]),float(l[i])
            col = (60,255,120,255) if cl>=op else (255,80,80,255)
            d.line([(cx,y_from(lo_)), (cx,y_from(hi_))], fill=col, width=2)
            y_op,y_cl = y_from(op),y_from(cl)
            top,bot = min(y_op,y_cl),max(y_op,y_cl)
            d.rectangle([cx-bar_w, top, cx+bar_w, bot], fill=col, outline=col)

        # Logos (white): company TL, TWD BR
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
# ---- Chart Captions -----
# =========================
_used_templates = set()
def _pick_unique(options: List[str]) -> str:
    rng.shuffle(options)
    for o in options:
        if o not in _used_templates:
            _used_templates.add(o)
            return o
    _used_templates.clear()
    return rng.choice(options)

def _pct(a, b):
    try:
        if b == 0: return 0.0
        return (a - b) / b * 100.0
    except Exception:
        return 0.0

def fetch_price_context(ticker: str) -> Dict:
    ctx = {"last": None, "chg1d": None, "chg5d": None, "chg30": None, "atr_pct": None}
    try:
        d = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=False, threads=False)
        if d.empty: return ctx
        close = _to_1d_float_array(d.get("Close"))
        high  = _to_1d_float_array(d.get("High"))
        low   = _to_1d_float_array(d.get("Low"))
        if close.size < 2: return ctx

        last = float(close[-1]); ctx["last"] = last
        ctx["chg1d"]  = _pct(close[-1], close[-2]) if close.size >= 2 else None
        ctx["chg5d"]  = _pct(close[-1], close[-6]) if close.size >= 6 else None
        ctx["chg30"]  = _pct(close[-1], close[-31]) if close.size >= 31 else None

        if close.size >= 16 and high.size >= 16 and low.size >= 16:
            tr = []
            prev = close[-16]
            for i in range(-15, 0):
                tr_i = max(high[i]-low[i], abs(high[i]-prev), abs(low[i]-prev))
                tr.append(tr_i)
                prev = close[i]
            atr = float(np.mean(tr))
            if last:
                ctx["atr_pct"] = (atr / last) * 100.0
    except Exception:
        pass
    return ctx

def _pct_line(chg1d: Optional[float], chg5d: Optional[float], chg30: Optional[float]) -> str:
    bits=[]
    if chg1d is not None:  bits.append(f"{chg1d:+.2f}% 1D")
    if chg5d is not None:  bits.append(f"{chg5d:+.2f}% 5D")
    if chg30 is not None:  bits.append(f"{chg30:+.2f}% 30D")
    return " · ".join(bits) if bits else ""

def caption_daily(ticker: str, last: float, chg30: float, near_support: bool) -> str:
    ctx = fetch_price_context(ticker)
    line = _pct_line(ctx.get("chg1d"), ctx.get("chg5d"), ctx.get("chg30"))
    sec  = sector_for(ticker)
    emo  = random.choice(EMO_MAP.get(sec, ["📈"]))

    cues=[]
    if ctx.get("chg30") is not None and ctx["chg30"] >= 10: cues.append("momentum in play")
    if ctx.get("chg30") is not None and ctx["chg30"] <= -8: cues.append("pressure building")
    if near_support: cues.append("buyers eye support")
    if not cues: cues.append("range shaping up")

    cue = " · ".join(cues)
    templates = [
        f"{emo} {ticker} — {line} • {cue}",
        f"{emo} {ticker} watchlist: {line} • {cue}",
        f"{emo} {ticker} setup check — {line} • {cue}",
        f"{emo} {ticker} on the radar: {line} • {cue}",
        f"{emo} {ticker}: {line} • {cue}",
    ]
    candidates = [t for t in templates if line] or templates
    return _pick_unique(candidates)

# =========================
# ---- Posters ------------
# =========================
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

def build_poster_subtext(ticker: str, headline: str, ctx: Dict) -> List[str]:
    is_commodity = ticker in COMMODITY_ETFS
    asset_name = COMMODITY_ETFS.get(ticker, ticker)
    chg1d  = ctx.get("chg1d"); chg5d = ctx.get("chg5d"); chg30 = ctx.get("chg30"); atrpct = ctx.get("atr_pct")

    facts = []
    if chg1d is not None:  facts.append(f"{chg1d:+.2f}% (1D)")
    if chg5d is not None:  facts.append(f"{chg5d:+.2f}% (5D)")
    if chg30 is not None:  facts.append(f"{chg30:+.2f}% (30D)")
    facts_line = " · ".join(facts) if facts else None

    vol_line = None
    if atrpct is not None:
        if atrpct >= 4:   vol_line = "volatility elevated; range expansion risk"
        elif atrpct <= 1.5: vol_line = "volatility compressed; watch for a break"
        else:             vol_line = "volatility within recent norms"

    title = headline.lower()
    angle = None
    if any(k in title for k in ["upgrade","raises target","price target","beats","surprise","record"]):
        angle = "bullish reaction hinges on follow-through"
    elif any(k in title for k in ["downgrade","miss","probe","lawsuit","guidance cut","halts"]):
        angle = "pressure building as risk is repriced"
    elif any(k in title for k in ["dividend","buyback","split"]):
        angle = "capital return in focus; positioning may recalibrate"
    elif any(k in title for k in ["sec","ftc","doj","antitrust","regulator"]):
        angle = "headline risk in play; watch tape for confirmation"
    elif any(k in title for k in ["demand","orders","bookings","shipments","capacity","pricing"]):
        angle = "fundamentals in focus as pricing power gets tested"

    lines = []
    if is_commodity:
        lead = f"{asset_name} in motion — {facts_line}" if facts_line else f"{asset_name} in motion."
        lines.append(lead)
        lines.append("flows and macro tone steering near-term direction")
        if vol_line: lines.append(vol_line)
    else:
        lead = f"{ticker} — {facts_line}" if facts_line else f"{ticker} stays in focus."
        lines.append(lead)
        if angle: lines.append(angle)
        else:     lines.append("tape sensitivity to headlines remains elevated")
        if vol_line: lines.append(vol_line)

    return lines[:4]

def generate_poster(ticker: str, headline: str, subtext_lines: List[str]) -> Optional[str]:
    try:
        W,H = 1080,1080
        PAD, GAP = 44, 22
        img = poster_background(W,H); d = ImageDraw.Draw(img)

        # Logos white (right side)
        tlogo = load_logo_white(ticker, 180)
        twd   = twd_logo_white(200)

        # NEWS tag (TL)
        tag_font = font_bold(42); tag_text = "NEWS"
        tw_tag, th_tag = measure_text(d, tag_text, tag_font)
        tag_rect = [PAD, PAD, PAD + tw_tag + 28, PAD + th_tag + 20]
        d.rounded_rectangle(tag_rect, radius=12, fill=(0,36,73,210))
        d.text((PAD+14, PAD+10), tag_text, font=tag_font, fill="white")

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

        # Headline (avoid TR logo)
        left = PAD
        right = min(W - PAD, (tlogo_pos[0] - GAP) if tlogo_pos else (W - PAD))
        head_max_w = max(320, right - left)
        h_lines, hfont = fit_headline(d, headline.upper(), FONT_BOLD, 112, head_max_w, 2)

        y = top_used + 10
        for l in h_lines:
            d.text((left, y), l, font=hfont, fill="white")
            _, lh = measure_text(d, l, hfont); y += lh + 8

        # Subtext (wrap; stop before BR TWD)
        sub_font = font_reg(48)
        sub_bottom_limit = H - PAD - bottom_reserved
        sub_y = y + 12
        max_w = right - left
        for para in subtext_lines:
            for l in wrap_to_width(d, para, sub_font, max_w):
                _, lh = measure_text(d, l, sub_font)
                if sub_y + lh > sub_bottom_limit: break
                d.text((left, sub_y), l, font=sub_font, fill=(235,243,255,255))
                sub_y += lh + 10
            sub_y += 8
            if sub_y >= sub_bottom_limit: break

        out = os.path.join(POSTER_DIR, f"{ticker}_poster_{DATESTAMP}.png")
        img.convert("RGB").save(out, "PNG")

        # Poster caption: sector emoji + % facts
        capfile = os.path.splitext(out)[0] + "_caption.txt"
        try:
            ctx = fetch_price_context(ticker)
            bits=[]
            if ctx.get("chg1d") is not None:  bits.append(f"{ctx['chg1d']:+.2f}% 1D")
            if ctx.get("chg5d") is not None:  bits.append(f"{ctx['chg5d']:+.2f}% 5D")
            if ctx.get("chg30") is not None:  bits.append(f"{ctx['chg30']:+.2f}% 30D")
            facts = " · ".join(bits) if bits else ""
            sec = sector_for(ticker); emo = random.choice(EMO_MAP.get(sec, ["📈"]))
            with open(capfile, "w", encoding="utf-8") as f:
                f.write(f"{emo} {ticker} — {facts}\n{headline}\nWatching levels and follow-through.")
        except Exception as e:
            log(f"[warn] could not save poster caption: {e}")
        return out
    except Exception as e:
        log(f"[error] generate_poster({ticker}): {e}")
        return None

# =========================
# ---- News ---------------
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
    # dedupe
    seen=set(); uniq=[]
    for it in items:
        key = re.sub(r"[^a-z0-9 ]+","", it["title"].lower()).strip()
        if key in seen: continue
        seen.add(key); uniq.append(it)
    return uniq[:max_items]

# =========================
# ---- Workflows ----------
# =========================
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
                d1 = yf.download(t, period="6mo", interval="1d",
                                 progress=False, auto_adjust=False, threads=False)
                close = _to_1d_float_array(d1.get("Close"))
                last = float(close[-1]) if close.size else 0.0
                chg30 = 0.0
                if close.size > 31 and close[-31] != 0:
                    chg30 = (last - float(close[-31])) / float(close[-31]) * 100.0

                wk = yf.download(t, period="1y", interval="1wk",
                                 progress=False, auto_adjust=False, threads=False)
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
    generated = []
    if news:
        rng.shuffle(news)
        chosen = news[:2]
        for item in chosen:
            tkr = item["ticker"]; title = item["title"].strip()
            ctx = fetch_price_context(tkr)
            sub_lines = build_poster_subtext(tkr, title, ctx)
            out = generate_poster(tkr, title, sub_lines)
            if out: generated.append(out)
    else:
        tickers = rng.sample(WATCHLIST, 2)
        for t in tickers:
            title = f"{COMMODITY_ETFS.get(t, t)} moves on flows and positioning"
            ctx = fetch_price_context(t)
            sub_lines = build_poster_subtext(t, title, ctx)
            out = generate_poster(t, title, sub_lines)
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
    ap.add_argument("--ci", action="store_true", help="Charts gate: exit 2 if none generated")
    ap.add_argument("--ci-posters", action="store_true", help="Posters gate: exit 2 if none generated")
    ap.add_argument("--daily", action="store_true", help="Generate charts + caption file (no exit gating)")
    ap.add_argument("--posters", action="store_true", help="Generate posters (no exit gating)")
    ap.add_argument("--both", action="store_true", help="Run charts then posters")
    ap.add_argument("--once", type=str, help="Generate a single chart for one ticker (e.g., --once AAPL)")
    args = ap.parse_args()

    try:
        if args.ci:
            c = run_daily_charts()
            raise SystemExit(0 if c>0 else 2)
        elif args.ci_posters:
            c = run_posters()
            raise SystemExit(0 if c>0 else 2)
        elif args.daily:
            run_daily_charts()
        elif args.posters:
            run_posters()
        elif args.both:
            run_daily_charts(); run_posters()
        elif args.once:
            t = args.once.upper()
            p = generate_chart(t)
            print("\n✅ Chart saved:", p if p else "❌ failed (see run.log)", "\n")
        else:
            run_daily_charts()
    except Exception as e:
        log(f"[fatal] {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
