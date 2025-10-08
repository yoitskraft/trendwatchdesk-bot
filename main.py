#!/usr/bin/env python3
# TrendWatchDesk — stable charts + ONE global news poster (2025-10-08)

import os, random, datetime, traceback, re, hashlib
import numpy as np
import pandas as pd
import yfinance as yf
from PIL import Image, ImageDraw, ImageFont
import requests

# ================== Version / Paths ==================
TWD_VERSION = "2025-10-08"
OUTPUT_DIR  = os.path.abspath("output")
POSTER_DIR  = os.path.join(OUTPUT_DIR, "posters")
TODAY       = datetime.date.today()
DATESTR     = TODAY.strftime("%Y%m%d")
DATESTR_HM  = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M")

# ================== Env knobs ==================
DEFAULT_TF = (os.getenv("TWD_TF", "D") or "D").upper()

FOURH_LOOKBACK_DAYS = int(os.getenv("TWD_4H_LOOKBACK_DAYS", "120"))
SWING_WINDOW        = int(os.getenv("TWD_SWING_WINDOW", "3"))
ATR_LEN             = int(os.getenv("TWD_ATR_LEN", "14"))
ZONE_PCT_TOL        = float(os.getenv("TWD_ZONE_PCT_TOL", "0.004"))  # 0.4%

# UI scaling
TWD_UI_SCALE    = float(os.getenv("TWD_UI_SCALE", "0.90"))
TWD_TEXT_SCALE  = float(os.getenv("TWD_TEXT_SCALE", "0.80"))
TWD_TLOGO_SCALE = float(os.getenv("TWD_TLOGO_SCALE","0.48"))

# Logos
BRAND_LOGO_PATH = os.getenv("BRAND_LOGO_PATH", "assets/brand_logo.png")
BRAND_MAXH      = int(os.getenv("TWD_BRAND_MAXH", "220"))
BRAND_MARGIN    = int(os.getenv("TWD_BRAND_MARGIN", "56"))

# Posters on/off
TWD_POSTER_ON = os.getenv("TWD_BREAKING_ON", "on").lower() in ("on","1","true","yes")

# Optional feed for dlvr.it (we don’t rely on it, branches are primary)
TWD_FEED_ON       = os.getenv("TWD_FEED_ON", "off").lower() in ("on","1","true","yes")
TWD_FEED_TITLE    = os.getenv("TWD_FEED_TITLE", "TrendWatchDesk – Latest")
TWD_FEED_LINKBASE = os.getenv("TWD_FEED_LINKBASE", "").rstrip("/")

# HTTP session (unused externally, but here for parity)
SESS = requests.Session(); SESS.headers.update({"User-Agent":"TWD/1.0"})

# ================== Pools / Tickers ==================
COMPANY_QUERY = {
    "META":"Meta Platforms","AMD":"Advanced Micro Devices","GOOG":"Google Alphabet","GOOGL":"Alphabet",
    "AAPL":"Apple","MSFT":"Microsoft","TSM":"Taiwan Semiconductor","TSLA":"Tesla",
    "JNJ":"Johnson & Johnson","MA":"Mastercard","V":"Visa","NVDA":"NVIDIA",
    "AMZN":"Amazon","SNOW":"Snowflake","SQ":"Block Inc","PYPL":"PayPal","UNH":"UnitedHealth"
}

# Broad symbols to scan for market-wide headlines
GLOBAL_NEWS_SYMBOLS = [
    "^GSPC","^NDX","^DJI","SPY","QQQ","DIA",
    "^TNX","DX-Y.NYB",
    "GLD","SLV","USO","UNG","DBC","GC=F","SI=F","CL=F","NG=F",
    "AAPL","MSFT","AMZN","NVDA","META","GOOGL","TSLA","AMD","TSM",
    "XLF","XLE","XLK","XLI","^VIX"
]

def choose_tickers_somehow():
    # keep your simple stable selector
    rnd = random.Random(DATESTR)
    pool = list(COMPANY_QUERY.keys())
    k = min(6, len(pool))
    return rnd.sample(pool, k)

# ================== Data helpers ==================
def _find_col(df: pd.DataFrame, name: str):
    if df is None or df.empty: return None
    if name in df.columns:
        ser = df[name]
        if isinstance(ser, pd.DataFrame): ser = ser.iloc[:, 0]
        return pd.to_numeric(ser, errors="coerce")
    if isinstance(df.columns, pd.MultiIndex):
        if name in df.columns.get_level_values(-1):
            sub = df.xs(name, axis=1, level=-1)
            ser = sub.iloc[:, 0] if isinstance(sub, pd.DataFrame) else sub
            return pd.to_numeric(ser, errors="coerce")
    try:
        norm = {str(c).lower().replace(" ",""): c for c in df.columns}
        key = name.lower().replace(" ","")
        if key in norm:
            ser = df[norm[key]]
            if isinstance(ser, pd.DataFrame): ser = ser.iloc[:, 0]
            return pd.to_numeric(ser, errors="coerce")
    except Exception:
        pass
    return None

def _get_ohlc_df(df: pd.DataFrame):
    if df is None or df.empty: return None
    o = _find_col(df,"Open"); h = _find_col(df,"High"); l = _find_col(df,"Low")
    c = _find_col(df,"Close")
    if c is None or c.dropna().empty:
        c = _find_col(df,"Adj Close")
    if c is None or c.dropna().empty: return None
    idx = c.index
    def _al(x): return pd.to_numeric(x, errors="coerce").reindex(idx) if x is not None else None
    o, h, l, c = _al(o), _al(h), _al(l), _al(c).astype(float)
    if o is None: o = c.copy()
    if h is None: h = c.copy()
    if l is None: l = c.copy()
    out = pd.DataFrame({"Open":o,"High":h,"Low":l,"Close":c}).dropna()
    return out if not out.empty else None

def atr(df: pd.DataFrame, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    prev = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def swing_points(df: pd.DataFrame, w=3):
    highs, lows = [], []
    h, l = df["High"], df["Low"]
    for i in range(w, len(df)-w):
        if h.iloc[i] >= h.iloc[i-w:i].max() and h.iloc[i] >= h.iloc[i+1:i+1+w].max():
            highs.append((i, float(h.iloc[i])))
        if l.iloc[i] <= l.iloc[i-w:i].min() and l.iloc[i] <= l.iloc[i+1:i+1+w].min():
            lows.append((i, float(l.iloc[i])))
    return highs, lows

def pick_support_level_from_4h(df4h, trend_bullish, w, pct_tol, atr_len):
    if df4h is None or df4h.empty: return (None, None)
    highs, lows = swing_points(df4h, w)
    last_px = float(df4h["Close"].iloc[-1])
    atrv = float(atr(df4h, atr_len).iloc[-1]) if len(df4h) > 1 else 0.0
    tol_abs = max(atrv, last_px * pct_tol)
    if trend_bullish:
        candidates = [(i, v) for (i, v) in highs if v <= last_px]
        if not candidates: return (None, None)
        i_sel, v_sel = sorted(candidates, key=lambda t: (abs(last_px - t[1]), -t[0]))[0]
        level = float(v_sel)
    else:
        if not lows: return (None, None)
        i_sel, v_sel = max(lows, key=lambda t: t[0])
        level = float(v_sel)
    return (level - tol_abs, level + tol_abs)

# ================== Fetcher ==================
def fetch_one(ticker):
    """Return payload: (df_render, last, chg30, sup_low, sup_high, tf_tag, chg1d)"""
    tf = (os.getenv("TWD_TF", DEFAULT_TF) or DEFAULT_TF).upper().strip()

    try:
        df_d = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=True)
    except Exception:
        df_d = None
    if df_d is None or df_d.empty:
        df_d = yf.download(ticker, period="1y", interval="1d", auto_adjust=True)

    ohlc_d = _get_ohlc_df(df_d)
    if ohlc_d is None or ohlc_d.empty: return None

    close_d = ohlc_d["Close"].dropna()
    if close_d.shape[0] < 2: return None

    last = float(close_d.iloc[-1])
    base_val = float(close_d.iloc[-31]) if close_d.shape[0] > 30 else float(close_d.iloc[0])
    chg30 = 100.0 * (last - base_val) / base_val if base_val != 0 else 0.0

    prev = float(close_d.iloc[-2])
    chg1d = 100.0 * (last - prev) / prev if prev != 0 else 0.0

    # 60m → 4H for S/R zone
    try:
        df_60 = yf.Ticker(ticker).history(period="6mo", interval="60m", auto_adjust=True)
    except Exception:
        df_60 = None
    if df_60 is None or df_60.empty:
        df_60 = yf.download(ticker, period="6mo", interval="60m", auto_adjust=True)

    sup_low, sup_high = (None, None)
    if df_60 is not None and not df_60.empty:
        ohlc_60 = _get_ohlc_df(df_60)
        if ohlc_60 is not None and not ohlc_60.empty:
            cutoff = ohlc_60.index.max() - pd.Timedelta(days=FOURH_LOOKBACK_DAYS)
            df_60_clip = ohlc_60.loc[ohlc_60.index >= cutoff].copy()
            if not df_60_clip.empty:
                df_4h = df_60_clip.resample("4H").agg(
                    {"Open":"first","High":"max","Low":"min","Close":"last"}
                ).dropna()
                if not df_4h.empty:
                    trend_bullish = (chg30 > 0)
                    sup_low, sup_high = pick_support_level_from_4h(
                        df_4h, trend_bullish, SWING_WINDOW, ZONE_PCT_TOL, ATR_LEN
                    )

    # render TF
    if tf == "W":
        df_render = ohlc_d.resample("W-FRI").agg(
            {"Open":"first","High":"max","Low":"min","Close":"last"}
        ).dropna().tail(60)
        tf_tag = "W"
    else:
        df_render = ohlc_d.dropna().tail(260)
        tf_tag = "D"

    if df_render is None or df_render.empty: return None
    return (df_render, last, float(chg30), sup_low, sup_high, tf_tag, float(chg1d))

# ================== Renderer (chart) ==================
def render_single_post(out_path, ticker, payload):
    (df, last, chg30, sup_low, sup_high, tf_tag, _chg1d) = payload

    def sp(x: float) -> int: return int(round(x * TWD_UI_SCALE))
    def st(x: float) -> int: return int(round(x * TWD_TEXT_SCALE))

    W, H = 1080, 1080
    BG       = (255,255,255,255)
    TEXT_DK  = (23,23,23,255)
    TEXT_MD  = (55,65,81,255)
    TEXT_LT  = (120,128,140,255)
    GRID_MAJ = (232,236,240,255)
    GRID_MIN = (242,244,247,255)
    GREEN    = (22,163,74,255)
    RED      = (239, 68,68,255)
    WICK     = (140,140,140,255)

    base = Image.new("RGBA", (W, H), BG)
    draw = ImageDraw.Draw(base)

    def _try_font(path, size):
        try: return ImageFont.truetype(path, size)
        except: return None
    def _font(size, bold=False):
        sz = st(size)
        grift_b = _try_font("assets/fonts/Grift-Bold.ttf", sz)
        grift_r = _try_font("assets/fonts/Grift-Regular.ttf", sz)
        robo_b  = _try_font("assets/fonts/Roboto-Bold.ttf", sz) or _try_font("Roboto-Bold.ttf", sz)
        robo_r  = _try_font("assets/fonts/Roboto-Regular.ttf", sz) or _try_font("Roboto-Regular.ttf", sz)
        if bold: return grift_b or robo_b or ImageFont.load_default()
        return grift_r or robo_r or ImageFont.load_default()

    f_ticker = _font(58, True)
    f_price  = _font(30, True)
    f_delta  = _font(28, True)
    f_sub    = _font(22)
    f_axis   = _font(18)

    outer_top = sp(56); outer_lr = sp(64); outer_bot = sp(56)
    header_h = sp(160); footer_h = sp(110)
    cx1, cy1, cx2, cy2 = (
        outer_lr,
        outer_top + header_h,
        W - outer_lr,
        H - outer_bot - footer_h
    )

    # header left (clean equal spacing)
    title_x, title_y = outer_lr, outer_top
    GAP = st(8)
    def draw_line(x, y, text, font, fill):
        draw.text((x, y), text, fill=fill, font=font)
        bbox = draw.textbbox((x, y), text, font=font)
        return y + (bbox[3] - bbox[1]) + GAP

    y_cur = draw_line(title_x, title_y, ticker, f_ticker, TEXT_DK)
    y_cur = draw_line(title_x, y_cur, f"{last:,.2f} USD", f_price, TEXT_MD)
    delta_col = GREEN if chg30 >= 0 else RED
    y_cur = draw_line(title_x, y_cur, f"{chg30:+.2f}% past 30d", f_delta, delta_col)
    sub_label = "Daily chart • last ~1 year" if tf_tag == "D" else "Weekly chart • last 52 weeks"
    _ = draw_line(title_x, y_cur, sub_label, f_sub, TEXT_LT)

    # ticker logo (top-right)
    tlogo_path = os.path.join("assets", "logos", f"{ticker}.png")
    if os.path.exists(tlogo_path):
        try:
            tlogo = Image.open(tlogo_path).convert("RGBA")
            hmax = int(sp(70) * TWD_TLOGO_SCALE)
            hmax = max(40, hmax)
            scl = min(1.0, hmax / max(1, tlogo.height))
            tlogo = tlogo.resize((int(tlogo.width * scl), int(tlogo.height * scl)), Image.LANCZOS)
            base.alpha_composite(tlogo, (W - outer_lr - tlogo.width, title_y))
        except Exception:
            pass

    # data prep
    df2 = df[["Open","High","Low","Close"]].dropna()
    if df2.shape[0] < 2:
        base.convert("RGB").save(out_path, quality=95); return
    if len(df2) > 190:
        df2 = df2.tail(190)

    ymin = float(np.nanmin(df2["Low"])); ymax = float(np.nanmax(df2["High"]))
    if not np.isfinite(ymin) or not np.isfinite(ymax) or abs(ymax - ymin) < 1e-6:
        ymin, ymax = (ymin - 0.5, ymax + 0.5) if np.isfinite(ymin) else (0, 1)
    yr = ymax - ymin; ymin -= 0.02*yr; ymax += 0.02*yr

    def sx(i): return cx1 + (i / max(1, len(df2)-1)) * (cx2 - cx1)
    def sy(v): return cy2 - ((float(v) - ymin) / (ymax - ymin)) * (cy2 - cy1)

    # grid
    grid = Image.new("RGBA", (W, H), (0,0,0,0)); g = ImageDraw.Draw(grid)
    for i in range(1, 7):
        y = cy1 + i * (cy2 - cy1) / 7.0
        g.line([(cx1, y), (cx2, y)], fill=GRID_MIN, width=sp(1))
    for frac in (0.33, 0.66):
        y = cy1 + frac * (cy2 - cy1)
        g.line([(cx1, y), (cx2, y)], fill=GRID_MAJ, width=sp(1))
    for i in range(1, 9):
        x = cx1 + i * (cx2 - cx1) / 9.0
        g.line([(x, cy1), (x, cy2)], fill=GRID_MIN, width=sp(1))
    base = Image.alpha_composite(base, grid); draw = ImageDraw.Draw(base)

    # support zone — single blue box
    if (payload[3] is not None) and (payload[4] is not None):
        sup_low, sup_high = payload[3], payload[4]
        sup_y1, sup_y2 = sy(sup_high), sy(sup_low)
        sup_rect = [cx1, min(sup_y1, sup_y2), cx2, max(sup_y1, sup_y2)]
        sup_layer = Image.new("RGBA", (W, H), (0,0,0,0))
        ImageDraw.Draw(sup_layer).rectangle(sup_rect, fill=(120,162,255,50), outline=(120,162,255,120), width=sp(2))
        base = Image.alpha_composite(base, sup_layer)
        draw = ImageDraw.Draw(base)

    # candles
    n = len(df2)
    body_px = max(1, int(((cx2 - cx1) / max(160, n * 1.05)) * TWD_UI_SCALE))
    half = max(1, body_px // 2)
    for i, row in enumerate(df2.itertuples(index=False)):
        O, Hh, Ll, C = row
        xx = sx(i)
        draw.line([(xx, sy(Hh)), (xx, sy(Ll))], fill=WICK, width=max(1, sp(1)))
        col = (22,163,74,255) if C >= O else (239,68,68,255)
        y1 = sy(max(O, C)); y2 = sy(min(O, C))
        if abs(y2 - y1) < 1: y2 = y1 + 1
        draw.rectangle([xx - half, y1, xx + half, y2], fill=col, outline=None)

    # right axis ticks
    f_axis = _font(18)
    ticks = np.linspace(ymin, ymax, 5)
    for tval in ticks:
        y = sy(tval); label = f"{tval:,.2f}"
        bbox = draw.textbbox((0, 0), label, font=f_axis)
        draw.text((cx2 + sp(8), y - (bbox[3]-bbox[1])/2), label, fill=(140,145,150,255), font=f_axis)

    # footer
    foot_x = outer_lr; foot_y = 1080 - BRAND_MARGIN - st(80)
    f_sub = _font(20)
    draw.text((foot_x, foot_y), "Support zone highlighted", fill=(120,120,120,255), font=f_sub)
    draw.text((foot_x, foot_y + st(22)), "Not financial advice", fill=(160,160,160,255), font=f_sub)

    # brand logo bottom-right
    if BRAND_LOGO_PATH and os.path.exists(BRAND_LOGO_PATH):
        try:
            blogo = Image.open(BRAND_LOGO_PATH).convert("RGBA")
            scale = min(1.0, BRAND_MAXH / max(1, blogo.height))
            new_w = max(1, int(round(blogo.width * scale)))
            new_h = max(1, int(round(blogo.height * scale)))
            blogo = blogo.resize((new_w, new_h), Image.LANCZOS)
            x = 1080 - BRAND_MARGIN - new_w
            y = 1080 - BRAND_MARGIN - new_h
            base.alpha_composite(blogo, (x, y))
        except Exception:
            pass

    base.convert("RGB").save(out_path, quality=95)

# ================== Captions (unchanged tone but less repetitive) ==================
def _emoji_for(ticker, rnd):
    if ticker in ["AAPL","MSFT","META","GOOG","GOOGL","AMZN","TSLA"]:
        pool = ["🖥️","🔎","🧠","💻","📡"]
    elif ticker in ["JNJ","UNH"]:
        pool = ["💊","🧬","⚕️","🩺"]
    elif ticker in ["MA","V","PYPL","SQ"]:
        pool = ["💳","🏦","📈","💸"]
    elif ticker in ["AMD","NVDA","TSM","ASML","QCOM","INTC","MU"]:
        pool = ["🔌","⚡","🔧","🧮"]
    else:
        pool = ["📈","🔎","⚡","🚀"]
    return rnd.choice(pool)

def _compute_recent_changes(df_render: pd.DataFrame, tf_tag: str):
    try:
        c = df_render["Close"].dropna()
        if len(c) < 2: return (0.0, 0.0)
        chg1d = 100.0 * (c.iloc[-1] - c.iloc[-2]) / c.iloc[-2] if c.iloc[-2] else 0.0
        if tf_tag == "D" and len(c) >= 6:
            chg5d = 100.0 * (c.iloc[-1] - c.iloc[-6]) / c.iloc[-6] if c.iloc[-6] else 0.0
        elif tf_tag == "W" and len(c) >= 2:
            chg5d = 100.0 * (c.iloc[-1] - c.iloc[-2]) / c.iloc[-2] if c.iloc[-2] else 0.0
        else:
            chg5d = chg1d
        return (float(chg1d), float(chg5d))
    except Exception:
        return (0.0, 0.0)

def caption_line(ticker, payload, seed=None):
    (df_render, last, chg30, sup_low, sup_high, tf_tag, chg1d_payload) = payload
    rnd = random.Random((seed or DATESTR) + ticker)
    emoji = _emoji_for(ticker, rnd)
    joiner = rnd.choice([" · ", " — "])

    chg1d, chg5d = _compute_recent_changes(df_render, tf_tag)
    if abs(chg1d) >= 2:
        lead = f"{'Up' if chg1d>0 else 'Down'} ~{abs(chg1d):.0f}% {('today' if tf_tag=='D' else 'this week')}"
    else:
        lead = rnd.choice(["steady tape","watchlist update","levels in focus"])

    cues = []
    if chg30 >= 12:
        cues.append(rnd.choice(["momentum looks strong 🔥","breakout pressure building 🚀","uptrend intact ✅"]))
    elif chg30 >= 4:
        cues.append(rnd.choice(["tone constructive 📈","buyers stepping in 🛒","watch pullbacks for bids 👀"]))
    elif chg30 <= -10:
        cues.append(rnd.choice(["recent pullback ⚠️","bearish lean 🐻","mixed tape—respect risk ⚖️"]))
    else:
        cues.append(rnd.choice(["price action steady","range-bound but coiling","neutral—let price confirm 🎯"]))
    if (sup_low is not None) and (sup_high is not None):
        cues.append(rnd.choice(["support zone in play 📍","buyers defended support 🛡️","watch reactions near support 👀"]))
    rnd.shuffle(cues)
    cues = cues[:2]
    cues_txt = "; ".join(cues)
    return f"• {emoji} {ticker} — {lead}{joiner}{cues_txt}"

# ================== News aggregation for ONE poster ==================
PREF_PUBLISHERS = ["reuters","financial times","wall street journal","wsj","bloomberg","yahoo finance","cnbc","marketwatch","barron"]

def _normalize_title_key(t: str) -> str:
    t = (t or "").lower().strip()
    t = re.split(r"[-–—:]\s+", t, maxsplit=1)[0]
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return " ".join(t.split()[:12])

def _collect_yf_news_for_symbols(symbols):
    """Collect yfinance .news across many symbols; newest first; de-dup similar titles."""
    news = []
    for sym in symbols:
        try:
            items = getattr(yf.Ticker(sym), "news", []) or []
            for it in items:
                news.append({
                    "symbol": sym,
                    "title": it.get("title") or "",
                    "publisher": (it.get("publisher") or "").strip(),
                    "published_ts": it.get("providerPublishTime"),
                    "url": it.get("link") or "",
                    "desc": it.get("summary","") if isinstance(it, dict) else ""
                })
        except Exception:
            continue
    # group by normalized title to remove variations
    groups = {}
    for it in news:
        key = _normalize_title_key(it["title"])
        if not key: 
            continue
        groups.setdefault(key, []).append(it)
    # select representative per group (prefer top publisher, then freshest)
    ranked = []
    def score_pub(p):
        p = (p or "").lower()
        for i, name in enumerate(PREF_PUBLISHERS[::-1]):
            if name in p:
                return 100 + i
        return 0
    for key, lst in groups.items():
        rep = sorted(lst, key=lambda x: (score_pub(x.get("publisher")), x.get("published_ts") or 0), reverse=True)[0]
        ranked.append(rep)
    ranked.sort(key=lambda x: (score_pub(x.get("publisher")), x.get("published_ts") or 0), reverse=True)
    return ranked

def _headline_caps_short(title: str) -> str:
    t = (title or "").strip()
    t = re.split(r"[-–—:]\s+", t, maxsplit=1)[0]
    t = re.sub(r"\s+", " ", t)
    if len(t) > 68: t = t[:65] + "…"
    return t.upper()

def _pick_story_background_path(symbol_or_theme: str, story: dict) -> str | None:
    base_dir = os.path.join("assets", "backgrounds", symbol_or_theme)
    if os.path.isdir(base_dir):
        cand = [os.path.join(base_dir, f) for f in os.listdir(base_dir)
                if f.lower().endswith((".jpg",".jpeg",".png"))]
        if cand: return sorted(cand)[0]
    title = (story.get("title") or "").lower()
    theme = None
    if any(k in title for k in ["gold","bullion","xau","gld","gc=f"]): theme = "GOLD"
    elif any(k in title for k in ["oil","crude","wti","brent","cl=f","uso"]): theme = "OIL"
    elif any(k in title for k in ["fed","rate","yields","cpi","inflation"]): theme = "FED"
    elif any(k in title for k in ["earnings","guidance","revenue","profit","results"]): theme = "EARNINGS"
    elif any(k in title for k in ["ai","gpu","chip","semi","fab","nvidia","amd","tsm"]): theme = "CHIPS"
    elif any(k in title for k in ["iphone","android","cloud","google","microsoft","meta"]): theme = "TECH"
    elif any(k in title for k in ["tesla","ev","auto","delivery","factory"]): theme = "AUTO"
    elif any(k in title for k in ["drug","vaccine","health","hospital","pharma"]): theme = "HEALTH"
    elif any(k in title for k in ["visa","mastercard","payment","fintech"]): theme = "PAYMENTS"
    if theme:
        tdir = os.path.join("assets", "backgrounds", theme)
        if os.path.isdir(tdir):
            cand = [os.path.join(tdir, f) for f in os.listdir(tdir)
                    if f.lower().endswith((".jpg",".jpeg",".png"))]
            if cand: return sorted(cand)[0]
    return None

def render_news_poster(out_path, symbol_for_visual, story):
    """Polished square 1080x1080 NEWS poster."""
    W, H = 1080, 1080
    BG       = (255,255,255,255)
    TEXT_DK  = (23,23,23,255)
    TEXT_MD  = (55,65,81,255)
    TEXT_LT  = (120,128,140,255)

    def sp(x: float) -> int: return int(round(x * TWD_UI_SCALE))
    def st(x: float) -> int: return int(round(x * TWD_TEXT_SCALE))

    base = Image.new("RGBA", (W, H), BG)
    draw = ImageDraw.Draw(base)

    # faint background
    bg_path = _pick_story_background_path(symbol_for_visual, story)
    if bg_path and os.path.exists(bg_path):
        try:
            bg_img = Image.open(bg_path).convert("RGBA")
            r = max(W / bg_img.width, H / bg_img.height)
            bg_img = bg_img.resize((int(bg_img.width * r), int(bg_img.height * r)), Image.LANCZOS)
            bx = (bg_img.width - W) // 2
            by = (bg_img.height - H) // 2
            bg_img = bg_img.crop((bx, by, bx + W, by + H))
            white = Image.new("RGBA", (W, H), (255,255,255,255))
            bg_faint = Image.blend(bg_img, white, 0.85)  # ~15%
            base = Image.alpha_composite(base, bg_faint)
        except Exception:
            pass

    # fonts
    def _try_font(path, size):
        try: return ImageFont.truetype(path, size)
        except: return None
    def _font(size, bold=False):
        sz = st(size)
        grift_b = _try_font("assets/fonts/Grift-Bold.ttf", sz)
        grift_r = _try_font("assets/fonts/Grift-Regular.ttf", sz)
        robo_b  = _try_font("assets/fonts/Roboto-Bold.ttf", sz) or _try_font("Roboto-Bold.ttf", sz)
        robo_r  = _try_font("assets/fonts/Roboto-Regular.ttf", sz) or _try_font("Roboto-Regular.ttf", sz)
        if bold: return grift_b or robo_b or ImageFont.load_default()
        return grift_r or robo_r or ImageFont.load_default()

    f_news  = _font(28, True)
    f_head  = _font(56, True)
    f_meta  = _font(26)
    f_para  = _font(30)

    PAD = sp(60)
    x1, y1, x2, y2 = PAD, PAD, W - PAD, H - PAD

    # NEWS label
    draw.text((x1, y1), "NEWS", fill=(90,90,90,255), font=f_news)

    # logo top-right (symbol or theme)
    tlogo_path = os.path.join("assets", "logos", f"{symbol_for_visual}.png")
    if not os.path.exists(tlogo_path):
        title = (story.get("title") or "").lower()
        theme_try = None
        if any(k in title for k in ["gold","bullion","xau","gld","gc=f"]): theme_try = "GOLD"
        elif any(k in title for k in ["oil","wti","brent","cl=f","uso"]): theme_try = "OIL"
        if theme_try:
            alt = os.path.join("assets", "logos", f"{theme_try}.png")
            if os.path.exists(alt): tlogo_path = alt
    if os.path.exists(tlogo_path):
        try:
            tlogo = Image.open(tlogo_path).convert("RGBA")
            hmax = int(sp(90) * TWD_TLOGO_SCALE); hmax = max(50, hmax)
            scl = min(1.0, hmax / max(1, tlogo.height))
            tlogo = tlogo.resize((int(tlogo.width * scl), int(tlogo.height * scl)), Image.LANCZOS)
            base.alpha_composite(tlogo, (x2 - tlogo.width, y1))
        except Exception:
            pass

    raw_title = (story.get("title") or "").strip()
    head_txt  = _headline_caps_short(raw_title)
    head_y = y1 + st(52)

    # wrap headline (≤3 lines)
    def wrap_lines(text, font, max_w, max_lines):
        words = text.split(); lines, cur = [], []
        for w in words:
            cur.append(w); test = " ".join(cur)
            bb = draw.textbbox((0,0), test, font=font)
            if bb[2]-bb[0] > max_w:
                cur.pop(); lines.append(" ".join(cur)); cur=[w]
                if len(lines) == max_lines: break
        if cur and len(lines) < max_lines: lines.append(" ".join(cur))
        return lines

    for ln in wrap_lines(head_txt, f_head, x2 - x1, 3):
