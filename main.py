#!/usr/bin/env python3
# TrendWatchDesk — charts + event-driven Stories (+ optional posters)
# Version: 2025-10-08-evtstories

import os, re, math, json, random, datetime, traceback, hashlib
import numpy as np
import pandas as pd
import yfinance as yf
from PIL import Image, ImageDraw, ImageFont

# ================== Paths / Version ==================
TWD_VERSION = "2025-10-08-evtstories"

OUTPUT_DIR   = os.path.abspath("output")
CHART_DIR    = OUTPUT_DIR  # charts go in output/
POSTER_DIR   = os.path.join(OUTPUT_DIR, "posters")
STORY_DIR    = os.path.join(OUTPUT_DIR, "stories")
STATE_DIR    = os.path.abspath("state")

TODAY    = datetime.date.today()
DATESTR  = TODAY.strftime("%Y%m%d")
DATESTR_HM = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M")

# ================== Env knobs ==================
# Mode: "all" (charts + posters + stories), "charts", "posters", "stories"
TWD_MODE = (os.getenv("TWD_MODE", "all") or "all").lower()

# Chart timeframe selector (D or W for render only)
DEFAULT_TF = (os.getenv("TWD_TF", "D") or "D").upper()

# 4H support zone computation knobs
FOURH_LOOKBACK_DAYS = int(os.getenv("TWD_4H_LOOKBACK_DAYS", "120"))
SWING_WINDOW        = int(os.getenv("TWD_SWING_WINDOW", "3"))
ATR_LEN             = int(os.getenv("TWD_ATR_LEN", "14"))
ZONE_PCT_TOL        = float(os.getenv("TWD_ZONE_PCT_TOL", "0.004"))  # 0.4% of price

# UI scaling
TWD_UI_SCALE    = float(os.getenv("TWD_UI_SCALE", "0.90"))
TWD_TEXT_SCALE  = float(os.getenv("TWD_TEXT_SCALE", "0.82"))
TWD_TLOGO_SCALE = float(os.getenv("TWD_TLOGO_SCALE","0.55"))

# Logos
BRAND_LOGO_PATH = os.getenv("BRAND_LOGO_PATH", "assets/brand_logo.png")
BRAND_MAXH      = int(os.getenv("TWD_BRAND_MAXH", "220"))
BRAND_MARGIN    = int(os.getenv("TWD_BRAND_MARGIN", "56"))

# Posters toggle (you can run posters-only on a separate schedule with TWD_MODE=posters)
TWD_POSTER_ON = (os.getenv("TWD_BREAKING_ON", "on").lower() in ("on","1","true","yes"))

# Stories (event-driven) — only create if cross-outlet confirmation detected
STORY_ON           = (os.getenv("TWD_STORY_ON", "on").lower() in ("on","1","true","yes"))
STORY_RECENCY_MIN  = int(os.getenv("TWD_STORY_RECENCY_MIN", "90"))   # lookback window (minutes)
STORY_MIN_SOURCES  = int(os.getenv("TWD_STORY_MIN_SOURCES", "2"))    # distinct publishers required
SEEN_STORIES_PATH  = os.path.join(STATE_DIR, "seen_stories.json")

# Optional Atom feed (not required for stories/posters)
TWD_FEED_ON       = os.getenv("TWD_FEED_ON", "off").lower() in ("on","1","true","yes")
TWD_FEED_TITLE    = os.getenv("TWD_FEED_TITLE", "TrendWatchDesk – Latest")
TWD_FEED_LINKBASE = os.getenv("TWD_FEED_LINKBASE", "").rstrip("/")

# ================== Pools / Tickers ==================
COMPANY_QUERY = {
    "META":"Meta Platforms","AMD":"Advanced Micro Devices","GOOG":"Google Alphabet","GOOGL":"Alphabet",
    "AAPL":"Apple","MSFT":"Microsoft","TSM":"Taiwan Semiconductor","TSLA":"Tesla",
    "JNJ":"Johnson & Johnson","MA":"Mastercard","V":"Visa","NVDA":"NVIDIA",
    "AMZN":"Amazon","SNOW":"Snowflake","SQ":"Block Inc","PYPL":"PayPal","UNH":"UnitedHealth"
}

# Symbols to sweep for macro/broad headlines (indices/ETFs/commodities/megacaps)
GLOBAL_NEWS_SYMBOLS = [
    "^GSPC","^NDX","^DJI","SPY","QQQ","DIA",
    "^TNX","DX-Y.NYB",
    "GLD","SLV","USO","UNG","DBC","GC=F","SI=F","CL=F","NG=F",
    "AAPL","MSFT","AMZN","NVDA","META","GOOGL","TSLA","AMD","TSM",
    "XLF","XLE","XLK","XLI"
]

def choose_tickers_somehow():
    # You said weights/randomisation live in your script; here we sample deterministically by date.
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
    # lenient match
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
    """One blue support box: previous swing HIGH nearest last if bullish; last swing LOW if bearish."""
    if df4h is None or df4h.empty: return (None, None)
    highs, lows = swing_points(df4h, w)
    last_px = float(df4h["Close"].iloc[-1])
    atrv = float(atr(df4h, atr_len).iloc[-1]) if len(df4h) > 1 else 0.0
    tol_abs = max(atrv, last_px * pct_tol)
    if trend_bullish:
        candidates = [(i, v) for (i, v) in highs if v <= last_px]
        if not candidates: return (None, None)
        # nearest prior swing-high to last price
        i_sel, v_sel = sorted(candidates, key=lambda t: (abs(last_px - t[1]), -t[0]))[0]
        level = float(v_sel)
    else:
        if not lows: return (None, None)
        # last swing low (most recent)
        i_sel, v_sel = max(lows, key=lambda t: t[0])
        level = float(v_sel)
    return (level - tol_abs, level + tol_abs)

# ================== Fetcher ==================
def fetch_one(ticker):
    """Return payload: (df_render, last, chg30, sup_low, sup_high, tf_tag, chg1d)"""
    tf = (os.getenv("TWD_TF", DEFAULT_TF) or DEFAULT_TF).upper().strip()

    # daily 1y
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

    # intraday 60m → resample to 4H for support zone
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

    # render frame
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
    GREEN    = (22,163,74,255)
    RED      = (239, 68,68,255)

    base = Image.new("RGBA", (W, H), BG)
    draw = ImageDraw.Draw(base)

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

    f_ticker = _font(58, True)
    f_price  = _font(32, True)
    f_delta  = _font(28, True)
    f_sub    = _font(22)
    f_axis   = _font(18)

    # layout
    outer_top = sp(56); outer_lr = sp(64); outer_bot = sp(56)
    header_h = sp(160); footer_h = sp(110)
    cx1, cy1, cx2, cy2 = (
        outer_lr,
        outer_top + header_h,
        W - outer_lr,
        H - outer_bot - footer_h
    )

    # header left aligned with even spacing
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

    # ticker logo top-right
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

    # prepare OHLC
    df2 = df[["Open","High","Low","Close"]].dropna()
    if df2.shape[0] < 2:
        base.convert("RGB").save(out_path, quality=95); return
    # show ~160-190 candles for readability
    if len(df2) > 180:
        df2 = df2.tail(180)

    ymin = float(np.nanmin(df2["Low"])); ymax = float(np.nanmax(df2["High"]))
    if not np.isfinite(ymin) or not np.isfinite(ymax) or abs(ymax - ymin) < 1e-6:
        ymin, ymax = (ymin - 0.5, ymax + 0.5) if np.isfinite(ymin) else (0, 1)
    yr = ymax - ymin; ymin -= 0.02*yr; ymax += 0.02*yr

    def sx(i): return cx1 + (i / max(1, len(df2)-1)) * (cx2 - cx1)
    def sy(v): return cy2 - ((float(v) - ymin) / (ymax - ymin)) * (cy2 - cy1)

    # grid (light)
    grid = Image.new("RGBA", (W, H), (0,0,0,0))
    g = ImageDraw.Draw(grid)
    for i in range(1, 7):
        y = cy1 + i * (cy2 - cy1) / 7.0
        g.line([(cx1, y), (cx2, y)], fill=(242,244,247,255), width=sp(1))
    for frac in (0.33, 0.66):
        y = cy1 + frac * (cy2 - cy1)
        g.line([(cx1, y), (cx2, y)], fill=(232,236,240,255), width=sp(1))
    for i in range(1, 9):
        x = cx1 + i * (cx2 - cx1) / 9.0
        g.line([(x, cy1), (x, cy2)], fill=(242,244,247,255), width=sp(1))
    base = Image.alpha_composite(base, grid)
    draw = ImageDraw.Draw(base)

    # support zone (single blue box)
    if sup_low is not None and sup_high is not None and np.isfinite(sup_low) and np.isfinite(sup_high):
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
    wick_w = max(1, sp(1))
    for i, row in enumerate(df2.itertuples(index=False)):
        O, Hh, Ll, C = row
        xx = sx(i)
        draw.line([(xx, sy(Hh)), (xx, sy(Ll))], fill=(140,140,140,255), width=wick_w)
        col = (22,163,74,255) if C >= O else (239,68,68,255)
        y1 = sy(max(O, C)); y2 = sy(min(O, C))
        if abs(y2 - y1) < 1: y2 = y1 + 1
        draw.rectangle([xx - half, y1, xx + half, y2], fill=col, outline=None)

    # right prices
    ticks = np.linspace(ymin, ymax, 5)
    for tval in ticks:
        y = sy(tval); label = f"{tval:,.2f}"
        bbox = draw.textbbox((0, 0), label, font=f_axis)
        draw.text((cx2 + sp(8), y - (bbox[3]-bbox[1])/2), label, fill=(140,145,150,255), font=f_axis)

    # footer left
    foot_x = outer_lr; foot_y = 1080 - BRAND_MARGIN - st(80)
    f_foot = _font(20)
    draw.text((foot_x, foot_y), "Support zone highlighted", fill=(120,120,120,255), font=f_foot)
    draw.text((foot_x, foot_y + st(20)), "Not financial advice", fill=(160,160,160,255), font=f_foot)

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

# ================== Captions (natural) ==================
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
    (df_render, last, chg30, sup_low, sup_high, tf_tag, _chg1d_payload) = payload
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
    random.shuffle(cues)
    cues = cues[:2]
    return f"• {emoji} {ticker} — {lead}{joiner}{'; '.join(cues)}"

# ================== News aggregation (cross-outlet clustering) ==================
def _normalize_title_key(t: str) -> str:
    t = (t or "").lower().strip()
    t = re.split(r"[-–—:]\s+", t, maxsplit=1)[0]  # strip subtitles for similarity
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return " ".join(t.split()[:10])  # first ~10 words

def _collect_yf_news_for_symbols(symbols):
    items = []
    for sym in symbols:
        try:
            for it in getattr(yf.Ticker(sym), "news", []) or []:
                items.append({
                    "symbol": sym,
                    "title": it.get("title") or "",
                    "publisher": (it.get("publisher") or "").strip(),
                    "url": it.get("link") or "",
                    "published_ts": it.get("providerPublishTime"),
                    "desc": it.get("summary", ""),
                })
        except Exception:
            continue
    # de-dup exact titles
    out, seen = [], set()
    for it in items:
        t = (it["title"] or "").strip()
        if not t or t in seen: 
            continue
        seen.add(t); out.append(it)
    out.sort(key=lambda x: x.get("published_ts") or 0, reverse=True)
    return out

def _cluster_and_pick_breaking(items, recency_min=90, min_sources=2):
    """Require >= min_sources distinct publishers and newest within recency window."""
    if not items: return None
    groups = {}
    now = datetime.datetime.utcnow().timestamp()
    for it in items:
        key = _normalize_title_key(it["title"])
        if not key:
            continue
        groups.setdefault(key, []).append(it)

    # preference table for credibility
    pref = ["reuters","financial times","wall street journal","wsj","bloomberg","yahoo finance","cnbc","marketwatch","barron's"]

    def score_item(it):
        p = (it.get("publisher") or "").lower()
        for idx, name in enumerate(reversed(pref)):  # earlier in list → higher weight
            if name in p:
                return 100 + idx
        return 0

    candidates = []
    for key, lst in groups.items():
        pubs = { (x.get("publisher") or "").lower() for x in lst if x.get("publisher") }
        newest = max((x.get("published_ts") or 0) for x in lst)
        age_min = (now - newest) / 60.0
        if len(pubs) >= min_sources and age_min <= recency_min:
            # rank by number of sources then freshness
            best_item = sorted(lst, key=lambda x: (score_item(x), x.get("published_ts") or 0), reverse=True)[0]
            candidates.append((len(pubs), newest, key, best_item))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    # return the top best_item
    return candidates[0][3]

def _hash_story_key(title: str) -> str:
    return hashlib.sha1((title or "").strip().lower().encode("utf-8")).hexdigest()

def _load_seen(path: str) -> set:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return set(data.get("seen", []))
    except Exception:
        return set()

def _save_seen(path: str, seen: set):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"seen": sorted(list(seen))}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# ================== Background picker & renderers (Poster/Story) ==================
def _headline_caps_short(title: str) -> str:
    t = (title or "").strip()
    t = re.split(r"[-–—:]\s+", t, maxsplit=1)[0]
    t = re.sub(r"\s+", " ", t)
    if len(t) > 68: t = t[:65] + "…"
    return t.upper()

def _pick_story_background_path(symbol_or_theme: str, story: dict) -> str | None:
    # 1) symbol folder
    base_dir = os.path.join("assets", "backgrounds", symbol_or_theme)
    if os.path.isdir(base_dir):
        cand = [os.path.join(base_dir, f) for f in os.listdir(base_dir)
                if f.lower().endswith((".jpg",".jpeg",".png"))]
        if cand: return sorted(cand)[0]

    # 2) theme by title
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

def _load_font_chain(size, bold=False):
    def _try_font(path, sz):
        try: return ImageFont.truetype(path, sz)
        except: return None
    grift_b = _try_font("assets/fonts/Grift-Bold.ttf", size)
    grift_r = _try_font("assets/fonts/Grift-Regular.ttf", size)
    robo_b  = _try_font("assets/fonts/Roboto-Bold.ttf", size) or _try_font("Roboto-Bold.ttf", size)
    robo_r  = _try_font("assets/fonts/Roboto-Regular.ttf", size) or _try_font("Roboto-Regular.ttf", size)
    if bold: return grift_b or robo_b or ImageFont.load_default()
    return grift_r or robo_r or ImageFont.load_default()

def render_news_poster(out_path, symbol_for_visual, story):
    """Square 1080x1080 NEWS card."""
    W, H = 1080, 1080
    BG       = (255,255,255,255)
    TEXT_DK  = (23,23,23,255)
    TEXT_MD  = (55,65,81,255)
    TEXT_LT  = (120,128,140,255)

    def sp(x): return int(round(x * TWD_UI_SCALE))
    def st(x): return int(round(x * TWD_TEXT_SCALE))

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
            bg_faint = Image.blend(bg_img, white, 0.85)
            base = Image.alpha_composite(base, bg_faint)
        except Exception:
            pass

    f_news  = _load_font_chain(st(28), True)
    f_head  = _load_font_chain(st(56), True)
    f_meta  = _load_font_chain(st(26), False)
    f_para  = _load_font_chain(st(30), False)

    PAD = sp(60)
    x1, y1, x2, y2 = PAD, PAD, W - PAD, H - PAD

    draw.text((x1, y1), "NEWS", fill=(90,90,90,255), font=f_news)

    # logo top-right
    tlogo_path = os.path.join("assets", "logos", f"{symbol_for_visual}.png")
    if not os.path.exists(tlogo_path):
        title = (story.get("title") or "").lower()
        theme_try = None
        if any(k in title for k in ["gold","bullion","xau","gld","gc=f"]): theme_try = "GOLD"
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

    # headline (ALL CAPS, concise)
    raw_title = (story.get("title") or "").strip()
    head_txt  = _headline_caps_short(raw_title)
    head_y = y1 + st(52)

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
        draw.text((x1, head_y), ln, fill=TEXT_DK, font=f_head)
        bb = draw.textbbox((x1, head_y), ln, font=f_head)
        head_y += (bb[3]-bb[1]) + st(6)

    # meta
    def _relative_phrase(ts_sec: float) -> str:
        if not ts_sec: return ""
        from datetime import datetime, timezone
        now = datetime.datetime.now(timezone.utc)
        t   = datetime.datetime.fromtimestamp(float(ts_sec), tz=timezone.utc)
        d   = now - t
        if d.days <= 0: return "today"
        if d.days == 1: return "yesterday"
        if d.days <= 3: return f"on {t.strftime('%a')}"
        if d.days <= 7: return "earlier this week"
        return t.strftime("%d %b")
    pub = (story.get("publisher") or "").strip()
    rel = _relative_phrase(story.get("published_ts"))
    meta_txt = " · ".join([p for p in (pub, rel) if p])
    if meta_txt:
        draw.text((x1, head_y + st(8)), meta_txt, fill=TEXT_LT, font=f_meta)
        head_y += st(46)

    # paragraph
    desc = (story.get("desc") or "").strip()
    if not desc:
        desc = "Market focus turns to how traders respond at key levels after the headline move."
    body = f"{raw_title.rstrip('.')}. {desc}"
    body = re.sub(r"\s+", " ", body).strip()
    if len(body) > 480: body = body[:477] + "…"

    body_y = head_y + st(16); max_w = x2 - x1
    words = body.split(); cur = ""; lines_p = []
    for w in words:
        test = (cur + " " + w).strip()
        bb = draw.textbbox((0,0), test, font=f_para)
        if bb[2]-bb[0] > max_w and cur:
            lines_p.append(cur); cur = w
        else:
            cur = test
    if cur: lines_p.append(cur)

    for ln in lines_p:
        draw.text((x1, body_y), ln, fill=TEXT_MD, font=f_para)
        bb = draw.textbbox((x1, body_y), ln, font=f_para)
        body_y += (bb[3]-bb[1]) + st(8)
        if body_y > (H - BRAND_MARGIN - st(180)):
            draw.text((x1, body_y), "…", fill=TEXT_MD, font=f_para)
            break

    # brand logo bottom-right
    if BRAND_LOGO_PATH and os.path.exists(BRAND_LOGO_PATH):
        try:
            blogo = Image.open(BRAND_LOGO_PATH).convert("RGBA")
            scale = min(1.0, BRAND_MAXH / max(1, blogo.height))
            blogo = blogo.resize((int(blogo.width*scale), int(blogo.height*scale)), Image.LANCZOS)
            base.alpha_composite(blogo, (W - BRAND_MARGIN - blogo.width, H - BRAND_MARGIN - blogo.height))
        except Exception:
            pass

    base.convert("RGB").save(out_path, quality=95)

def render_news_story(out_path, symbol_for_visual, story):
    """Vertical 1080x1920 Story card (same look, more space)."""
    W, H = 1080, 1920
    BG       = (255,255,255,255)
    TEXT_DK  = (23,23,23,255)
    TEXT_MD  = (55,65,81,255)
    TEXT_LT  = (120,128,140,255)

    def sp(x): return int(round(x * TWD_UI_SCALE))
    def st(x): return int(round(x * TWD_TEXT_SCALE))

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
            bg_faint = Image.blend(bg_img, white, 0.85)
            base = Image.alpha_composite(base, bg_faint)
        except Exception:
            pass

    f_news  = _load_font_chain(st(34), True)
    f_head  = _load_font_chain(st(72), True)
    f_meta  = _load_font_chain(st(30), False)
    f_para  = _load_font_chain(st(38), False)

    PAD = sp(64)
    x1, y1, x2, y2 = PAD, PAD, W - PAD, H - PAD

    draw.text((x1, y1), "NEWS", fill=(90,90,90,255), font=f_news)

    # logo top-right
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
            hmax = int(sp(130) * TWD_TLOGO_SCALE); hmax = max(70, hmax)
            scl = min(1.0, hmax / max(1, tlogo.height))
            tlogo = tlogo.resize((int(tlogo.width * scl), int(tlogo.height * scl)), Image.LANCZOS)
            base.alpha_composite(tlogo, (x2 - tlogo.width, y1))
        except Exception:
            pass

    # headline
    raw_title = (story.get("title") or "").strip()
    head_txt  = _headline_caps_short(raw_title)
    head_y = y1 + st(70)

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

    for ln in wrap_lines(head_txt, f_head, x2 - x1, 4):
        draw.text((x1, head_y), ln, fill=TEXT_DK, font=f_head)
        bb = draw.textbbox((x1, head_y), ln, font=f_head)
        head_y += (bb[3]-bb[1]) + st(8)

    # meta
    def _rel(ts):
        if not ts: return ""
        from datetime import datetime, timezone
        now = datetime.datetime.now(timezone.utc)
        t   = datetime.datetime.fromtimestamp(float(ts), tz=timezone.utc)
        d   = now - t
        if d.days <= 0: return "today"
        if d.days == 1: return "yesterday"
        if d.days <= 3: return f"on {t.strftime('%a')}"
        if d.days <= 7: return "earlier this week"
        return t.strftime("%d %b")
    pub = (story.get("publisher") or "").strip()
    rel = _rel(story.get("published_ts"))
    if pub or rel:
        draw.text((x1, head_y + st(12)), " · ".join([p for p in (pub, rel) if p]), fill=TEXT_LT, font=f_meta)
        head_y += st(58)

    # paragraph
    desc = (story.get("desc") or "").strip()
    if not desc:
        desc = "Traders watch key levels after the headline move and look for follow-through."
    body = f"{raw_title.rstrip('.')}. {desc}"
    body = re.sub(r"\s+", " ", body).strip()
    if len(body) > 700: body = body[:697] + "…"

    body_y = head_y + st(20); max_w = x2 - x1
    words = body.split(); cur = ""; lines_p = []
    for w in words:
        test = (cur + " " + w).strip()
        bb = draw.textbbox((0,0), test, font=f_para)
        if bb[2]-bb[0] > max_w and cur:
            lines_p.append(cur); cur = w
        else:
            cur = test
    if cur: lines_p.append(cur)

    for ln in lines_p:
        draw.text((x1, body_y), ln, fill=TEXT_MD, font=f_para)
        bb = draw.textbbox((x1, body_y), ln, font=f_para)
        body_y += (bb[3]-bb[1]) + st(10)
        if body_y > (H - BRAND_MARGIN - st(240)):
            draw.text((x1, body_y), "…", fill=TEXT_MD, font=f_para)
            break

    # brand logo
    if BRAND_LOGO_PATH and os.path.exists(BRAND_LOGO_PATH):
        try:
            blogo = Image.open(BRAND_LOGO_PATH).convert("RGBA")
            scale = min(1.0, BRAND_MAXH / max(1, blogo.height))
            blogo = blogo.resize((int(blogo.width*scale), int(blogo.height*scale)), Image.LANCZOS)
            base.alpha_composite(blogo, (W - BRAND_MARGIN - blogo.width, H - BRAND_MARGIN - blogo.height))
        except Exception:
            pass

    base.convert("RGB").save(out_path, quality=95)

# ================== Feed (optional) ==================
def write_atom_feed(items, out_path, feed_title, linkbase):
    from xml.sax.saxutils import escape
    now_iso = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    feed = []
    feed.append('<?xml version="1.0" encoding="UTF-8"?>')
    feed.append('<feed xmlns="http://www.w3.org/2005/Atom">')
    feed.append(f"<title>{escape(feed_title)}</title>")
    feed.append(f"<updated>{now_iso}</updated>")
    feed.append(f"<id>tag:trendwatchdesk,{DATESTR}:feed</id>")
    for it in items:
        title = escape(it.get("title","TrendWatchDesk"))
        caption = escape(it.get("caption",""))
        dt = it.get("dt", datetime.datetime.utcnow())
        updated = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        img_rel = os.path.basename(it["image_path"])
        img_url = f"{linkbase}/{img_rel}" if linkbase else img_rel
        feed.append("<entry>")
        feed.append(f"<title>{title}</title>")
        feed.append
