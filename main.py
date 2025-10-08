#!/usr/bin/env python3
# TrendWatchDesk — stable build (2025-10-08c)

import os, random, datetime, traceback, time, re, math
import numpy as np
import pandas as pd
import yfinance as yf
from PIL import Image, ImageDraw, ImageFont
import requests

# ================== Version / Paths ==================
TWD_VERSION = "2025-10-08c"
OUTPUT_DIR  = os.path.abspath("output")
TODAY       = datetime.date.today()
DATESTR     = TODAY.strftime("%Y%m%d")

# ================== Env knobs ==================
# Render frame: 'D' (daily ~1y) or 'W' (weekly ~60 bars)
DEFAULT_TF = (os.getenv("TWD_TF", "D") or "D").upper()

# 4H support detection
FOURH_LOOKBACK_DAYS = int(os.getenv("TWD_4H_LOOKBACK_DAYS", "120"))
SWING_WINDOW        = int(os.getenv("TWD_SWING_WINDOW", "3"))
ATR_LEN             = int(os.getenv("TWD_ATR_LEN", "14"))
ZONE_PCT_TOL        = float(os.getenv("TWD_ZONE_PCT_TOL", "0.004"))  # 0.4%

# UI scaling
TWD_UI_SCALE    = float(os.getenv("TWD_UI_SCALE", "0.90"))
TWD_TEXT_SCALE  = float(os.getenv("TWD_TEXT_SCALE", "0.70"))
TWD_TLOGO_SCALE = float(os.getenv("TWD_TLOGO_SCALE","0.55"))

# Logos
BRAND_LOGO_PATH = os.getenv("BRAND_LOGO_PATH", "assets/brand_logo.png")
BRAND_MAXH      = int(os.getenv("TWD_BRAND_MAXH", "220"))
BRAND_MARGIN    = int(os.getenv("TWD_BRAND_MARGIN", "56"))

# News
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "").strip()
SESS = requests.Session()
SESS.headers.update({"User-Agent": "TWD/1.0"})

# Caption chatter when no fresh headline
TWD_PT_CHATTER_ON   = os.getenv("TWD_PT_CHATTER", "on").lower() in ("on","1","true","yes")
TWD_PT_CHANCE_DENOM = int(os.getenv("TWD_PT_CHANCE", "8") or 8)

# Breaking poster feature
TWD_BREAKING_ON            = os.getenv("TWD_BREAKING_ON", "on").lower() in ("on","1","true","yes")
TWD_BREAKING_RECENCY_HOURS = int(os.getenv("TWD_BREAKING_RECENCY_HOURS", "12"))
TWD_BREAKING_MIN_MOVE      = float(os.getenv("TWD_BREAKING_MIN_MOVE", "5.0"))
TWD_BREAKING_MAX_PER_RUN   = int(os.getenv("TWD_BREAKING_MAX_PER_RUN", "2"))

# ================== Pools / Tickers ==================
COMPANY_QUERY = {
    "META":"Meta Platforms","AMD":"Advanced Micro Devices","GOOG":"Google Alphabet","GOOGL":"Alphabet",
    "AAPL":"Apple","MSFT":"Microsoft","TSM":"Taiwan Semiconductor","TSLA":"Tesla",
    "JNJ":"Johnson & Johnson","MA":"Mastercard","V":"Visa","NVDA":"NVIDIA",
    "AMZN":"Amazon","SNOW":"Snowflake","SQ":"Block Inc","PYPL":"PayPal","UNH":"UnitedHealth"
}

def choose_tickers_somehow():
    rnd = random.Random(DATESTR)
    pool = list(COMPANY_QUERY.keys())
    k = min(6, len(pool))
    return rnd.sample(pool, k)

# ================== Data helpers ==================
def _find_col(df: pd.DataFrame, name: str):
    if df is None or df.empty: return None
    # direct
    if name in df.columns:
        ser = df[name]
        if isinstance(ser, pd.DataFrame): ser = ser.iloc[:, 0]
        return pd.to_numeric(ser, errors="coerce")
    # multi-index
    if isinstance(df.columns, pd.MultiIndex):
        if name in df.columns.get_level_values(-1):
            sub = df.xs(name, axis=1, level=-1)
            ser = sub.iloc[:, 0] if isinstance(sub, pd.DataFrame) else sub
            return pd.to_numeric(ser, errors="coerce")
    # normalized
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
        # previous swing HIGH below price, closest
        candidates = [(i, v) for (i, v) in highs if v <= last_px]
        if not candidates: return (None, None)
        i_sel, v_sel = sorted(candidates, key=lambda t: (abs(last_px - t[1]), -t[0]))[0]
        level = float(v_sel)
    else:
        # last swing LOW
        if not lows: return (None, None)
        i_sel, v_sel = max(lows, key=lambda t: t[0])
        level = float(v_sel)

    return (level - tol_abs, level + tol_abs)

# ================== Fetcher ==================
def fetch_one(ticker):
    """Return payload: (df_render, last, chg30, sup_low, sup_high, tf_tag, chg1d)"""
    tf = (os.getenv("TWD_TF", DEFAULT_TF) or DEFAULT_TF).upper().strip()

    # 1) Daily 1y for render
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

    # 2) 60m → resample 4H for support zone
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

    # 3) Choose render frame
    if tf == "W":
        df_render = ohlc_d.resample("W-FRI").agg(
            {"Open":"first","High":"max","Low":"min","Close":"last"}
        ).dropna().tail(60)
        tf_tag = "W"
    else:
        df_render = ohlc_d.dropna().tail(260)  # ~1y daily
        tf_tag = "D"

    if df_render is None or df_render.empty: return None
    return (df_render, last, float(chg30), sup_low, sup_high, tf_tag, float(chg1d))

# ================== Renderer (chart) ==================
def render_single_post(out_path, ticker, payload):
    (df, last, chg30, sup_low, sup_high, tf_tag, _chg1d) = payload

    # ---- scales ----
    def sp(x: float) -> int: return int(round(x * TWD_UI_SCALE))
    def st(x: float) -> int: return int(round(x * TWD_TEXT_SCALE))

    # ---- theme ----
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
    SR_FILL  = (120,162,255,50)
    SR_STROK = (120,162,255,120)

    base = Image.new("RGBA", (W, H), BG)
    draw = ImageDraw.Draw(base)

    # ---- fonts ----
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

    f_ticker = _font(76, True)
    f_price  = _font(40, True)
    f_delta  = _font(34, True)
    f_sub    = _font(26)
    f_axis   = _font(22)

    # ---- layout ----
    outer_top = sp(56); outer_lr = sp(64); outer_bot = sp(56)
    header_h = sp(190); footer_h = sp(120)
    cx1, cy1, cx2, cy2 = (
        outer_lr,
        outer_top + header_h,
        W - outer_lr,
        H - outer_bot - footer_h
    )

    # ---- header left, vertically spaced ----
    title_x, title_y = outer_lr, outer_top
    GAP = st(12)

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

    # ---- ticker logo (top-right) ----
    tlogo_path = os.path.join("assets", "logos", f"{ticker}.png")
    if os.path.exists(tlogo_path):
        try:
            tlogo = Image.open(tlogo_path).convert("RGBA")
            hmax = int(sp(80) * TWD_TLOGO_SCALE)
            hmax = max(40, hmax)
            scl = min(1.0, hmax / max(1, tlogo.height))
            tlogo = tlogo.resize((int(tlogo.width * scl), int(tlogo.height * scl)), Image.LANCZOS)
            base.alpha_composite(tlogo, (W - outer_lr - tlogo.width, title_y))
        except Exception:
            pass

    # ---- data ----
    df2 = df[["Open","High","Low","Close"]].dropna()
    if df2.shape[0] < 2:
        out = base.convert("RGB"); os.makedirs(os.path.dirname(out_path), exist_ok=True); out.save(out_path, quality=95); return

    # cut to last ~220 candles to keep uncluttered
    if len(df2) > 220:
        df2 = df2.tail(220)

    ymin = float(np.nanmin(df2["Low"])); ymax = float(np.nanmax(df2["High"]))
    if not np.isfinite(ymin) or not np.isfinite(ymax) or abs(ymax - ymin) < 1e-6:
        ymin, ymax = (ymin - 0.5, ymax + 0.5) if np.isfinite(ymin) else (0, 1)
    yr = ymax - ymin; ymin -= 0.02*yr; ymax += 0.02*yr

    def sx(i): return cx1 + (i / max(1, len(df2)-1)) * (cx2 - cx1)
    def sy(v): return cy2 - ((float(v) - ymin) / (ymax - ymin)) * (cy2 - cy1)

    # ---- grid ----
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

    # ---- support zone (single blue box) ----
    if sup_low is not None and sup_high is not None and np.isfinite(sup_low) and np.isfinite(sup_high):
        sup_y1, sup_y2 = sy(sup_high), sy(sup_low)
        sup_rect = [cx1, min(sup_y1, sup_y2), cx2, max(sup_y1, sup_y2)]
        sup_layer = Image.new("RGBA", (W, H), (0,0,0,0))
        ImageDraw.Draw(sup_layer).rectangle(sup_rect, fill=SR_FILL, outline=SR_STROK, width=sp(2))
        base = Image.alpha_composite(base, sup_layer)
        draw = ImageDraw.Draw(base)

    # ---- candles ----
    n = len(df2)
    body_px = max(1, int(((cx2 - cx1) / max(180, n * 1.05)) * TWD_UI_SCALE))
    half = max(1, body_px // 2)
    wick_w = max(1, sp(1))
    for i, row in enumerate(df2.itertuples(index=False)):
        O, Hh, Ll, C = row
        xx = sx(i)
        draw.line([(xx, sy(Hh)), (xx, sy(Ll))], fill=WICK, width=wick_w)
        col = GREEN if C >= O else RED
        y1 = sy(max(O, C)); y2 = sy(min(O, C))
        if abs(y2 - y1) < 1: y2 = y1 + 1
        draw.rectangle([xx - half, y1, xx + half, y2], fill=col, outline=None)

    # ---- right axis ticks ----
    ticks = np.linspace(ymin, ymax, 5)
    for tval in ticks:
        y = sy(tval); label = f"{tval:,.2f}"
        bbox = draw.textbbox((0, 0), label, font=f_axis)
        th = bbox[3] - bbox[1]
        draw.text((cx2 + sp(8), y - th/2), label, fill=TEXT_LT, font=f_axis)

    # ---- footer ----
    foot_x = outer_lr; foot_y = H - outer_bot - st(54)
    draw.text((foot_x, foot_y), "Support zone highlighted", fill=TEXT_LT, font=f_sub)
    draw.text((foot_x, foot_y + st(22)), "Not financial advice", fill=(160,160,160,255), font=f_sub)

    # ---- brand logo (bottom-right) ----
    if BRAND_LOGO_PATH and os.path.exists(BRAND_LOGO_PATH):
        try:
            blogo = Image.open(BRAND_LOGO_PATH).convert("RGBA")
            scale = min(1.0, BRAND_MAXH / max(1, blogo.height))
            new_w = max(1, int(round(blogo.width * scale)))
            new_h = max(1, int(round(blogo.height * scale)))
            blogo = blogo.resize((new_w, new_h), Image.LANCZOS)
            x = W - BRAND_MARGIN - new_w
            y = H - BRAND_MARGIN - new_h
            base.alpha_composite(blogo, (x, y))
        except Exception:
            pass

    out = base.convert("RGB")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.save(out_path, quality=95)

# ================== News + captions (time-aware) ==================
PREF_PUBLISHERS = {
    "The Wall Street Journal": 10, "Wall Street Journal": 10, "WSJ": 10,
    "Financial Times": 10, "FT": 10,
    "Reuters": 10,
    "Bloomberg": 9,
    "Yahoo Finance": 8, "Yahoo": 7,
    "CNBC": 7, "MarketWatch": 7, "Barron's": 7,
    "The Verge": 5, "TechCrunch": 5, "Seeking Alpha": 5,
}
NEWS_KEYWORDS_BOOST = re.compile(
    r"(upgrade|downgrade|price target|pt|earnings|guidance|revenue|profit|"
    r"beats|misses|surge|plunge|deal|acquisition|merger|layoff|dividend|"
    r"lawsuit|antitrust|regulator|AI|GPU|chip|fab|capacity|contract|partnership|"
    r"Buy|Overweight|Neutral|Sell)",
    re.I
)
_BANK_PAT = re.compile(r"(Morgan Stanley|Barclays|Goldman(?: Sachs)?|Citi|JPMorgan|Bank of America|BofA|Deutsche(?: Bank)?)", re.I)
_PT_PAT   = re.compile(r"\$?\s*(\d{2,4})(?:\s*(?:price\s*target|target|pt))?", re.I)

def news_fetch_all(ticker):
    """Aggregate NewsAPI + yfinance with publisher, epoch ts, url, and description."""
    out = []
    name = COMPANY_QUERY.get(ticker, ticker)

    # yfinance block
    try:
        items = getattr(yf.Ticker(ticker), "news", []) or []
        for it in items:
            out.append({
                "title": it.get("title") or "",
                "publisher": it.get("publisher") or "",
                "published_ts": it.get("providerPublishTime"),
                "url": it.get("link") or "",
                "desc": ""
            })
    except Exception:
        pass

    # NewsAPI (optional)
    if NEWSAPI_KEY:
        try:
            r = SESS.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": f'"{name}" OR {ticker}',
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 12,
                },
                headers={"X-Api-Key": NEWSAPI_KEY},
                timeout=10
            )
            if r.ok:
                d = r.json()
                for a in d.get("articles", []):
                    ts = None
                    try:
                        ts = pd.to_datetime(a.get("publishedAt")).timestamp() if a.get("publishedAt") else None
                    except Exception:
                        ts = None
                    out.append({
                        "title": a.get("title") or "",
                        "publisher": (a.get("source") or {}).get("name","") or "",
                        "published_ts": ts,
                        "url": a.get("url") or "",
                        "desc": a.get("description") or ""
                    })
        except Exception:
            pass

    # De-dup by title
    seen = set(); cleaned = []
    for it in out:
        t = (it["title"] or "").strip()
        if not t or t in seen:
            continue
        seen.add(t)
        cleaned.append(it)
    return cleaned

def _best_news_from_items(items, max_age_days=5):
    if not items: return None
    now = time.time()
    scored = []
    for it in items:
        t = (it.get("title") or "").strip()
        if not t: continue
        pub = (it.get("publisher") or "").strip()
        ts  = it.get("published_ts")
        if ts is None:
            age_days = 999.0
        else:
            age_days = max(0.0, (now - float(ts)) / 86400.0)
        if age_days > max_age_days:
            continue
        pub_score = PREF_PUBLISHERS.get(pub, 3)
        recency_score = max(0, 6 - int(age_days))
        kw_score = 3 if NEWS_KEYWORDS_BOOST.search(t) else 0
        score = pub_score*2 + recency_score + kw_score
        scored.append((score, pub, t))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    _, pub, title = scored[0]
    return f"{title} ({pub})" if pub else title

def news_headline_for(ticker):
    items = news_fetch_all(ticker)
    return _best_news_from_items(items, max_age_days=5)

# Time-aware caption helpers
def _shorten(txt, n=100):
    txt = (txt or "").strip()
    return txt if len(txt) <= n else (txt[: n-1] + "…")

def _relative_phrase(ts_sec: float) -> str:
    if not ts_sec: return ""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    t   = datetime.datetime.fromtimestamp(float(ts_sec), tz=timezone.utc) if hasattr(datetime, "datetime") else None

def _relative_phrase(ts_sec: float) -> str:
    if not ts_sec:
        return ""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    t   = datetime.fromtimestamp(float(ts_sec), tz=timezone.utc)
    delta = now - t
    days = delta.days
    if days <= 0:
        return "today"
    if days == 1:
        return "yesterday"
    if days <= 3:
        return f"on {t.strftime('%a')}"
    if days <= 7:
        return "earlier this week"
    return "last week"

def _compute_recent_changes(df_render: pd.DataFrame, tf_tag: str):
    try:
        c = df_render["Close"].dropna()
        if len(c) < 2:
            return (0.0, 0.0)
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

def _pick_news_windows(items):
    if not items: return (None, None)
    now = time.time()
    prim, sec = [], []
    for it in items:
        t = (it.get("title") or "").strip()
        if not t: continue
        pub = (it.get("publisher") or "").strip()
        ts  = it.get("published_ts")
        age_d = 999.0
        if ts is not None:
            age_d = max(0.0, (now - float(ts)) / 86400.0)
        bucket = prim if age_d <= 3.0 else (sec if age_d <= 10.0 else None)
        if bucket is None: 
            continue
        pub_score = PREF_PUBLISHERS.get(pub, 3)
        rec = max(0, int(6 - min(6, age_d)))
        kw  = 3 if NEWS_KEYWORDS_BOOST.search(t) else 0
        score = pub_score*2 + rec + kw
        bucket.append((score, it))
    best_p = sorted(prim, key=lambda x: x[0], reverse=True)[0][1] if prim else None
    best_s = sorted(sec,  key=lambda x: x[0], reverse=True)[0][1] if sec  else None
    return (best_p, best_s)

_SECTOR_EMOJIS = {
    "TECH":    ["🖥️","🔎","🧠","💻","📡"],
    "HEALTH":  ["💊","🧬","⚕️","🩺"],
    "FINANCE": ["💳","🏦","📈","💸"],
    "SEMIS":   ["🔌","⚡","🔧","🧮"],
    "GENERIC": ["📈","🔎","⚡","🚀"],
}
def _emoji_for(ticker, rnd):
    if ticker in ["AAPL","MSFT","META","GOOG","GOOGL","AMZN","TSLA"]:
        pool = _SECTOR_EMOJIS["TECH"]
    elif ticker in ["JNJ","UNH"]:
        pool = _SECTOR_EMOJIS["HEALTH"]
    elif ticker in ["MA","V","PYPL","SQ"]:
        pool = _SECTOR_EMOJIS["FINANCE"]
    elif ticker in ["AMD","NVDA","TSM","ASML","QCOM","INTC","MU"]:
        pool = _SECTOR_EMOJIS["SEMIS"]
    else:
        pool = _SECTOR_EMOJIS["GENERIC"]
    return rnd.choice(pool)

def caption_line(ticker, _headline_unused, payload, seed=None):
    (df_render, last, chg30, sup_low, sup_high, tf_tag, chg1d_payload) = payload
    rnd = random.Random((seed or DATESTR) + ticker)
    emoji = _emoji_for(ticker, rnd)
    joiner = rnd.choice([" · ", " — "])

    chg1d, chg5d = _compute_recent_changes(df_render, tf_tag)
    items = news_fetch_all(ticker)
    primary, secondary = _pick_news_windows(items)

    # Lead (news-aware across days)
    lead = ""
    if primary:
        p_title = _shorten(primary.get("title"), 110)
        p_pub   = (primary.get("publisher") or "").strip()
        p_ts    = primary.get("published_ts")
        rel     = _relative_phrase(p_ts)
        lead_templates = [
            'Latest {rel}: “{h}”.',
            'Fresh {rel}: “{h}”.',
            'In focus {rel}: “{h}”.',
            'With “{h}” {rel}.',
        ]
        lead = rnd.choice(lead_templates).format(h=p_title, rel=rel)
        # Price target hook
        if _BANK_PAT.search(p_title):
            m = _PT_PAT.search(p_title)
            if m:
                lead += f" PT ~${m.group(1)} 🎯"
        # Prior story for continuity
        if secondary and rnd.random() < 0.7:
            s_title = _shorten(secondary.get("title"), 80)
            s_rel   = _relative_phrase(secondary.get("published_ts"))
            lead += f" Following {s_rel}: “{s_title}”."

    else:
        mv = abs(chg1d)
        if mv >= 5:
            dir_word = "Up" if chg1d > 0 else "Down"
            week_note = f" (~{chg5d:+.0f}% this week)" if abs(chg5d) >= 3 else ""
            lead = f"{dir_word} ~{abs(chg1d):.0f}% today{week_note}."
        else:
            week_note = f"{chg5d:+.1f}% this week" if chg5d != 0 else "watchlist update"
            lead = rnd.choice([
                f"Steady tape; {week_note}.",
                f"Range holding; {week_note}.",
                f"Eyes on weekly structure; {week_note}.",
            ])

    # Cues
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

    if tf_tag == "D" and rnd.random() < 0.6:
        cues.append(rnd.choice([
            "weekly trend doing the heavy lifting",
            "bigger picture still driving flows",
            "higher timeframe levels matter here",
        ]))

    rnd.shuffle(cues)
    cues = cues[: rnd.choice([2,3])]
    cues_txt = "; ".join(cues)

    return f"• {emoji} {ticker} — {lead}{joiner}{cues_txt}"

# ================== Breaking poster (optional) ==================
def select_breaking_story(ticker, items, chg1d, max_age_hours=12):
    if not items: return None
    now = time.time()
    candidates = []
    for it in items:
        t = (it.get("title") or "").strip()
        if not t: continue
        pub = (it.get("publisher") or "").strip()
        ts  = it.get("published_ts")
        age_hours = max(0.0, (now - float(ts)) / 3600.0) if ts is not None else 999.0
        age_ok = (age_hours <= max_age_hours)
        move_big = abs(chg1d) >= TWD_BREAKING_MIN_MOVE
        if not (age_ok or move_big):
            continue
        pub_score = PREF_PUBLISHERS.get(pub, 3)
        kw_score = 3 if NEWS_KEYWORDS_BOOST.search(t) else 0
        recency_score = max(0, int(12 - min(age_hours, 12)))
        move_bonus = 4 if move_big else 0
        score = pub_score*2 + kw_score + recency_score + move_bonus
        candidates.append((score, it))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

def render_breaking_poster(out_path, ticker, story):
    W, H = 1080, 1080
    BG       = (255,255,255,255)
    TEXT_DK  = (23,23,23,255)
    TEXT_MD  = (55,65,81,255)
    TEXT_LT  = (120,128,140,255)
    ACCENT   = (120,162,255,255)

    base = Image.new("RGBA", (W, H), BG)
    draw = ImageDraw.Draw(base)

    def _try_font(path, size):
        try: return ImageFont.truetype(path, size)
        except: return None
    def _font(size, bold=False):
        sz = int(round(size * TWD_TEXT_SCALE))
        grift_b = _try_font("assets/fonts/Grift-Bold.ttf", sz)
        grift_r = _try_font("assets/fonts/Grift-Regular.ttf", sz)
        robo_b  = _try_font("assets/fonts/Roboto-Bold.ttf", sz) or _try_font("Roboto-Bold.ttf", sz)
        robo_r  = _try_font("assets/fonts/Roboto-Regular.ttf", sz) or _try_font("Roboto-Regular.ttf", sz)
        if bold: return grift_b or robo_b or ImageFont.load_default()
        return grift_r or robo_r or ImageFont.load_default()

    f_kicker = _font(30, True)
    f_head0  = _font(68, True)
    f_meta   = _font(28)
    f_bull   = _font(32)
    f_tkr    = _font(40, True)

    PAD = int(64 * TWD_UI_SCALE)
    x1, y1, x2, y2 = PAD, PAD, W - PAD, H - PAD

    # kicker stripe
    stripe_h = int(10 * TWD_UI_SCALE)
    draw.rectangle([x1, y1, x2, y1 + stripe_h], fill=ACCENT)
    draw.text((x1, y1 + stripe_h + int(18 * TWD_UI_SCALE)), "Breaking", fill=ACCENT, font=f_kicker)

    # ticker logo required
    tlogo_path = os.path.join("assets", "logos", f"{ticker}.png")
    if os.path.exists(tlogo_path):
        try:
            tlogo = Image.open(tlogo_path).convert("RGBA")
            hmax = int(110 * TWD_TLOGO_SCALE)
            hmax = max(60, hmax)
            scl = min(1.0, hmax / max(1, tlogo.height))
            tlogo = tlogo.resize((int(tlogo.width * scl), int(tlogo.height * scl)), Image.LANCZOS)
            base.alpha_composite(tlogo, (x2 - tlogo.width, y1 + int(10 * TWD_UI_SCALE)))
        except Exception:
            pass
    draw.text((x2 - 180, y1 + stripe_h + int(18 * TWD_UI_SCALE)), ticker, fill=TEXT_LT, font=f_tkr)

    # headline auto-fit
    title = (story.get("title") or "").strip() or f"{ticker}: Market-moving update"
    def wrap_lines(text, font, max_w, max_lines):
        words = text.split()
        lines, cur = [], []
        for w in words:
            cur.append(w)
            test = " ".join(cur)
            bb = draw.textbbox((0,0), test, font=font)
            if bb[2]-bb[0] > max_w:
                cur.pop()
                lines.append(" ".join(cur))
                cur = [w]
                if len(lines) == max_lines:
                    break
        if cur and len(lines) < max_lines:
            lines.append(" ".join(cur))
        return lines

    y_cur = y1 + int(80 * TWD_UI_SCALE)
    head_max_w = x2 - x1
    head_max_h = int(360 * TWD_UI_SCALE)
    max_lines  = 4
    min_px     = 34
    size_px = f_head0.size
    head_font = f_head0
    while True:
        lines = wrap_lines(title, head_font, head_max_w, max_lines)
        htot = 0
        for ln in lines:
            bb = draw.textbbox((0,0), ln, font=head_font)
            htot += (bb[3]-bb[1]) + int(6 * TWD_UI_SCALE)
        if htot <= head_max_h or size_px <= min_px:
            break
        size_px = int(size_px * 0.92)
        head_font = _font(size_px, True)

    for ln in lines:
        draw.text((x1, y_cur), ln, fill=TEXT_DK, font=head_font)
        bb = draw.textbbox((0,0), ln, font=head_font)
        y_cur += (bb[3]-bb[1]) + int(6 * TWD_UI_SCALE)

    # source + time
    pub = (story.get("publisher") or "").strip()
    ts  = story.get("published_ts")
    ago_txt = ""
    if ts:
        delta = max(0.0, (time.time() - float(ts)) / 3600.0)
        ago_txt = f"{int(delta*60)}m ago" if delta < 1 else f"{int(delta)}h ago"
    meta_line = " · ".join([p for p in (pub, ago_txt) if p])

    def truncate_to_width(text, font, max_w):
        if not text: return text
        bb = draw.textbbox((0,0), text, font=font)
        if bb[2]-bb[0] <= max_w:
            return text
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo+hi)//2
            test = text[:mid] + "…"
            bb = draw.textbbox((0,0), test, font=font)
            if bb[2]-bb[0] <= max_w:
                lo = mid + 1
            else:
                hi = mid
        return text[:max(1, lo-1)] + "…"

    if meta_line:
        meta_line = truncate_to_width(meta_line, f_meta, head_max_w)
        draw.text((x1, y_cur + int(8 * TWD_UI_SCALE)), meta_line, fill=TEXT_LT, font=f_meta)
        y_cur += int(44 * TWD_UI_SCALE)

    # bullets
    desc = (story.get("desc") or "").strip()
    base_text = (title + ". " + desc).strip()
    parts = re.split(r"(?i)(?:;|:|–|—|-|•| \| )", base_text)
    parts = [p.strip() for p in parts if p.strip()]
    bullets = []
    for p in parts:
        if any(k in p.lower() for k in ("breaking","latest","news")): continue
        bullets.append(p if len(p) <= 110 else (p[:107]+"…"))
        if len(bullets) == 3: break
    if not bullets:
        bullets = ["Key update for traders.", "Watch reactions at key levels.", "More details in the source."]

    bullets_max_h = H - (y_cur) - int(260 * TWD_UI_SCALE)
    used_h = 0; line_gap = int(10 * TWD_UI_SCALE)
    for b in bullets:
        bb = draw.textbbox((0,0), f"• {b}", font=f_bull)
        hline = (bb[3]-bb[1])
        if used_h + hline + line_gap > bullets_max_h:
            break
        draw.text((x1, y_cur), f"• {b}", fill=TEXT_MD, font=f_bull)
        y_cur += hline + line_gap
        used_h += hline + line_gap

    # brand logo on plain background
    if BRAND_LOGO_PATH and os.path.exists(BRAND_LOGO_PATH):
        try:
            blogo = Image.open(BRAND_LOGO_PATH).convert("RGBA")
            scale = min(1.0, BRAND_MAXH / max(1, blogo.height))
            new_w = max(1, int(round(blogo.width * scale)))
            new_h = max(1, int(round(blogo.height * scale)))
            blogo = blogo.resize((new_w, new_h), Image.LANCZOS)
            base.alpha_composite(blogo, (W - BRAND_MARGIN - new_w, H - BRAND_MARGIN - new_h))
        except Exception:
            pass

    base.convert("RGB").save(out_path, quality=95)

# ================== CTA rotation ==================
CTA_MONDAY = [
    "Save for later 📌 · Comment your levels 💬 · See charts in carousel ➡️",
    "Tap save 📌 · What’s your take? 💬 · Swipe for charts ➡️",
    "Bookmark this 📌 · Which ticker next? 💬 · More inside ➡️",
]
CTA_WEDNESDAY = [
    "Midweek check-in ✅ · Drop your view 💬 · Swipe for setups ➡️",
    "Save 📌 · Agree or disagree? 💬 · See full charts ➡️",
    "Add to watchlist 📌 · Share your levels 💬 · Carousel inside ➡️",
]
CTA_FRIDAY = [
    "Wrap the week 🎯 · Comment your plan 💬 · Swipe for charts ➡️",
    "Bookmark for the weekend 📌 · Your levels below 💬 · More charts ➡️",
    "Save 📌 · What stood out this week? 💬 · Full set inside ➡️",
]
def pick_cta_for_today(day_idx: int) -> str:
    rnd = random.Random(DATESTR + "-cta")
    if day_idx == 0: return rnd.choice(CTA_MONDAY)
    if day_idx == 2: return rnd.choice(CTA_WEDNESDAY)
    if day_idx == 4: return rnd.choice(CTA_FRIDAY)
    return rnd.choice(CTA_MONDAY + CTA_WEDNESDAY + CTA_FRIDAY)

# ================== Main ==================
def main():
    print(f"[info] TrendWatchDesk {TWD_VERSION} starting…")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tickers = choose_tickers_somehow()
    print("[info] selected tickers:", tickers)

    saved = 0
    posters_made = 0
    captions = []

    for t in tickers:
        try:
            payload = fetch_one(t)
            if not payload:
                print(f"[warn] no data for {t}, skipping")
                continue

            out_path = os.path.join(OUTPUT_DIR, f"twd_{t}_{DATESTR}.png")
            render_single_post(out_path, t, payload)
            print("done:", out_path)
            saved += 1

            # captions
            # We ignore the old single-headline input and build time-aware lead from news windows
            line = caption_line(t, None, payload, seed=DATESTR)
            captions.append(line)

            # optional breaking posters
            if TWD_BREAKING_ON and posters_made < TWD_BREAKING_MAX_PER_RUN:
                chg1d = payload[-1]
                items_all = news_fetch_all(t)
                story = select_breaking_story(t, items_all, chg1d, max_age_hours=TWD_BREAKING_RECENCY_HOURS)
                if story:
                    p_out = os.path.join(OUTPUT_DIR, "posters", f"twd_BREAKING_{t}_{DATESTR}.png")
                    os.makedirs(os.path.dirname(p_out), exist_ok=True)
                    render_breaking_poster(p_out, t, story)
                    posters_made += 1
                    print(f"[info] poster created: {p_out}")

        except Exception as e:
            print(f"Error: failed for {t}: {e}")
            traceback.print_exc()

    print(f"[info] saved images: {saved}")

    if saved > 0:
        caption_path = os.path.join(OUTPUT_DIR, f"caption_{DATESTR}.txt")
        now_str = TODAY.strftime("%d %b %Y")
        header = f"Ones to Watch – {now_str}\n\n"
        footer_cta = pick_cta_for_today(TODAY.weekday())
        footer = f"\n\n{footer_cta}\n\nIdeas only — not financial advice"
        with open(caption_path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write("\n\n".join(captions))
            f.write(footer)
        print("[info] wrote caption:", caption_path)

if __name__ == "__main__":
    main()
