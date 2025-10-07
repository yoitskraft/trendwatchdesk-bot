#!/usr/bin/env python3
import os, sys, math, json, random, datetime, traceback
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

# Controls
DEFAULT_TF = (os.getenv("TWD_TF", "D") or "D").upper()  # 'D' daily (render), 'W' weekly (render). S/R detection uses 4H regardless.
FOURH_LOOKBACK_DAYS = int(os.getenv("TWD_4H_LOOKBACK_DAYS", "120"))  # 4h S/R detection window
SWING_WINDOW = int(os.getenv("TWD_SWING_WINDOW", "3"))               # swing window for local extrema (3–5 is typical)
MIN_TOUCHES = int(os.getenv("TWD_MIN_TOUCHES", "3"))                  # require >= touches per zone
ZONE_PCT_TOL = float(os.getenv("TWD_ZONE_PCT_TOL", "0.004"))          # 0.4% price tolerance for clustering
ATR_LEN = int(os.getenv("TWD_ATR_LEN", "14"))                         # ATR for tolerance/visuals

# ------------------ Company names (pool) ------------------
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
    # NewsAPI (if key provided)
    if NEWSAPI_KEY:
        try:
            r = SESS.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": f'"{name}" OR {ticker}',
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 1
                },
                headers={"X-Api-Key": NEWSAPI_KEY},
                timeout=8
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
    # yfinance fallback
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

# ------------------ Ticker chooser (deterministic by date) ------------------
def choose_tickers_somehow():
    pool = list(COMPANY_QUERY.keys())
    rnd = random.Random(DATESTR)
    k = min(6, len(pool))
    return rnd.sample(pool, k)

# ------------------ Robust OHLC helpers ------------------
def _find_col(df: pd.DataFrame, name: str):
    if df is None or df.empty:
        return None
    # single-index
    if name in df.columns:
        ser = df[name]
        if isinstance(ser, pd.DataFrame):
            ser = ser.iloc[:, 0]
        return pd.to_numeric(ser, errors="coerce")
    # normalized names
    try:
        norm = {str(c).lower().replace(" ", ""): c for c in df.columns}
        key = name.lower().replace(" ", "")
        if key in norm:
            ser = df[norm[key]]
            if isinstance(ser, pd.DataFrame):
                ser = ser.iloc[:, 0]
            return pd.to_numeric(ser, errors="coerce")
    except Exception:
        pass
    # MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        lvlN = df.columns.get_level_values(-1)
        if name in set(lvl0):
            sub = df[name]
            ser = sub.iloc[:, 0] if isinstance(sub, pd.DataFrame) else sub
            return pd.to_numeric(ser, errors="coerce")
        if name in set(lvlN):
            sub = df.xs(name, axis=1, level=-1)
            ser = sub.iloc[:, 0] if isinstance(sub, pd.DataFrame) else sub
            return pd.to_numeric(ser, errors="coerce")
    return None

def _get_ohlc_df(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    o = _find_col(df, "Open")
    h = _find_col(df, "High")
    l = _find_col(df, "Low")
    c = _find_col(df, "Close")
    if c is None or c.dropna().empty:
        c = _find_col(df, "Adj Close")
    if c is None or c.dropna().empty:
        return None
    idx = c.index
    def _align(x):
        return pd.to_numeric(x, errors="coerce").reindex(idx) if x is not None else None
    o = _align(o); h = _align(h); l = _align(l); c = _align(c).astype(float)
    if o is None: o = c.copy()
    if h is None: h = c.copy()
    if l is None: l = c.copy()
    out = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c}).dropna()
    return out if not out.empty else None

# ------------------ Technical helpers (ATR, swings, clustering) ------------------
def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h = df["High"]; l = df["Low"]; c = df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def swing_points(df: pd.DataFrame, w: int = 3):
    """Return indices and values of swing highs/lows (local extrema) for window w."""
    highs_idx, lows_idx = [], []
    h, l = df["High"], df["Low"]
    for i in range(w, len(df) - w):
        left_h = h.iloc[i - w:i].max(); right_h = h.iloc[i + 1:i + 1 + w].max()
        left_l = l.iloc[i - w:i].min(); right_l = l.iloc[i + 1:i + 1 + w].min()
        hi = h.iloc[i]; lo = l.iloc[i]
        if hi >= left_h and hi >= right_h:
            highs_idx.append(i)
        if lo <= left_l and lo <= right_l:
            lows_idx.append(i)
    highs = [(i, float(h.iloc[i])) for i in highs_idx]
    lows  = [(i, float(l.iloc[i])) for i in lows_idx]
    return highs, lows

def cluster_levels(levels, tol_abs):
    """Cluster 1D price levels into horizontal bands within ±tol_abs."""
    if not levels:
        return []
    levels = sorted(levels)
    clusters = [[levels[0]]]
    for lv in levels[1:]:
        if abs(lv - np.mean(clusters[-1])) <= tol_abs:
            clusters[-1].append(lv)
        else:
            clusters.append([lv])
    bands = []
    for cl in clusters:
        center = float(np.mean(cl))
        lo = float(min(cl)); hi = float(max(cl))
        bands.append({"center": center, "low": lo, "high": hi, "touches": len(cl)})
    return bands

def pick_sr_zones_from_4h(df4h: pd.DataFrame, min_touches=3, w=3, pct_tol=0.004, atr_len=14):
    """
    Build support/resistance zones from 4H swings.
    - Find swing highs/lows
    - Cluster by tolerance (ATR-anchored with pct fallback)
    - Keep clusters with >= min_touches
    - Return (support_zone, resistance_zone) as (low, high) rectangles
    """
    if df4h is None or df4h.empty:
        return None, None

    # swings
    highs, lows = swing_points(df4h, w=w)

    # raw levels (values only)
    hi_vals = [v for _, v in highs]
    lo_vals = [v for _, v in lows]

    # tolerance: ATR or pct of recent price
    atrv = atr(df4h, n=atr_len).iloc[-1]
    last = float(df4h["Close"].iloc[-1])
    tol_abs = float(max(atrv, last * pct_tol))

    # cluster into bands
    hi_bands = cluster_levels(hi_vals, tol_abs)
    lo_bands = cluster_levels(lo_vals, tol_abs)

    # filter by touches
    hi_bands = [b for b in hi_bands if b["touches"] >= min_touches]
    lo_bands = [b for b in lo_bands if b["touches"] >= min_touches]

    if not hi_bands and not lo_bands:
        return None, None

    # choose nearest significant zones around current price
    res_zone = None
    sup_zone = None
    # resistance: band center >= last, pick nearest
    candidates_r = sorted([b for b in hi_bands if b["center"] >= last], key=lambda x: x["center"])
    if candidates_r:
        b = candidates_r[0]
        # widen band slightly by tolerance for visualization
        res_zone = (max(b["low"], b["center"] - tol_abs), min(b["high"], b["center"] + tol_abs))
    # support: band center <= last, pick nearest
    candidates_s = sorted([b for b in lo_bands if b["center"] <= last], key=lambda x: x["center"], reverse=True)
    if candidates_s:
        b = candidates_s[0]
        sup_zone = (max(b["low"], b["center"] - tol_abs), min(b["high"], b["center"] + tol_abs))

    return sup_zone, res_zone

# ------------------ Data fetcher ------------------
def fetch_one(ticker):
    """
    - Daily or Weekly frame for rendering (TWD_TF)
    - 4H S/R zones via resampled 60m data (last ~FOURH_LOOKBACK_DAYS)
    - 30d % change from daily closes
    """
    tf = (os.getenv("TWD_TF", DEFAULT_TF) or DEFAULT_TF).upper().strip()  # 'D' or 'W'

    # --- Daily (1y) for change calc and possibly rendering
    try:
        df_d = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=True)
    except Exception:
        df_d = None
    if df_d is None or df_d.empty:
        df_d = yf.download(ticker, period="1y", interval="1d", auto_adjust=True)

    ohlc_d = _get_ohlc_df(df_d)
    if ohlc_d is None or ohlc_d.empty:
        return None

    close_d = ohlc_d["Close"]
    if close_d.dropna().shape[0] < 2:
        return None

    last = float(close_d.iloc[-1])
    last_date = close_d.index[-1]
    base_date = last_date - pd.Timedelta(days=30)
    base_series = close_d.loc[:base_date]
    if base_series.empty:
        base_val = float(close_d.iloc[0]); chg30 = 0.0
    else:
        base_val = float(base_series.iloc[-1])
        chg30 = float(100.0 * (float(close_d.iloc[-1]) - base_val) / base_val) if base_val != 0 else 0.0

    # --- 60m → 4H for S/R
    try:
        df_60 = yf.Ticker(ticker).history(period="6mo", interval="60m", auto_adjust=True)
    except Exception:
        df_60 = None
    if df_60 is None or df_60.empty:
        df_60 = yf.download(ticker, period="6mo", interval="60m", auto_adjust=True)

    ohlc_60 = _get_ohlc_df(df_60)
    sr_sup_zone, sr_res_zone = (None, None)
    if ohlc_60 is not None and not ohlc_60.empty:
        # Limit lookback
        cutoff = ohlc_60.index.max() - pd.Timedelta(days=FOURH_LOOKBACK_DAYS)
        df_60_clip = ohlc_60.loc[ohlc_60.index >= cutoff].copy()
        if not df_60_clip.empty:
            df_4h = df_60_clip.resample("4H").agg(
                {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
            ).dropna()
            if not df_4h.empty:
                sr_sup_zone, sr_res_zone = pick_sr_zones_from_4h(
                    df_4h, min_touches=MIN_TOUCHES, w=SWING_WINDOW, pct_tol=ZONE_PCT_TOL, atr_len=ATR_LEN
                )

    # fallback S/R if 4H detection failed
    if sr_sup_zone is None or sr_res_zone is None:
        if tf == "W":
            df_render = ohlc_d.resample("W-FRI").agg({"Open":"first","High":"max","Low":"min","Close":"last"}).dropna().tail(60)
            if df_render.empty: return None
            w_low, w_high = float(df_render["Low"].min()), float(df_render["High"].max())
            sup_low, sup_high = w_low, w_low * 1.03
            res_low, res_high = w_high * 0.97, w_high
        else:
            df_render = ohlc_d.dropna().tail(260)
            if df_render.empty: return None
            q15, q25 = np.nanquantile(df_render["Close"], [0.15, 0.25])
            q75, q85 = np.nanquantile(df_render["Close"], [0.75, 0.85])
            sup_low, sup_high = float(q15), float(q25)
            res_low, res_high = float(q75), float(q85)
    else:
        # Use 4H-detected zones
        sup_low, sup_high = sr_sup_zone if sr_sup_zone else (None, None)
        res_low, res_high = sr_res_zone if sr_res_zone else (None, None)
        # choose render frame
        if tf == "W":
            df_render = ohlc_d.resample("W-FRI").agg({"Open":"first","High":"max","Low":"min","Close":"last"}).dropna().tail(60)
            if df_render.empty: return None
        else:
            df_render = ohlc_d.dropna().tail(260)
            if df_render.empty: return None

    bos_dir = "up" if chg30 > 5 else ("down" if chg30 < -5 else None)
    bos_level = last
    bos_idx = int(len(df_render) - 1)
    tf_tag = "Weekly" if tf == "W" else "Daily"

    return (df_render, last, float(chg30),
            sup_low, sup_high, res_low, res_high,
            "Support", "Resistance", bos_dir, bos_level, bos_idx, ("W" if tf == "W" else "D"))

# ------------------ Renderer (white, even header spacing, no pill badges) ------------------
def render_single_post(out_path, ticker, payload):
    (df, last, chg30, sup_low, sup_high, res_low, res_high,
     sup_label, res_label, bos_dir, bos_level, bos_idx, bos_tf) = payload

    # Scales
    try:    S_LAYOUT = float(os.getenv("TWD_UI_SCALE", "0.90"))
    except: S_LAYOUT = 0.90
    try:    S_TEXT   = float(os.getenv("TWD_TEXT_SCALE", "0.70"))
    except: S_TEXT   = 0.70
    try:    S_LOGO   = float(os.getenv("TWD_TLOGO_SCALE", "0.55"))
    except: S_LOGO   = 0.55
    def sp(x: float) -> int: return int(round(x * S_LAYOUT))
    def st(x: float) -> int: return int(round(x * S_TEXT))

    # Theme
    W, H = 1080, 1080
    BG       = (255, 255, 255, 255)
    TEXT_DK  = (23, 23, 23, 255)
    TEXT_MD  = (55, 65, 81, 255)
    TEXT_LT  = (120, 128, 140, 255)
    GRID_MAJ = (232, 236, 240, 255)
    GRID_MIN = (242, 244, 247, 255)
    GREEN    = (22, 163, 74, 255)
    RED      = (239, 68, 68, 255)
    WICK     = (140, 140, 140, 255)
    SR_BLUE  = (120, 162, 255, 50)
    SR_RED   = (255, 154, 154, 50)
    SR_BLUE_ST = (120, 162, 255, 120)
    SR_RED_ST  = (255, 154, 154, 120)
    ACCENT   = (255, 210, 0, 255)

    base = Image.new("RGBA", (W, H), BG)
    draw = ImageDraw.Draw(base)

    # Fonts
    def _try_font(path, size):
        try:    return ImageFont.truetype(path, size)
        except: return None
    def _font(size, bold=False):
        sz = st(size)
        grift_bold = _try_font("assets/fonts/Grift-Bold.ttf", sz)
        grift_reg  = _try_font("assets/fonts/Grift-Regular.ttf", sz)
        roboto_bold = (_try_font("assets/fonts/Roboto-Bold.ttf", sz)
                       or _try_font("Roboto-Bold.ttf", sz))
        roboto_reg  = (_try_font("assets/fonts/Roboto-Regular.ttf", sz)
                       or _try_font("Roboto-Regular.ttf", sz))
        if bold: return grift_bold or roboto_bold or ImageFont.load_default()
        return grift_reg or roboto_reg or ImageFont.load_default()

    f_ticker = _font(92, bold=True)
    f_price  = _font(48, bold=True)
    f_delta  = _font(42, bold=True)
    f_sub    = _font(30)
    f_sm     = _font(26)
    f_axis   = _font(24)

    # Layout
    outer_top = sp(60)
    outer_lr  = sp(64)
    outer_bot = sp(60)
    header_h  = sp(200)
    footer_h  = sp(140)
    chart = [outer_lr, outer_top + header_h, W - outer_lr, H - outer_bot - footer_h]
    cx1, cy1, cx2, cy2 = chart

    # Header (equal vertical spacing)
    title_x, title_y = outer_lr, outer_top
    GAP = st(18)
    def draw_line(x, y, text, font, fill):
        draw.text((x, y), text, fill=fill, font=font)
        bbox = draw.textbbox((x, y), text, font=font)
        return y + (bbox[3] - bbox[1]) + GAP

    y_cursor = draw_line(title_x, title_y, ticker, f_ticker, TEXT_DK)
    y_cursor = draw_line(title_x, y_cursor, f"{last:,.2f} USD", f_price, TEXT_MD)
    delta_col = GREEN if chg30 >= 0 else RED
    y_cursor = draw_line(title_x, y_cursor, f"{chg30:+.2f}% past 30d", f_delta, delta_col)
    sub_label = "Daily chart • last ~1 year" if bos_tf == "D" else "Weekly chart • last 52 weeks"
    y_cursor = draw_line(title_x, y_cursor, sub_label, f_sub, TEXT_LT)

    # Ticker logo
    tlogo_path = os.path.join("assets", "logos", f"{ticker}.png")
    if os.path.exists(tlogo_path):
        try:
            tlogo = Image.open(tlogo_path).convert("RGBA")
            hmax = int(sp(86) * S_LOGO)
            hmax = max(1, hmax)
            scl = min(1.0, hmax / max(1, tlogo.height))
            tlogo = tlogo.resize((int(tlogo.width * scl), int(tlogo.height * scl)))
            base.alpha_composite(tlogo, (W - outer_lr - tlogo.width, title_y))
        except Exception:
            pass

    # Data for render
    df2 = df[["Open", "High", "Low", "Close"]].dropna()
    if df2.shape[0] < 2:
        out = base.convert("RGB"); os.makedirs(os.path.dirname(out_path), exist_ok=True); out.save(out_path, quality=95); return

    ymin = float(np.nanmin(df2["Low"])); ymax = float(np.nanmax(df2["High"]))
    if not np.isfinite(ymin) or not np.isfinite(ymax) or abs(ymax - ymin) < 1e-6:
        ymin, ymax = (ymin - 0.5, ymax + 0.5) if np.isfinite(ymin) else (0, 1)
    yr = ymax - ymin; ymin -= 0.02 * yr; ymax += 0.02 * yr

    def sx(i): return cx1 + (i / max(1, len(df2) - 1)) * (cx2 - cx1)
    def sy(v): return cy2 - ((float(v) - ymin) / (ymax - ymin)) * (cy2 - cy1)

    # Grid
    grid = Image.new("RGBA", (W, H), (0, 0, 0, 0)); g = ImageDraw.Draw(grid)
    for i in range(1, 7):
        y = cy1 + i * (cy2 - cy1) / 7.0; g.line([(cx1, y), (cx2, y)], fill=GRID_MIN, width=sp(1))
    for frac in (0.33, 0.66):
        y = cy1 + frac * (cy2 - cy1); g.line([(cx1, y), (cx2, y)], fill=GRID_MAJ, width=sp(1))
    for i in range(1, 9):
        x = cx1 + i * (cx2 - cx1) / 9.0; g.line([(x, cy1), (x, cy2)], fill=GRID_MIN, width=sp(1))
    base = Image.alpha_composite(base, grid); draw = ImageDraw.Draw(base)

    # S/R zones from 4H detection (if available)
    if sup_low is not None and sup_high is not None:
        sup_y1, sup_y2 = sy(sup_high), sy(sup_low)
        sup_rect = [cx1, min(sup_y1, sup_y2), cx2, max(sup_y1, sup_y2)]
        sup_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(sup_layer).rectangle(sup_rect, fill=SR_BLUE, outline=SR_BLUE_ST, width=sp(2))
        base = Image.alpha_composite(base, sup_layer); draw = ImageDraw.Draw(base)
        draw.text((cx2 - sp(220), min(sup_y1, sup_y2) + sp(6)),
                  f"{sup_label} ~{sup_low:.2f}", fill=(65, 90, 140, 255), font=f_sm)

    if res_low is not None and res_high is not None:
        res_y1, res_y2 = sy(res_high), sy(res_low)
        res_rect = [cx1, min(res_y1, res_y2), cx2, max(res_y1, res_y2)]
        res_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(res_layer).rectangle(res_rect, fill=SR_RED, outline=SR_RED_ST, width=sp(2))
        base = Image.alpha_composite(base, res_layer); draw = ImageDraw.Draw(base)
        draw.text((cx2 - sp(220), min(res_y1, res_y2) + sp(6)),
                  f"{res_label} ~{res_high:.2f}", fill=(150, 60, 60, 255), font=f_sm)

    # Candles
    n = len(df2)
    base_body_px = max(2, int((cx2 - cx1) / max(260, n * 1.1)))
    body_px = max(1, int(round(base_body_px * S_LAYOUT)))
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

    # BOS line
    if bos_dir is not None and np.isfinite(bos_level):
        by = sy(bos_level); draw.line([(cx1, by), (cx2, by)], fill=ACCENT, width=sp(3))

    # Right axis ticks
    ticks = np.linspace(ymin, ymax, 5)
    for tval in ticks:
        y = sy(tval); label = f"{tval:,.2f}"
        bbox = draw.textbbox((0, 0), label, font=f_axis)
        tw, th = (bbox[2]-bbox[0], bbox[3]-bbox[1])
        draw.text((cx2 + sp(8), y - th/2), label, fill=TEXT_LT, font=f_axis)

    # Footer
    meta_x = outer_lr; meta_y = H - outer_bot - st(64)
    if res_high is not None: draw.text((meta_x, meta_y),          f"Resistance ~{res_high:.2f}", fill=TEXT_LT, font=f_sm)
    if sup_low  is not None: draw.text((meta_x, meta_y + st(24)), f"Support ~{sup_low:.2f}",     fill=TEXT_LT, font=f_sm)
    draw.text((meta_x, meta_y + st(48)), "Not financial advice",  fill=(160,160,160,255), font=f_sm)

    # Brand logo
    logo_path = BRAND_LOGO_PATH or "assets/brand_logo.png"
    if logo_path and os.path.exists(logo_path):
        try:
            blogo = Image.open(logo_path).convert("RGBA")
            maxh = sp(110)
            scl = min(1.0, max(1, maxh) / max(1, blogo.height))
            blogo = blogo.resize((int(blogo.width * scl), int(blogo.height * scl)))
            base.alpha_composite(blogo, (W - outer_lr - blogo.width, H - outer_bot - blogo.height))
        except Exception:
            pass

    out = base.convert("RGB")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.save(out_path, quality=95)

# ------------------ Captions ------------------
def plain_english_line(ticker, headline, payload, seed=None):
    (df,last,chg30,sup_low,sup_high,res_low,res_high,
     sup_label,res_label,bos_dir,bos_level,bos_idx,bos_tf) = payload

    if seed is None: seed = f"{ticker}-{DATESTR}"
    rnd = random.Random(str(seed))

    if headline and len(headline) > 160: headline = headline[:157] + "…"
    lead = headline or rnd.choice(["No major headlines today.","Quiet on the news front.","News flow is light."])

    cues = []
    if chg30 >= 8: cues.append("momentum looks strong 🔥")
    elif chg30 <= -8: cues.append("recent pullback showing ⚠️")
    if bos_dir == "up": cues.append("breakout pressure building 🚀")
    if bos_dir == "down": cues.append("post-breakdown chop ⚠️")
    if not cues:
        cues = rnd.sample([
            "price action is steady", "range bound but coiling",
            "watching for a decisive move soon", "tightening ranges"
        ], k=1)
    cue_txt = "; ".join(cues)
    return f"📈 {ticker} — {lead} — {cue_txt}"[:280]

CTA_POOL = [
    "Save for later 📌 · Comment your levels 💬 · See charts in carousel ➡️",
    "Tap save 📌 · Drop your take below 💬 · Full charts in carousel ➡️",
    "Save this post 📌 · Share your view 💬 · Swipe for charts ➡️"
]

# ------------------ Main ------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tickers = choose_tickers_somehow()
    print("[info] selected tickers:", tickers)

    saved, captions = 0, []
    for t in tickers:
        try:
            payload = fetch_one(t)
            print(f"[debug] fetched {t}: payload is {'ok' if payload else 'None'}")
            if not payload:
                print(f"[warn] no data for {t}, skipping"); continue

            out_path = os.path.join(OUTPUT_DIR, f"twd_{t}_{DATESTR}.png")
            print(f"[debug] saving {out_path}")
            render_single_post(out_path, t, payload)
            print("done:", out_path); saved += 1

            headline = news_headline_for(t)
            captions.append(plain_english_line(t, headline, payload, seed=DATESTR))

        except Exception as e:
            print(f"Error: failed for {t}: {e}")
            traceback.print_exc()

    print(f"[info] saved images: {saved}")

    if saved > 0:
        caption_path = os.path.join(OUTPUT_DIR, f"caption_${DATESTR}.txt".replace("$","") )
        now_str = TODAY.strftime("%d %b %Y")
        footer = f"\n\n{random.choice(CTA_POOL)}\n\nIdeas only — not financial advice"
        with open(caption_path, "w", encoding="utf-8") as f:
            f.write(f"Ones to Watch – {now_str}\n\n")
            f.write("\n\n".join(captions))
            f.write(footer)
        print("[info] wrote caption:", caption_path)

if __name__ == "__main__":
    main()
