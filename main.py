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

# Render frame: 'D' daily (last ~1y) or 'W' weekly (last ~60 bars)
DEFAULT_TF = (os.getenv("TWD_TF", "D") or "D").upper()

# 4H S/R + BOS detection settings
FOURH_LOOKBACK_DAYS = int(os.getenv("TWD_4H_LOOKBACK_DAYS", "120"))
SWING_WINDOW = int(os.getenv("TWD_SWING_WINDOW", "3"))           # 3–5 typical
ATR_LEN = int(os.getenv("TWD_ATR_LEN", "14"))
ZONE_PCT_TOL = float(os.getenv("TWD_ZONE_PCT_TOL", "0.004"))     # 0.4% band thickness

# BOS toggle (1 on, 0 off)
SHOW_BOS = os.getenv("TWD_SHOW_BOS", "1") == "1"

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
    if name in df.columns:
        ser = df[name]
        if isinstance(ser, pd.DataFrame):
            ser = ser.iloc[:, 0]
        return pd.to_numeric(ser, errors="coerce")
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

# ------------------ Technical helpers ------------------
def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h = df["High"]; l = df["Low"]; c = df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def swing_points(df: pd.DataFrame, w: int = 3):
    """Return swing highs and lows as lists of tuples (index_position, value)."""
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

# ----- Support selection per your rule -----
def pick_support_level_from_4h(df4h: pd.DataFrame, trend_bullish: bool, w: int, pct_tol: float, atr_len: int):
    """
    If bullish: support = previous swing HIGH below current price, closest to price.
    If bearish: support = last swing LOW (most recent low before last bar).
    Returns (sup_low, sup_high) using a tolerance band (max(ATR, pct)).
    """
    if df4h is None or df4h.empty:
        return (None, None)

    highs, lows = swing_points(df4h, w=w)
    last_px = float(df4h["Close"].iloc[-1])
    atrv = float(atr(df4h, n=atr_len).iloc[-1]) if len(df4h) > 1 else 0.0
    tol_abs = max(atrv, last_px * pct_tol)

    if trend_bullish:
        candidates = [(i, v) for (i, v) in highs if v <= last_px]
        if not candidates:
            return (None, None)
        i_sel, v_sel = sorted(candidates, key=lambda t: (abs(last_px - t[1]), -t[0]))[0]
        level = float(v_sel)
    else:
        prior_lows = [(i, v) for (i, v) in lows if i < len(df4h) - 1]
        if not prior_lows:
            return (None, None)
        i_sel, v_sel = max(prior_lows, key=lambda t: t[0])
        level = float(v_sel)

    return (float(level - tol_abs), float(level + tol_abs))

# ----- BOS detection per your definition on 4H -----
def detect_bos_from_4h(df4h: pd.DataFrame, chg30: float, w: int):
    """
    Bullish BOS (in uptrend): last close > most recent swing high.
    Bearish BOS (in downtrend): last close < most recent swing low.
    Returns (bos_dir, bos_level) where bos_level is the broken swing price.
    """
    if df4h is None or df4h.empty:
        return (None, np.nan)
    highs, lows = swing_points(df4h, w=w)
    if not highs and not lows:
        return (None, np.nan)

    last_c = float(df4h["Close"].iloc[-1])

    if chg30 > 0 and highs:
        # most recent swing high before last bar
        i_h, v_h = max(highs, key=lambda t: t[0])
        if last_c > v_h:
            return ("up", float(v_h))
    elif chg30 < 0 and lows:
        i_l, v_l = max(lows, key=lambda t: t[0])
        if last_c < v_l:
            return ("down", float(v_l))

    return (None, np.nan)

# ------------------ Data fetcher ------------------
def fetch_one(ticker):
    """
    - Render on Daily/Weekly (env TWD_TF)
    - Support from 4H per rule (bullish: prior swing high; bearish: last swing low)
    - BOS from 4H per rule
    - 30d % change from daily closes for trend decision
    """
    tf = (os.getenv("TWD_TF", DEFAULT_TF) or DEFAULT_TF).upper().strip()  # 'D' or 'W'

    # Daily 1y for change calc (and maybe render)
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

    # 60m → 4H for support selection and BOS detection
    try:
        df_60 = yf.Ticker(ticker).history(period="6mo", interval="60m", auto_adjust=True)
    except Exception:
        df_60 = None
    if df_60 is None or df_60.empty:
        df_60 = yf.download(ticker, period="6mo", interval="60m", auto_adjust=True)

    sup_low, sup_high = (None, None)
    bos_dir, bos_level = (None, np.nan)

    if df_60 is not None and not df_60.empty:
        ohlc_60 = _get_ohlc_df(df_60)
        if ohlc_60 is not None and not ohlc_60.empty:
            cutoff = ohlc_60.index.max() - pd.Timedelta(days=FOURH_LOOKBACK_DAYS)
            df_60_clip = ohlc_60.loc[ohlc_60.index >= cutoff].copy()
            if not df_60_clip.empty:
                df_4h = df_60_clip.resample("4H").agg(
                    {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
                ).dropna()
                if not df_4h.empty:
                    trend_bullish = (chg30 > 0)
                    sup_low, sup_high = pick_support_level_from_4h(
                        df_4h, trend_bullish=trend_bullish, w=SWING_WINDOW,
                        pct_tol=ZONE_PCT_TOL, atr_len=ATR_LEN
                    )
                    bos_dir, bos_level = detect_bos_from_4h(df_4h, chg30=chg30, w=SWING_WINDOW)

    # choose render frame
    if tf == "W":
        df_render = ohlc_d.resample("W-FRI").agg(
            {"Open":"first","High":"max","Low":"min","Close":"last"}
        ).dropna().tail(60)
        if df_render.empty: return None
        tf_tag = "W"
    else:
        df_render = ohlc_d.dropna().tail(260)
        if df_render.empty: return None
        tf_tag = "D"

    bos_idx = int(len(df_render) - 1)

    # Only support values used in renderer; resistance omitted per your rule
    return (df_render, last, float(chg30),
            sup_low, sup_high, None, None,
            "Support", "", bos_dir, bos_level, bos_idx, tf_tag)

# ------------------ Renderer (white, single support zone, BOS line optional) ------------------
def render_single_post(out_path, ticker, payload):
    (df, last, chg30, sup_low, sup_high, _res_low, _res_high,
     _sup_label, _res_label, bos_dir, bos_level, bos_idx, bos_tf) = payload

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
    SR_BLUE  = (120, 162, 255, 50)   # support fill
    SR_BLUE_ST = (120, 162, 255, 120)
    BOS_UP   = (22, 163, 74, 255)    # green
    BOS_DN   = (239, 68, 68, 255)    # red

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
        roboto_bold = (_try_font("assets/fonts/Roboto-Bold.ttf", sz) or _try_font("Roboto-Bold.ttf", sz))
        roboto_reg  = (_try_font("assets/fonts/Roboto-Regular.ttf", sz) or _try_font("Roboto-Regular.ttf", sz))
        if bold: return grift_bold or roboto_bold or ImageFont.load_default()
        return grift_reg or roboto_reg or ImageFont.load_default()

    f_ticker = _font(92, bold=True)
    f_price  = _font(48, bold=True)
    f_delta  = _font(42, bold=True)
    f_sub    = _font(30)
    f_sm     = _font(26)
    f_axis   = _font(24)

    # Layout (breathing room + smaller chart box)
    outer_top = sp(60); outer_lr = sp(64); outer_bot = sp(60)
    header_h = sp(200); footer_h = sp(140)
    chart = [outer_lr, outer_top + header_h, W - outer_lr, H - outer_bot - footer_h]
    cx1, cy1, cx2, cy2 = chart

    # Header with equal spacing
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

    # Ticker logo (top-right)
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

    # ----- Support zone ONLY (blue), NO text label -----
    if sup_low is not None and sup_high is not None and np.isfinite(sup_low) and np.isfinite(sup_high):
        sup_y1, sup_y2 = sy(sup_high), sy(sup_low)
        sup_rect = [cx1, min(sup_y1, sup_y2), cx2, max(sup_y1, sup_y2)]
        sup_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(sup_layer).rectangle(sup_rect, fill=SR_BLUE, outline=SR_BLUE_ST, width=sp(2))
        base = Image.alpha_composite(base, sup_layer)
        draw = ImageDraw.Draw(base)

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

    # ----- BOS line (true BOS per your definition; toggle via env) -----
    if SHOW_BOS and bos_dir is not None and np.isfinite(bos_level):
        by = sy(bos_level)
        draw.line([(cx1, by), (cx2, by)], fill=(BOS_UP if bos_dir == "up" else BOS_DN), width=sp(3))

    # Right axis ticks
    ticks = np.linspace(ymin, ymax, 5)
    for tval in ticks:
        y = sy(tval); label = f"{tval:,.2f}"
        bbox = draw.textbbox((0, 0), label, font=f_axis)
        tw, th = (bbox[2]-bbox[0], bbox[3]-bbox[1])
        draw.text((cx2 + sp(8), y - th/2), label, fill=TEXT_LT, font=f_axis)

    # Footer
    meta_x = outer_lr; meta_y = H - outer_bot - st(64)
    draw.text((meta_x, meta_y),          "Support zone highlighted", fill=TEXT_LT, font=f_sm)
    draw.text((meta_x, meta_y + st(24)), "Not financial advice",    fill=(160,160,160,255), font=f_sm)

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
    (df,last,chg30,sup_low,sup_high,_res_low,_res_high,
     _sup_label,_res_label,bos_dir,bos_level,bos_idx,bos_tf) = payload

    if seed is None: seed = f"{ticker}-{DATESTR}"
    rnd = random.Random(str(seed))

    if headline and len(headline) > 160: headline = headline[:157] + "…"
    lead = headline or rnd.choice(["No major headlines today.","Quiet on the news front.","News flow is light."])

    if chg30 > 0:
        cue = "trend constructive — prior swing high mapped as support 🔎"
    elif chg30 < 0:
        cue = "tone cautious — last swing low in focus for a bounce 👀"
    else:
        cue = "neutral tone — watching key support 🎯"

    if bos_dir == "up":
        bos_txt = " (bullish BOS confirmed)"
    elif bos_dir == "down":
        bos_txt = " (bearish BOS confirmed)"
    else:
        bos_txt = ""

    return f"📈 {ticker} — {lead} — {cue}{bos_txt}"[:280]

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
        caption_path = os.path.join(OUTPUT_DIR, f"caption_{DATESTR}.txt")
        now_str = TODAY.strftime("%d %b %Y")
        footer = f"\n\n{random.choice(CTA_POOL)}\n\nIdeas only — not financial advice"
        with open(caption_path, "w", encoding="utf-8") as f:
            f.write(f"Ones to Watch – {now_str}\n\n")
            f.write("\n\n".join(captions))
            f.write(footer)
        print("[info] wrote caption:", caption_path)

if __name__ == "__main__":
    main()
