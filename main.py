#!/usr/bin/env python3
import os, random, datetime, traceback, re
import numpy as np
import pandas as pd
import yfinance as yf
from PIL import Image, ImageDraw, ImageFont
import requests

# ================== Config ==================
OUTPUT_DIR = os.path.abspath("output")
TODAY = datetime.date.today()
DATESTR = TODAY.strftime("%Y%m%d")

# News (optional)
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "").strip()
SESS = requests.Session()
SESS.headers.update({"User-Agent": "TWD/1.0"})

# Logos
BRAND_LOGO_PATH = os.getenv("BRAND_LOGO_PATH", "assets/brand_logo.png")
BRAND_MAXH = int(os.getenv("TWD_BRAND_MAXH", "160"))     # make bigger by raising this
BRAND_MARGIN = int(os.getenv("TWD_BRAND_MARGIN", "60"))  # padding from edges

# Render frame: 'D' daily (last ~1y) or 'W' weekly (last ~60 bars)
DEFAULT_TF = (os.getenv("TWD_TF", "D") or "D").upper()

# 4H support detection
FOURH_LOOKBACK_DAYS = int(os.getenv("TWD_4H_LOOKBACK_DAYS", "120"))
SWING_WINDOW = int(os.getenv("TWD_SWING_WINDOW", "3"))
ATR_LEN = int(os.getenv("TWD_ATR_LEN", "14"))
ZONE_PCT_TOL = float(os.getenv("TWD_ZONE_PCT_TOL", "0.004"))  # 0.4%

# UI scaling knobs
TWD_UI_SCALE   = float(os.getenv("TWD_UI_SCALE", "0.90"))   # layout (chart box) scale
TWD_TEXT_SCALE = float(os.getenv("TWD_TEXT_SCALE", "0.70")) # text scale
TWD_TLOGO_SCALE= float(os.getenv("TWD_TLOGO_SCALE","0.55")) # ticker-logo scale

# ================== Pools ==================
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

# ================== News ==================
def news_headline_for(ticker):
    name = COMPANY_QUERY.get(ticker, ticker)
    # Try NewsAPI
    if NEWSAPI_KEY:
        try:
            r = SESS.get(
                "https://newsapi.org/v2/everything",
                params={"q": f'"{name}" OR {ticker}', "language":"en", "sortBy":"publishedAt", "pageSize":1},
                headers={"X-Api-Key": NEWSAPI_KEY}, timeout=8
            )
            if r.ok:
                d = r.json()
                if d.get("articles"):
                    a = d["articles"][0]
                    title = a.get("title") or ""
                    src = a.get("source", {}).get("name","")
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
    """Return swing highs/lows as [(i, price), ...]"""
    highs, lows = [], []
    h, l = df["High"], df["Low"]
    for i in range(w, len(df)-w):
        left_h = h.iloc[i-w:i].max(); right_h = h.iloc[i+1:i+1+w].max()
        left_l = l.iloc[i-w:i].min(); right_l = l.iloc[i+1:i+1+w].min()
        if h.iloc[i] >= left_h and h.iloc[i] >= right_h: highs.append((i, float(h.iloc[i])))
        if l.iloc[i] <= left_l and l.iloc[i] <= right_l: lows.append((i, float(l.iloc[i])))
    return highs, lows

def pick_support_level_from_4h(df4h, trend_bullish, w, pct_tol, atr_len):
    """
    If bullish: support = previous swing HIGH below current price (closest).
    If bearish: support = last swing LOW (most recent).
    Returns (sup_low, sup_high) using band width = max(ATR, pct_tol*price).
    """
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
        i_sel, v_sel = max(lows, key=lambda t: t[0])  # most recent low
        level = float(v_sel)

    return (level - tol_abs, level + tol_abs)

# ================== Fetcher ==================
def fetch_one(ticker):
    """Build payload: (df_render, last, chg30, sup_low, sup_high, tf_tag)"""
    tf = (os.getenv("TWD_TF", DEFAULT_TF) or DEFAULT_TF).upper().strip()

    # Daily 1y
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
    if close_d.shape[0] > 30:
        base_val = float(close_d.iloc[-31])  # ~30 sessions ago
    else:
        base_val = float(close_d.iloc[0])
    chg30 = 100.0 * (last - base_val) / base_val if base_val != 0 else 0.0

    # Build 4H from 60m for support
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

    # Choose render frame
    if tf == "W":
        df_render = ohlc_d.resample("W-FRI").agg(
            {"Open":"first","High":"max","Low":"min","Close":"last"}
        ).dropna().tail(60)
        tf_tag = "W"
    else:
        df_render = ohlc_d.dropna().tail(260)  # ~1y daily
        tf_tag = "D"
    if df_render is None or df_render.empty: return None

    return (df_render, last, float(chg30), sup_low, sup_high, tf_tag)

# ================== Renderer ==================
def render_single_post(out_path, ticker, payload):
    (df, last, chg30, sup_low, sup_high, tf_tag) = payload

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

    f_ticker = _font(88, True)
    f_price  = _font(44, True)
    f_delta  = _font(40, True)
    f_sub    = _font(28)
    f_axis   = _font(24)

    # ---- layout ----
    outer_top = sp(56); outer_lr = sp(64); outer_bot = sp(56)
    header_h = sp(190); footer_h = sp(130)
    cx1, cy1, cx2, cy2 = (
        outer_lr,
        outer_top + header_h,
        W - outer_lr,
        H - outer_bot - footer_h
    )

    # ---- header (equal spacing) ----
    title_x, title_y = outer_lr, outer_top
    GAP = st(16)

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
            hmax = int(sp(86) * TWD_TLOGO_SCALE)
            hmax = max(1, hmax)
            scl = min(1.0, hmax / max(1, tlogo.height))
            tlogo = tlogo.resize((int(tlogo.width * scl), int(tlogo.height * scl)))
            base.alpha_composite(tlogo, (W - outer_lr - tlogo.width, title_y))
        except Exception:
            pass

    # ---- data ----
    df2 = df[["Open","High","Low","Close"]].dropna()
    if df2.shape[0] < 2:
        out = base.convert("RGB"); os.makedirs(os.path.dirname(out_path), exist_ok=True); out.save(out_path, quality=95); return

    ymin = float(np.nanmin(df2["Low"])); ymax = float(np.nanmax(df2["High"]))
    if not np.isfinite(ymin) or not np.isfinite(ymax) or abs(ymax - ymin) < 1e-6:
        ymin, ymax = (ymin - 0.5, ymax + 0.5) if np.isfinite(ymin) else (0, 1)
    yr = ymax - ymin; ymin -= 0.02*yr; ymax += 0.02*yr

    def sx(i): return cx1 + (i / max(1, len(df2)-1)) * (cx2 - cx1)
    def sy(v): return cy2 - ((float(v) - ymin) / (ymax - ymin)) * (cy2 - cy1)

    # ---- grid (light) ----
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

    # ---- support zone (one blue box, no label) ----
    if sup_low is not None and sup_high is not None and np.isfinite(sup_low) and np.isfinite(sup_high):
        sup_y1, sup_y2 = sy(sup_high), sy(sup_low)
        sup_rect = [cx1, min(sup_y1, sup_y2), cx2, max(sup_y1, sup_y2)]
        sup_layer = Image.new("RGBA", (W, H), (0,0,0,0))
        ImageDraw.Draw(sup_layer).rectangle(sup_rect, fill=SR_FILL, outline=SR_STROK, width=sp(2))
        base = Image.alpha_composite(base, sup_layer)
        draw = ImageDraw.Draw(base)

    # ---- candles (slim, tidy) ----
    n = len(df2)
    body_px = max(1, int(((cx2 - cx1) / max(260, n * 1.05)) * TWD_UI_SCALE))
    half = max(1, body_px // 2)
    wick_w = max(1, sp(1))

    GREEN = (22,163,74,255)
    RED   = (239, 68,68,255)
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
    foot_x = outer_lr; foot_y = H - outer_bot - st(60)
    draw.text((foot_x, foot_y), "Support zone highlighted", fill=TEXT_LT, font=f_sub)
    draw.text((foot_x, foot_y + st(24)), "Not financial advice", fill=(160,160,160,255), font=f_sub)

    # ---- brand logo (bottom-right, scalable, keeps aspect ratio) ----
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

# ================== Captions (new, natural, varied, MS PT-aware) ==================
_SECTOR_EMOJIS = {
    "TECH": ["🖥️","🔎","🧠","💻","📡"],
    "HEALTH": ["💊","🧬","⚕️","🩺"],
    "FINANCE": ["💳","🏦","📈","📊"],
    "SEMIS": ["🔌","⚡","🔧","🧮"],
    "GENERIC": ["📈","🔎","⚡","🚀"],
}
def _emoji_for(ticker, rnd):
    if ticker in ["AAPL","MSFT","META","GOOG","GOOGL"]: pool = _SECTOR_EMOJIS["TECH"]
    elif ticker in ["JNJ","UNH"]: pool = _SECTOR_EMOJIS["HEALTH"]
    elif ticker in ["MA","V","PYPL","SQ"]: pool = _SECTOR_EMOJIS["FINANCE"]
    elif ticker in ["AMD","NVDA","TSM"]: pool = _SECTOR_EMOJIS["SEMIS"]
    else: pool = _SECTOR_EMOJIS["GENERIC"]
    return rnd.choice(pool)

_MS_PAT = re.compile(r"(morgan\s+stanley)", re.IGNORECASE)
_PT_PAT = re.compile(r"(price\s*target|pt)\s*(to|at|of|:)?\s*\$?(\d{2,4})", re.IGNORECASE)

def caption_line(ticker, headline, payload, seed=None):
    """
    Build natural, varied caption lines.
    Format:
      • [emoji] TICKER — [news phrase] · [chart cues joined by ;]
    Adds Morgan Stanley PT mention only if present in headline (safe).
    """
    (_, last, chg30, sup_low, sup_high, tf_tag) = payload
    rnd = random.Random((seed or DATESTR) + ticker)

    # Emoji
    emoji = _emoji_for(ticker, rnd)

    # News phrase (varied)
    news_leads = [
        "Latest: “{h}”", "Fresh headlines: “{h}”", "In the news: “{h}”",
        "With {h}", "Headline: “{h}”"
    ]
    no_news = [
        "News flow is light.", "Quiet session on headlines.", "Few fresh headlines."
    ]
    ms_hooks = [
        "Morgan Stanley highlights a price target near ${pt} 🎯",
        "Morgan Stanley PT around ${pt} noted 📌",
        "MS price target ~${pt} back in focus 🔎"
    ]

    # Trim headline
    h = (headline or "").strip()
    if h:
        if len(h) > 80: h = h[:77] + "…"
        news_phrase = rnd.choice(news_leads).format(h=h)
        # If the headline itself mentions Morgan Stanley + a PT, surface it
        pt_txt = ""
        if _MS_PAT.search(h):
            m = _PT_PAT.search(h)
            if m:
                pt_val = m.group(3)
                pt_txt = " — " + rnd.choice(ms_hooks).format(pt=pt_val)
        news_part = f"{news_phrase}{pt_txt}"
    else:
        news_part = rnd.choice(no_news)

    # Chart cues (natural, non-repetitive)
    cues = []
    if chg30 >= 8:
        cues.append(rnd.choice(["momentum looks strong 🔥","breakout pressure building 🚀","uptrend intact ✅"]))
    elif chg30 >= 2:
        cues.append(rnd.choice(["constructive tone 📈","buyers stepping in 🛒","gradual strength ✅"]))
    elif chg30 <= -8:
        cues.append(rnd.choice(["recent pullback showing ⚠️","bearish lean 🐻","sellers pressing here 🧱"]))
    else:
        cues.append(rnd.choice(["price action is steady","range-bound but coiling","neutral bias—let price confirm 🎯"]))

    if sup_low is not None and sup_high is not None:
        cues.append(rnd.choice([
            "buyers defended support 🛡️",
            "support zone in play 📍",
            "watch reactions near support 👀"
        ]))

    # Limit to 2–3 cues, shuffled
    rnd.shuffle(cues)
    cues = cues[: rnd.choice([2,2,3]) ]
    cue_part = "; ".join(cues)

    return f"• {emoji} {ticker} — {news_part} · {cue_part}"

# Legacy convenience wrapper (kept name used in main)
def plain_english_line(ticker, headline, payload, seed=None):
    return caption_line(ticker, headline, payload, seed=seed)

CTA_POOL = [
    "Save for later 📌 · Comment your levels 💬 · See charts in carousel ➡️",
    "Tap save 📌 · Drop your take below 💬 · Full charts in carousel ➡️",
    "Save this post 📌 · Share your view 💬 · Swipe for charts ➡️",
]

# ================== Main ==================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tickers = choose_tickers_somehow()
    print("[info] selected tickers:", tickers)

    saved = 0
    captions = []
    for t in tickers:
        try:
            payload = fetch_one(t)
            if not payload:
                print(f"[warn] no data for {t}, skipping"); continue

            out_path = os.path.join(OUTPUT_DIR, f"twd_{t}_{DATESTR}.png")
            render_single_post(out_path, t, payload)
            saved += 1

            headline = news_headline_for(t)
            line = plain_english_line(t, headline, payload, seed=DATESTR)
            captions.append(line)

        except Exception as e:
            print(f"Error: failed for {t}: {e}")
            traceback.print_exc()

    print(f"[info] saved images: {saved}")

    if saved > 0:
        caption_path = os.path.join(OUTPUT_DIR, f"caption_{DATESTR}.txt")
        now_str = TODAY.strftime("%d %b %Y")
        header = f"Ones to Watch – {now_str}\n\n"
        footer = f"\n\n{random.choice(CTA_POOL)}\n\nIdeas only — not financial advice"
        with open(caption_path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write("\n\n".join(captions))
            f.write(footer)
        print("[info] wrote caption:", caption_path)

if __name__ == "__main__":
    main()
