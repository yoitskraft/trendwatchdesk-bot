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
    """
    Deterministic daily pick of 6 tickers from the pool.
    Seeded by DATESTR so Mon/Wed/Fri differ while staying reproducible per day.
    """
    pool = list(COMPANY_QUERY.keys())
    rnd = random.Random(DATESTR)
    k = min(6, len(pool))
    return rnd.sample(pool, k)

# ------------------ Robust OHLC helpers ------------------
def _find_col(df: pd.DataFrame, name: str):
    """
    Return a 1D numeric Series for 'Open'/'High'/'Low'/'Close'/'Adj Close'
    from either single-level or MultiIndex yfinance frames. Returns None if not found.
    """
    if df is None or df.empty:
        return None

    # Direct single-level match
    if name in df.columns:
        ser = df[name]
        if isinstance(ser, pd.DataFrame):
            ser = ser.iloc[:, 0]
        return pd.to_numeric(ser, errors="coerce")

    # Normalized-name match (lower, no spaces)
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

    # MultiIndex match
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
    """
    Build a clean OHLC DataFrame with simple columns ['Open','High','Low','Close'].
    If some fields are missing, fill from Close (best-effort).
    """
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

    o = _align(o)
    h = _align(h)
    l = _align(l)
    c = _align(c).astype(float)

    if o is None: o = c.copy()
    if h is None: h = c.copy()
    if l is None: l = c.copy()

    out = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c}).dropna()
    if out.empty:
        return None
    return out

# ------------------ Data fetcher (daily → weekly via env TWD_TF) ------------------
def fetch_one(ticker):
    """
    Load 1y of daily (auto_adjust), compute 30d % change.
    TWD_TF='D' (default): daily last ~1y (~220-260 bars)
    TWD_TF='W': weekly last ~52-60 weeks
    """
    tf = (os.getenv("TWD_TF", "D") or "D").upper().strip()  # 'D' or 'W'

    # Try history() first (cleaner), fallback to download()
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

    # 30d change on daily series
    last_date = close_d.index[-1]
    base_date = last_date - pd.Timedelta(days=30)
    base_series = close_d.loc[:base_date]
    if base_series.empty:
        base_val = float(close_d.iloc[0])
        chg30 = 0.0
    else:
        base_val = float(base_series.iloc[-1])
        chg30 = float(100.0 * (float(close_d.iloc[-1]) - base_val) / base_val) if base_val != 0 else 0.0
    last = float(close_d.iloc[-1])

    if tf == "W":
        # Weekly
        df_w = ohlc_d.resample("W-FRI").agg({"Open":"first","High":"max","Low":"min","Close":"last"}).dropna()
        df_w = df_w.tail(60).dropna()
        if df_w.empty:
            return None
        w_low  = float(df_w["Low"].min())
        w_high = float(df_w["High"].max())
        sup_low, sup_high = w_low, w_low * 1.03
        res_high, res_low = w_high, w_high * 0.97
        bos_dir = "up" if chg30 > 5 else ("down" if chg30 < -5 else None)
        bos_level = last
        bos_idx = int(len(df_w) - 1)
        return (df_w, last, float(chg30), sup_low, sup_high, res_low, res_high,
                "Support", "Resistance", bos_dir, bos_level, bos_idx, "W")

    # Daily (default)
    df_d1 = ohlc_d.dropna().tail(260)  # ~1 trading year
    if df_d1.empty:
        return None

    # SR via quantiles (cleaner than extremes)
    q15, q25 = np.nanquantile(df_d1["Close"], [0.15, 0.25])
    q75, q85 = np.nanquantile(df_d1["Close"], [0.75, 0.85])
    sup_low, sup_high = float(q15), float(q25)
    res_low, res_high = float(q75), float(q85)

    bos_dir = "up" if chg30 > 5 else ("down" if chg30 < -5 else None)
    bos_level = last
    bos_idx = int(len(df_d1) - 1)

    return (df_d1, last, float(chg30), sup_low, sup_high, res_low, res_high,
            "Support", "Resistance", bos_dir, bos_level, bos_idx, "D")

# ------------------ Renderer (clean white, no pill/badges) ------------------
def render_single_post(out_path, ticker, payload):
    """
    Clean white background, improved header alignment:
      - Left header (ticker, price, % change, timeframe) uses equal vertical spacing
      - S/R shaded zones with plain labels (no rounded badges)
      - Right-side price ticks
      - Independent scales:
          TWD_UI_SCALE   : layout/chart/brand (default 0.90)
          TWD_TEXT_SCALE : text               (default 0.70)
          TWD_TLOGO_SCALE: ticker logo        (default 0.55)
    """
    (df, last, chg30, sup_low, sup_high, res_low, res_high,
     sup_label, res_label, bos_dir, bos_level, bos_idx, bos_tf) = payload

    # ---------- scales ----------
    try:    S_LAYOUT = float(os.getenv("TWD_UI_SCALE", "0.90"))
    except: S_LAYOUT = 0.90
    try:    S_TEXT   = float(os.getenv("TWD_TEXT_SCALE", "0.70"))
    except: S_TEXT   = 0.70
    try:    S_LOGO   = float(os.getenv("TWD_TLOGO_SCALE", "0.55"))
    except: S_LOGO   = 0.55

    def sp(x: float) -> int:  # layout/geometry
        return int(round(x * S_LAYOUT))
    def st(x: float) -> int:  # text
        return int(round(x * S_TEXT))

    # ---------- theme ----------
    W, H = 1080, 1080
    BG       = (255, 255, 255, 255)          # clean white
    TEXT_DK  = (23, 23, 23, 255)
    TEXT_MD  = (55, 65, 81, 255)
    TEXT_LT  = (120, 128, 140, 255)
    GRID_MAJ = (232, 236, 240, 255)
    GRID_MIN = (242, 244, 247, 255)
    GREEN    = (22, 163, 74, 255)
    RED      = (239, 68, 68, 255)
    WICK     = (140, 140, 140, 255)
    SR_BLUE  = (120, 162, 255, 58)
    SR_RED   = (255, 154, 154, 58)
    SR_BLUE_ST = (120, 162, 255, 140)
    SR_RED_ST  = (255, 154, 154, 140)
    ACCENT   = (255, 210, 0, 255)

    base = Image.new("RGBA", (W, H), BG)
    draw = ImageDraw.Draw(base)

    # ---------- fonts ----------
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

    # ---------- layout (whitespace + smaller chart area) ----------
    outer_top = sp(60)
    outer_lr  = sp(64)
    outer_bot = sp(60)

    header_h = sp(200)
    footer_h = sp(140)

    chart = [outer_lr, outer_top + header_h, W - outer_lr, H - outer_bot - footer_h]
    cx1, cy1, cx2, cy2 = chart

    # ---------- header (equal spacing) ----------
    title_x, title_y = outer_lr, outer_top
    GAP = st(18)  # equal vertical gap between each header line

    def draw_line(x, y, text, font, fill):
        # draw, then return new y with equal gap using measured text height
        draw.text((x, y), text, fill=fill, font=font)
        bbox = draw.textbbox((x, y), text, font=font)
        h = bbox[3] - bbox[1]
        return y + h + GAP

    # 1) Ticker
    y_cursor = draw_line(title_x, title_y, ticker, f_ticker, TEXT_DK)

    # 2) Price
    y_cursor = draw_line(title_x, y_cursor, f"{last:,.2f} USD", f_price, TEXT_MD)

    # 3) 30d change
    delta_col = GREEN if chg30 >= 0 else RED
    y_cursor = draw_line(title_x, y_cursor, f"{chg30:+.2f}% past 30d", f_delta, delta_col)

    # 4) Sub label (timeframe)
    sub_label = "Daily chart • last ~1 year" if bos_tf == "D" else "Weekly chart • last 52 weeks"
    y_cursor = draw_line(title_x, y_cursor, sub_label, f_sub, TEXT_LT)

    # Ticker logo (independent scale), aligned to top of header block
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

    # ---------- data ----------
    df2 = df[["Open", "High", "Low", "Close"]].dropna()
    if df2.shape[0] < 2:
        out = base.convert("RGB")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out.save(out_path, quality=95)
        return

    ymin = float(np.nanmin(df2["Low"]))
    ymax = float(np.nanmax(df2["High"]))
    if not np.isfinite(ymin) or not np.isfinite(ymax) or abs(ymax - ymin) < 1e-6:
        ymin, ymax = (ymin - 0.5, ymax + 0.5) if np.isfinite(ymin) else (0, 1)

    # tiny padding so labels/zones breathe
    yr = ymax - ymin
    ymin -= 0.02 * yr
    ymax += 0.02 * yr

    def sx(i): return cx1 + (i / max(1, len(df2) - 1)) * (cx2 - cx1)
    def sy(v): return cy2 - ((float(v) - ymin) / (ymax - ymin)) * (cy2 - cy1)

    # ---------- grid (light) ----------
    grid = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    g = ImageDraw.Draw(grid)
    for i in range(1, 7):
        y = cy1 + i * (cy2 - cy1) / 7.0
        g.line([(cx1, y), (cx2, y)], fill=GRID_MIN, width=sp(1))
    for frac in (0.33, 0.66):
        y = cy1 + frac * (cy2 - cy1)
        g.line([(cx1, y), (cx2, y)], fill=GRID_MAJ, width=sp(1))
    for i in range(1, 9):
        x = cx1 + i * (cx2 - cx1) / 9.0
        g.line([(x, cy1), (x, cy2)], fill=GRID_MIN, width=sp(1))
    base = Image.alpha_composite(base, grid)
    draw = ImageDraw.Draw(base)

    # ---------- S/R shaded zones (plain labels, no rounded backgrounds) ----------
    # support
    sup_y1, sup_y2 = sy(sup_high), sy(sup_low)
    sup_rect = [cx1, min(sup_y1, sup_y2), cx2, max(sup_y1, sup_y2)]
    sup_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(sup_layer).rectangle(sup_rect, fill=SR_BLUE, outline=SR_BLUE_ST, width=sp(2))
    base = Image.alpha_composite(base, sup_layer)
    draw = ImageDraw.Draw(base)
    draw.text((cx2 - sp(240), min(sup_y1, sup_y2) + sp(6)),
              f"{sup_label} ~{sup_low:.2f}", fill=(65, 90, 140, 255), font=f_sm)

    # resistance
    res_y1, res_y2 = sy(res_high), sy(res_low)
    res_rect = [cx1, min(res_y1, res_y2), cx2, max(res_y1, res_y2)]
    res_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(res_layer).rectangle(res_rect, fill=SR_RED, outline=SR_RED_ST, width=sp(2))
    base = Image.alpha_composite(base, res_layer)
    draw = ImageDraw.Draw(base)
    draw.text((cx2 - sp(240), min(res_y1, res_y2) + sp(6)),
              f"{res_label} ~{res_high:.2f}", fill=(150, 60, 60, 255), font=f_sm)

    # ---------- candlesticks (thin for density) ----------
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
        if abs(y2 - y1) < 1:
            y2 = y1 + 1
        draw.rectangle([xx - half, y1, xx + half, y2], fill=col, outline=None)

    # ---------- BOS line (optional) ----------
    if bos_dir is not None and np.isfinite(bos_level):
        by = sy(bos_level)
        draw.line([(cx1, by), (cx2, by)], fill=ACCENT, width=sp(3))

    # ---------- right-side price ticks ----------
    ticks = np.linspace(ymin, ymax, 5)
    for tval in ticks:
        y = sy(tval)
        label = f"{tval:,.2f}"
        bbox = draw.textbbox((0, 0), label, font=f_axis)
        tw, th = (bbox[2]-bbox[0], bbox[3]-bbox[1])
        draw.text((cx2 + sp(8), y - th/2), label, fill=TEXT_LT, font=f_axis)

    # ---------- footer ----------
    meta_x = outer_lr
    meta_y = H - outer_bot - st(64)
    draw.text((meta_x, meta_y),          f"Resistance ~{res_high:.2f}", fill=TEXT_LT, font=f_sm)
    draw.text((meta_x, meta_y + st(24)), f"Support ~{sup_low:.2f}",     fill=TEXT_LT, font=f_sm)
    draw.text((meta_x, meta_y + st(48)), "Not financial advice",        fill=(160,160,160,255), font=f_sm)

    # ---------- brand logo ----------
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

# ------------------ Caption Builder ------------------
def plain_english_line(ticker, headline, payload, seed=None):
    """
    Human caption with varied phrasing, emojis, and light technical tilt.
    """
    (df,last,chg30,sup_low,sup_high,res_low,res_high,
     sup_label,res_label,bos_dir,bos_level,bos_idx,bos_tf) = payload

    if seed is None:
        seed = f"{ticker}-{DATESTR}"
    rnd = random.Random(str(seed))

    if headline and len(headline) > 160:
        headline = headline[:157] + "…"
    lead = headline or rnd.choice([
        "No major headlines today.", "Quiet on the news front.", "News flow is light."
    ])

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
    line = f"📈 {ticker} — {lead} — {cue_txt}"
    return line[:280]

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
                print(f"[warn] no data for {t}, skipping")
                continue

            out_path = os.path.join(OUTPUT_DIR, f"twd_{t}_{DATESTR}.png")
            print(f"[debug] saving {out_path}")
            render_single_post(out_path, t, payload)
            print("done:", out_path)
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
        footer = f"\n\n{random.choice(CTA_POOL)}\n\nIdeas only — not financial advice"
        with open(caption_path, "w", encoding="utf-8") as f:
            f.write(f"Ones to Watch – {now_str}\n\n")
            f.write("\n\n".join(captions))
            f.write(footer)
        print("[info] wrote caption:", caption_path)

if __name__ == "__main__":
    main()
