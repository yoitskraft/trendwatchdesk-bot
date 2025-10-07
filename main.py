#!/usr/bin/env python3
import os, sys, io, math, json, random, datetime, traceback
import numpy as np
import pandas as pd
import yfinance as yf
from PIL import Image, ImageDraw, ImageFont, ImageFilter
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

# ------------------ Ticker chooser (keep simple/deterministic by day) ------------------
def choose_tickers_somehow():
    """
    Deterministic daily pick of 6 tickers from the pool.
    Seeded by DATESTR so Mon/Wed/Fri differ.
    """
    pool = list(COMPANY_QUERY.keys())
    rnd = random.Random(DATESTR)
    k = min(6, len(pool))
    return rnd.sample(pool, k)

# ------------------ Robust OHLC helpers (fixes MultiIndex + missing cols) ------------------
def _find_col(df, name):
    """
    Return a 1D numeric Series for 'Open'/'High'/'Low'/'Close'/'Adj Close'
    from either single-level or MultiIndex yfinance frames. Returns None if not found.
    """
    if df is None or df.empty:
        return None

    # 1) Direct single-level match
    if name in df.columns:
        ser = df[name]
        if isinstance(ser, pd.DataFrame):
            ser = ser.iloc[:, 0]
        return pd.to_numeric(ser, errors="coerce")

    # 2) Normalized-name match (lowercase, no spaces)
    norm = {str(c).lower().replace(" ", ""): c for c in df.columns}
    key = name.lower().replace(" ", "")
    if key in norm:
        ser = df[norm[key]]
        if isinstance(ser, pd.DataFrame):
            ser = ser.iloc[:, 0]
        return pd.to_numeric(ser, errors="coerce")

    # 3) MultiIndex match (yfinance often uses level0=field, level1=ticker)
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        lvlN = df.columns.get_level_values(-1)
        # Try level 0 (field name)
        if name in set(lvl0):
            sub = df[name]
            if isinstance(sub, pd.DataFrame):
                ser = sub.iloc[:, 0]
            else:
                ser = sub
            return pd.to_numeric(ser, errors="coerce")
        # Try last level (in case someone swapped levels)
        if name in set(lvlN):
            sub = df.xs(name, axis=1, level=-1)
            if isinstance(sub, pd.DataFrame):
                ser = sub.iloc[:, 0]
            else:
                ser = sub
            return pd.to_numeric(ser, errors="coerce")

    return None

def _get_ohlc_df(df):
    """
    Build a clean OHLC DataFrame with simple columns ['Open','High','Low','Close'].
    If some fields are missing, fill from Close as best-effort.
    """
    if df is None or df.empty:
        return None

    o = _find_col(df, "Open")
    h = _find_col(df, "High")
    l = _find_col(df, "Low")

    # DO NOT use Python `or` on Series — it’s ambiguous.
    c = _find_col(df, "Close")
    if c is None or c.dropna().empty:
        c = _find_col(df, "Adj Close")
    if c is None or c.dropna().empty:
        return None

    # Align to Close index
    idx = c.index

    def _align(x):
        return pd.to_numeric(x, errors="coerce").reindex(idx) if x is not None else None

    o = _align(o)
    h = _align(h)
    l = _align(l)
    c = _align(c).astype(float)

    # Fill missing OHLC from Close (best effort)
    if o is None: o = c.copy()
    if h is None: h = c.copy()
    if l is None: l = c.copy()

    out = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c}).dropna()
    if out.empty:
        return None
    return out
# ------------------ Data fetcher (daily → weekly) ------------------
def fetch_one(ticker):
    """
    Load 1y of daily (auto_adjust), compute 30d % change, resample to weekly (W-FRI).
    Returns payload:
      (df_weekly, last, chg30, sup_low, sup_high, res_low, res_high,
       'Support','Resistance', bos_dir, bos_level, bos_idx, 'W')
    """
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

    # Weekly resample (flat OHLC guaranteed)
    df_w = ohlc_d.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    df_w = df_w.tail(60).dropna()  # last ~52-60 weeks
    if df_w.empty:
        return None

    # Support/Resistance zones from weekly extremes (tight bands)
    w_low  = float(df_w["Low"].min())
    w_high = float(df_w["High"].max())
    sup_low, sup_high = w_low, w_low * 1.03     # +3% band
    res_high, res_low = w_high, w_high * 0.97   # -3% band

    bos_dir = "up" if chg30 > 5 else ("down" if chg30 < -5 else None)
    bos_level = last
    bos_idx = int(len(df_w) - 1)

    return (df_w, last, float(chg30), sup_low, sup_high, res_low, res_high,
            "Support", "Resistance", bos_dir, bos_level, bos_idx, "W")

# ------------------ Renderer (polished weekly, Grift font) ------------------
def render_single_post(out_path, ticker, payload):
    """
    Weekly chart (last ~52w) — fully scalable.
    Use env TWD_UI_SCALE (float) to scale the whole layout (fonts, paddings, chart, logos).
    """
    (df_w, last, chg30, sup_low, sup_high, res_low, res_high,
     sup_label, res_label, bos_dir, bos_level, bos_idx, bos_tf) = payload

    # ---------- scale (global) ----------
    try:
        S = float(os.getenv("TWD_UI_SCALE", "0.90"))  # 0.90 = 10% smaller than before
    except Exception:
        S = 0.90

    def sp(x):  # scale to int px
        return int(round(x * S))

    # ---------- theme ----------
    W, H = 1080, 1080
    BG       = (242, 244, 247, 255)
    CARD     = (255, 255, 255, 255)
    BORDER   = (226, 232, 240, 255)
    TEXT_DK  = (23, 23, 23, 255)
    TEXT_MD  = (55, 65, 81, 255)
    TEXT_LT  = (120, 128, 140, 255)
    GREEN    = (22, 163, 74, 255)
    RED      = (239, 68, 68, 255)
    WICK     = (140, 140, 140, 255)
    SR_BLUE  = (120, 162, 255, 72)
    SR_RED   = (255, 154, 154, 72)
    SR_BLUE_ST = (120, 162, 255, 160)
    SR_RED_ST  = (255, 154, 154, 160)
    ACCENT   = (255, 210, 0, 255)
    MAJOR_GL = (231, 235, 240, 255)
    MINOR_GL = (238, 241, 245, 255)

    # ---------- canvas + card + shadow ----------
    base = Image.new("RGBA", (W, H), BG)
    outer_margin = sp(44)
    card = [outer_margin, outer_margin, W - outer_margin, H - outer_margin]

    shadow_rect = [card[0] + sp(6), card[1] + sp(10), card[2] + sp(6), card[3] + sp(10)]
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(shadow_rect, radius=sp(32), fill=(0, 0, 0, 70))
    shadow = shadow.filter(ImageFilter.GaussianBlur(sp(12)))
    base = Image.alpha_composite(base, shadow)

    card_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(card_layer).rounded_rectangle(card, radius=sp(28), fill=CARD, outline=BORDER, width=sp(2))
    base = Image.alpha_composite(base, card_layer)
    draw = ImageDraw.Draw(base)

    # ---------- fonts (Grift → Roboto → default), scaled ----------
    def _try_font(path, size):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            return None

    def _font(size, bold=False):
        sz = sp(size)
        grift_bold = _try_font("assets/fonts/Grift-Bold.ttf", sz)
        grift_reg  = _try_font("assets/fonts/Grift-Regular.ttf", sz)
        roboto_bold = (_try_font("assets/fonts/Roboto-Bold.ttf", sz)
                       or _try_font("Roboto-Bold.ttf", sz))
        roboto_reg  = (_try_font("assets/fonts/Roboto-Regular.ttf", sz)
                       or _try_font("Roboto-Regular.ttf", sz))
        if bold:
            return grift_bold or roboto_bold or ImageFont.load_default()
        else:
            return grift_reg  or roboto_reg  or ImageFont.load_default()

    # base sizes now pass through _font which scales via sp()
    f_ticker = _font(100, bold=True)
    f_price  = _font(56,  bold=True)
    f_delta  = _font(50,  bold=True)
    f_sub    = _font(34)
    f_sm     = _font(28)
    f_badge  = _font(26,  bold=True)

    # ---------- layout boxes (scaled) ----------
    header_h = sp(180)
    footer_h = sp(116)
    pad_l, pad_r = sp(64), sp(58)
    chart = [card[0] + pad_l, card[1] + header_h, card[2] - pad_r, card[3] - footer_h]
    cx1, cy1, cx2, cy2 = chart

    # ---------- header (scaled offsets) ----------
    title_x, title_y = card[0] + sp(28), card[1] + sp(24)
    draw.text((title_x, title_y), ticker, fill=TEXT_DK, font=f_ticker)

    price_y = title_y + sp(96)
    draw.text((title_x, price_y), f"{last:,.2f} USD", fill=TEXT_MD, font=f_price)

    delta_col = GREEN if chg30 >= 0 else RED
    draw.text((title_x, price_y + sp(50)), f"{chg30:+.2f}% past 30d", fill=delta_col, font=f_delta)
    draw.text((title_x, price_y + sp(98)), "Weekly chart • last 52 weeks", fill=TEXT_LT, font=f_sub)

    # ticker logo (scaled)
    tlogo_path = os.path.join("assets", "logos", f"{ticker}.png")
    if os.path.exists(tlogo_path):
        try:
            tlogo = Image.open(tlogo_path).convert("RGBA")
            hmax = sp(92)
            scl = min(1.0, hmax / max(1, tlogo.height))
            tlogo = tlogo.resize((int(tlogo.width * scl), int(tlogo.height * scl)))
            base.alpha_composite(tlogo, (card[2] - sp(28) - tlogo.width, title_y + sp(2)))
        except Exception:
            pass

    # ---------- grid (scaled stroke positions) ----------
    grid = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    g = ImageDraw.Draw(grid)
    for i in range(1, 5):
        y = cy1 + i * (cy2 - cy1) / 5.0
        g.line([(cx1, y), (cx2, y)], fill=MINOR_GL, width=sp(1))
    g.line([(cx1, (cy1 + cy2) / 2), (cx2, (cy1 + cy2) / 2)], fill=MAJOR_GL, width=sp(1))
    for i in range(1, 6):
        x = cx1 + i * (cx2 - cx1) / 6.0
        g.line([(x, cy1), (x, cy2)], fill=MINOR_GL, width=sp(1))
    base = Image.alpha_composite(base, grid)
    draw = ImageDraw.Draw(base)

    # ---------- OHLC (weekly) ----------
    df2 = df_w[["Open", "High", "Low", "Close"]].dropna()
    if df2.shape[0] < 2:
        out = base.convert("RGB")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out.save(out_path, quality=95)
        return

    ymin = float(np.nanmin(df2["Low"]))
    ymax = float(np.nanmax(df2["High"]))
    if not np.isfinite(ymin) or not np.isfinite(ymax) or abs(ymax - ymin) < 1e-9:
        ymin, ymax = (ymin - 0.5, ymax + 0.5) if np.isfinite(ymin) else (0, 1)

    def sx(i): return cx1 + (i / max(1, len(df2) - 1)) * (cx2 - cx1)
    def sy(v): return cy2 - ((float(v) - ymin) / (ymax - ymin)) * (cy2 - cy1)

    # ---------- S/R zones + badges (scaled) ----------
    # support
    sup_y1, sup_y2 = sy(sup_high), sy(sup_low)
    sup_rect = [cx1, min(sup_y1, sup_y2), cx2, max(sup_y1, sup_y2)]
    sup_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(sup_layer).rectangle(sup_rect, fill=SR_BLUE, outline=SR_BLUE_ST, width=sp(2))
    base = Image.alpha_composite(base, sup_layer)
    draw = ImageDraw.Draw(base)
    # badge
    badge_txt = f"{sup_label} ~{sup_low:.2f}"
    tw, th = draw.textbbox((0, 0), badge_txt, font=f_badge)[2:]
    bx1 = cx2 - tw - sp(14)
    by1 = min(sup_y1, sup_y2) + sp(8)
    bg = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(bg).rounded_rectangle([bx1 - sp(10), by1 - sp(6), bx1 + tw + sp(10), by1 + th + sp(6)],
                                         radius=sp(12), fill=(120, 162, 255, 40))
    base = Image.alpha_composite(base, bg)
    draw = ImageDraw.Draw(base)
    draw.text((bx1, by1), badge_txt, fill=(65, 90, 140, 255), font=f_badge)

    # resistance
    res_y1, res_y2 = sy(res_high), sy(res_low)
    res_rect = [cx1, min(res_y1, res_y2), cx2, max(res_y1, res_y2)]
    res_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(res_layer).rectangle(res_rect, fill=SR_RED, outline=SR_RED_ST, width=sp(2))
    base = Image.alpha_composite(base, res_layer)
    draw = ImageDraw.Draw(base)
    # badge
    badge_txt2 = f"{res_label} ~{res_high:.2f}"
    tw2, th2 = draw.textbbox((0, 0), badge_txt2, font=f_badge)[2:]
    bx2 = cx2 - tw2 - sp(14)
    by2 = min(res_y1, res_y2) + sp(8)
    bg2 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(bg2).rounded_rectangle([bx2 - sp(10), by2 - sp(6), bx2 + tw2 + sp(10), by2 + th2 + sp(6)],
                                          radius=sp(12), fill=(255, 154, 154, 40))
    base = Image.alpha_composite(base, bg2)
    draw = ImageDraw.Draw(base)
    draw.text((bx2, by2), badge_txt2, fill=(150, 60, 60, 255), font=f_badge)

    # ---------- candlesticks (scaled widths) ----------
    n = len(df2)
    # narrower bodies when more weeks; width scaled with S
    base_body_px = max(4, int((cx2 - cx1) / max(60, n * 1.35)))
    body_px = max(2, int(round(base_body_px * S)))
    half = body_px // 2

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

    # ---------- BOS line (scaled) ----------
    if bos_dir is not None and np.isfinite(bos_level)):
        by = sy(bos_level)
        draw.line([(cx1, by), (cx2, by)], fill=ACCENT, width=sp(3))

    # ---------- footer (scaled) ----------
    meta_x = card[0] + sp(24)
    meta_y = card[3] - sp(76)
    draw.text((meta_x, meta_y),          f"Resistance ~{res_high:.2f}", fill=TEXT_LT, font=f_sm)
    draw.text((meta_x, meta_y + sp(26)), f"Support ~{sup_low:.2f}",     fill=TEXT_LT, font=f_sm)
    draw.text((meta_x, meta_y + sp(52)), "Not financial advice",        fill=(160, 160, 160, 255), font=f_sm)

    # ---------- brand logo (scaled) ----------
    logo_path = BRAND_LOGO_PATH or "assets/brand_logo.png"
    if logo_path and os.path.exists(logo_path):
        try:
            blogo = Image.open(logo_path).convert("RGBA")
            maxh = sp(120)
            scl = min(1.0, maxh / max(1, blogo.height))
            blogo = blogo.resize((int(blogo.width * scl), int(blogo.height * scl)))
            base.alpha_composite(blogo, (card[2] - sp(24) - blogo.width, card[3] - sp(24) - blogo.height))
        except Exception:
            pass

    # ---------- save ----------
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
            "watching for a decisive move soon", "tightening ranges on the weekly"
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
