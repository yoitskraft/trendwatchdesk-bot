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

# ------------------ Company names ------------------
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

# ------------------ Ticker chooser ------------------
def choose_tickers_somehow():
    """
    Deterministic daily pick of 6 tickers from the pool.
    (Seeded by DATESTR so Mon/Wed/Fri differ.)
    """
    pool = list(COMPANY_QUERY.keys())
    rnd = random.Random(DATESTR)
    k = min(6, len(pool))
    return rnd.sample(pool, k)

# ------------------ Data fetcher (daily → weekly) ------------------
def _coerce_close_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    for name in ("Close", "Adj Close", "close"):
        if name in df.columns:
            close = df[name]
            break
    else:
        close = None
        # MultiIndex fallback
        if isinstance(df.columns, pd.MultiIndex):
            for name in ("Close", "Adj Close"):
                if name in df.columns.get_level_values(-1):
                    sub = df.xs(name, axis=1, level=-1, drop_level=False)
                    if isinstance(sub, pd.DataFrame) and sub.shape[1] >= 1:
                        close = sub.iloc[:, 0]
                        break
    if close is None:
        return pd.Series(dtype=float)
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return pd.to_numeric(close, errors="coerce").dropna()

def fetch_one(ticker):
    """
    Load 1y of daily (auto_adjust), compute 30d % change,
    resample to weekly (W-FRI) OHLC for plotting last ~52 weeks.
    Returns payload:
      (df_weekly, last, chg30, sup_low, sup_high, res_low, res_high,
       'Support','Resistance', bos_dir, bos_level, bos_idx, 'W')
    """
    df_d = yf.download(ticker, period="1y", interval="1d", auto_adjust=True)
    if df_d is None or df_d.empty:
        return None

    close_d = _coerce_close_series(df_d)
    if close_d.empty:
        return None

    # 30-day change on daily series
    last_date = close_d.index[-1]
    base_date = last_date - pd.Timedelta(days=30)
    base_series = close_d.loc[:base_date]
    if base_series.empty:
        chg30 = 0.0
        base_val = float(close_d.iloc[0])
    else:
        base_val = float(base_series.iloc[-1])
        chg30 = float(100.0 * (float(close_d.iloc[-1]) - base_val) / base_val) if base_val != 0 else 0.0
    last = float(close_d.iloc[-1])

    # Ensure OHLC exists; if not, build from close (best effort)
    if not set(["Open", "High", "Low", "Close"]).issubset(df_d.columns):
        df_d = df_d.assign(Open=close_d, High=close_d, Low=close_d, Close=close_d)

    # Weekly resample
    df_w = df_d[["Open", "High", "Low", "Close"]].resample("W-FRI").agg(
        {"Open":"first", "High":"max", "Low":"min", "Close":"last"}
    ).dropna()
    df_w = df_w.tail(60).dropna()  # ~last 52-60 weeks for safety

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
    Weekly chart (last ~52w) with polished styling & big Grift/Roboto text.
    Expects payload:
      (df_w, last, chg30, sup_low, sup_high, res_low, res_high,
       sup_label, res_label, bos_dir, bos_level, bos_idx, bos_tf)
    """
    (df_w, last, chg30, sup_low, sup_high, res_low, res_high,
     sup_label, res_label, bos_dir, bos_level, bos_idx, bos_tf) = payload

    # ---------- theme ----------
    W, H = 1080, 1080
    BG       = (242, 244, 247, 255)   # page bg
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
    ACCENT   = (255, 210, 0, 255)     # BOS line
    MAJOR_GL = (231, 235, 240, 255)
    MINOR_GL = (238, 241, 245, 255)

    # ---------- canvas + drop shadow ----------
    base = Image.new("RGBA", (W, H), BG)
    outer_margin = 44
    card = [outer_margin, outer_margin, W - outer_margin, H - outer_margin]

    shadow_rect = [card[0] + 6, card[1] + 10, card[2] + 6, card[3] + 10]
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(shadow_rect, radius=32, fill=(0, 0, 0, 70))
    shadow = shadow.filter(ImageFilter.GaussianBlur(12))
    base = Image.alpha_composite(base, shadow)

    card_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(card_layer).rounded_rectangle(card, radius=28, fill=CARD, outline=BORDER, width=2)
    base = Image.alpha_composite(base, card_layer)
    draw = ImageDraw.Draw(base)

    # ---------- fonts (Grift → Roboto → default) ----------
    def _try_font(path, size):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            return None

    def _font(size, bold=False):
        grift_bold = _try_font("assets/fonts/Grift-Bold.ttf", size)
        grift_reg  = _try_font("assets/fonts/Grift-Regular.ttf", size)
        roboto_bold = (_try_font("assets/fonts/Roboto-Bold.ttf", size)
                       or _try_font("Roboto-Bold.ttf", size))
        roboto_reg  = (_try_font("assets/fonts/Roboto-Regular.ttf", size)
                       or _try_font("Roboto-Regular.ttf", size))
        if bold:
            return grift_bold or roboto_bold or ImageFont.load_default()
        else:
            return grift_reg  or roboto_reg  or ImageFont.load_default()

    f_ticker = _font(100, bold=True)  # BIG ticker
    f_price  = _font(56,  bold=True)  # price
    f_delta  = _font(50,  bold=True)  # 30d change
    f_sub    = _font(34)              # subtitle
    f_sm     = _font(28)              # footer/meta
    f_badge  = _font(26,  bold=True)  # S/R badges

    # ---------- layout ----------
    header_h = 180
    footer_h = 116
    pad_l, pad_r = 64, 58
    chart = [card[0] + pad_l, card[1] + header_h, card[2] - pad_r, card[3] - footer_h]
    cx1, cy1, cx2, cy2 = chart

    # ---------- header ----------
    title_x, title_y = card[0] + 28, card[1] + 24
    draw.text((title_x, title_y), ticker, fill=TEXT_DK, font=f_ticker)

    price_y = title_y + 96
    draw.text((title_x, price_y), f"{last:,.2f} USD", fill=TEXT_MD, font=f_price)

    delta_col = GREEN if chg30 >= 0 else RED
    draw.text((title_x, price_y + 50), f"{chg30:+.2f}% past 30d", fill=delta_col, font=f_delta)
    draw.text((title_x, price_y + 98), "Weekly chart • last 52 weeks", fill=TEXT_LT, font=f_sub)

    # ticker logo (optional)
    tlogo_path = os.path.join("assets", "logos", f"{ticker}.png")
    if os.path.exists(tlogo_path):
        try:
            tlogo = Image.open(tlogo_path).convert("RGBA")
            hmax = 92
            scl = min(1.0, hmax / max(1, tlogo.height))
            tlogo = tlogo.resize((int(tlogo.width * scl), int(tlogo.height * scl)))
            base.alpha_composite(tlogo, (card[2] - 28 - tlogo.width, title_y + 2))
        except Exception:
            pass

    # ---------- grid (minor + major) ----------
    grid = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    g = ImageDraw.Draw(grid)
    for i in range(1, 5):
        y = cy1 + i * (cy2 - cy1) / 5.0
        g.line([(cx1, y), (cx2, y)], fill=MINOR_GL, width=1)
    g.line([(cx1, (cy1 + cy2) / 2), (cx2, (cy1 + cy2) / 2)], fill=MAJOR_GL, width=1)
    for i in range(1, 6):
        x = cx1 + i * (cx2 - cx1) / 6.0
        g.line([(x, cy1), (x, cy2)], fill=MINOR_GL, width=1)
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

    def sx(i):
        return cx1 + (i / max(1, len(df2) - 1)) * (cx2 - cx1)

    def sy(v):
        return cy2 - ((float(v) - ymin) / (ymax - ymin)) * (cy2 - cy1)

    # ---------- shaded S/R zones + badges ----------
    # support
    sup_y1, sup_y2 = sy(sup_high), sy(sup_low)
    sup_rect = [cx1, min(sup_y1, sup_y2), cx2, max(sup_y1, sup_y2)]
    sup_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(sup_layer).rectangle(sup_rect, fill=SR_BLUE, outline=SR_BLUE_ST, width=2)
    base = Image.alpha_composite(base, sup_layer)
    draw = ImageDraw.Draw(base)
    # badge
    badge_txt = f"{sup_label} ~{sup_low:.2f}"
    tw, th = draw.textbbox((0, 0), badge_txt, font=f_badge)[2:]
    bx1 = cx2 - tw - 14
    by1 = min(sup_y1, sup_y2) + 8
    bg = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(bg).rounded_rectangle([bx1 - 10, by1 - 6, bx1 + tw + 10, by1 + th + 6],
                                         radius=12, fill=(120, 162, 255, 40))
    base = Image.alpha_composite(base, bg)
    draw = ImageDraw.Draw(base)
    draw.text((bx1, by1), badge_txt, fill=(65, 90, 140, 255), font=f_badge)

    # resistance
    res_y1, res_y2 = sy(res_high), sy(res_low)
    res_rect = [cx1, min(res_y1, res_y2), cx2, max(res_y1, res_y2)]
    res_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(res_layer).rectangle(res_rect, fill=SR_RED, outline=SR_RED_ST, width=2)
    base = Image.alpha_composite(base, res_layer)
    draw = ImageDraw.Draw(base)
    # badge
    badge_txt2 = f"{res_label} ~{res_high:.2f}"
    tw2, th2 = draw.textbbox((0, 0), badge_txt2, font=f_badge)[2:]
    bx2 = cx2 - tw2 - 14
    by2 = min(res_y1, res_y2) + 8
    bg2 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(bg2).rounded_rectangle([bx2 - 10, by2 - 6, bx2 + tw2 + 10, by2 + th2 + 6],
                                          radius=12, fill=(255, 154, 154, 40))
    base = Image.alpha_composite(base, bg2)
    draw = ImageDraw.Draw(base)
    draw.text((bx2, by2), badge_txt2, fill=(150, 60, 60, 255), font=f_badge)

    # ---------- candlesticks ----------
    n = len(df2)
    body_px = max(4, int((cx2 - cx1) / max(60, n * 1.35)))
    half = body_px // 2
    for i, row in enumerate(df2.itertuples(index=False)):
        O, Hh, Ll, C = row
        xx = sx(i)
        draw.line([(xx, sy(Hh)), (xx, sy(Ll))], fill=WICK, width=1)
        col = GREEN if C >= O else RED
        y1 = sy(max(O, C))
        y2 = sy(min(O, C))
        if abs(y2 - y1) < 1:
            y2 = y1 + 1
        draw.rectangle([xx - half, y1, xx + half, y2], fill=col, outline=(0, 0, 0, 0))

    # ---------- BOS line ----------
    if bos_dir is not None and np.isfinite(bos_level):
        by = sy(bos_level)
        draw.line([(cx1, by), (cx2, by)], fill=ACCENT, width=3)

    # ---------- footer ----------
    meta_x = card[0] + 24
    meta_y = card[3] - 76
    draw.text((meta_x, meta_y),      f"Resistance ~{res_high:.2f}", fill=TEXT_LT, font=f_sm)
    draw.text((meta_x, meta_y + 26), f"Support ~{sup_low:.2f}",     fill=TEXT_LT, font=f_sm)
    draw.text((meta_x, meta_y + 52), "Not financial advice",        fill=(160, 160, 160, 255), font=f_sm)

    # brand logo (bottom-right, optional)
    logo_path = BRAND_LOGO_PATH or "assets/brand_logo.png"
    if logo_path and os.path.exists(logo_path):
        try:
            blogo = Image.open(logo_path).convert("RGBA")
            maxh = 120
            scl = min(1.0, maxh / max(1, blogo.height))
            blogo = blogo.resize((int(blogo.width * scl), int(blogo.height * scl)))
            base.alpha_composite(blogo, (card[2] - 24 - blogo.width, card[3] - 24 - blogo.height))
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

    # headline lead-in
    if headline and len(headline) > 160:
        headline = headline[:157] + "…"
    lead = headline or rnd.choice([
        "No major headlines today.", "Quiet on the news front.", "News flow is light."
    ])

    # cues (non-jargony)
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
