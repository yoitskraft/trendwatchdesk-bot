#!/usr/bin/env python3
import os, sys, io, math, json, random, datetime, traceback
import numpy as np
import pandas as pd
import yfinance as yf
from PIL import Image, ImageDraw, ImageFont

# ------------------ Config ------------------
OUTPUT_DIR = os.path.abspath("output")
TODAY = datetime.date.today()
DATESTR = TODAY.strftime("%Y%m%d")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "").strip()

# brand assets
BRAND_LOGO_PATH = os.getenv("BRAND_LOGO_PATH", "assets/brand_logo.png")  # optional

# ------------------ Utilities you already had ------------------
# atr(df, n=14), swing_points(df, w=2), fetch_one(ticker) -> payload
# render_single_post(out_path, ticker, payload) -> saves 1080x1080 chart PNG
# (keep your existing implementations)

# ------------------ News helpers ------------------
COMPANY_QUERY = {
    "META":"Meta Platforms","AMD":"Advanced Micro Devices","GOOG":"Google Alphabet","GOOGL":"Alphabet",
    "AAPL":"Apple","MSFT":"Microsoft","TSM":"Taiwan Semiconductor","TSLA":"Tesla",
    "JNJ":"Johnson & Johnson","MA":"Mastercard","V":"Visa","NVDA":"NVIDIA",
    "AMZN":"Amazon","SNOW":"Snowflake","SQ":"Block Inc","PYPL":"PayPal","UNH":"UnitedHealth"
}

import requests
SESS = requests.Session()
SESS.headers.update({"User-Agent":"TWD/1.0"})

def news_headline_for(ticker):
    """Try NewsAPI (if key present), else yfinance .news. Return short headline or None."""
    name = COMPANY_QUERY.get(ticker, ticker)
    # NewsAPI
    if NEWSAPI_KEY:
        try:
            r = SESS.get(
                "https://newsapi.org/v2/everything",
                params={"q": f'"{name}" OR {ticker}', "language":"en", "sortBy":"publishedAt", "pageSize":1},
                headers={"X-Api-Key": NEWSAPI_KEY},
                timeout=8
            )
            # ---- FIX: avoid walrus operator; two-step parse
            if r.ok:
                d = r.json().get("articles")
                if d:
                    title = d[0].get("title") or ""
                    src = d[0].get("source", {}).get("name", "")
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

# ------------------ Natural caption builder (varied + emojis) ------------------
from datetime import date
def plain_english_line(ticker, headline, payload, seed=None):
    """
    Human caption with varied phrasing, emojis, and light technical tilt.
    Avoids repetition across a run by seeding randomness with the date.
    """
    (df,last,chg30,sup_low,sup_high,res_low,res_high,
     sup_label,res_label,bos_dir,bos_level,bos_idx,bos_tf) = payload

    if seed is None:
        seed = f"{ticker}-{DATESTR}"
    rnd = random.Random(str(seed))

    # headline lead-in
    lead_pool = [
        "With {h}", "{h}", "Fresh headlines: {h}",
        "Latest: {h}", "In the news: {h}", "{h}"
    ]
    if headline and len(headline) > 160:
        headline = headline[:157] + "…"
    lead = (rnd.choice(lead_pool).format(h=headline)
            if headline else rnd.choice(["No major headlines today.",
                                         "Quiet on the news front.",
                                         "News flow is light."]))

    # cues (non-jargony)
    cues = []
    if chg30 >= 8: cues.append("momentum looks strong 🔥")
    elif chg30 <= -8: cues.append("recent pullback showing ⚠️")

    def near(l, h, p, k=1.0):
        if l is None or h is None: return False
        mid = 0.5*(l+h); rng = (h-l)*k + 1e-8
        return abs(p - mid) <= 0.6*rng

    near_sup = near(sup_low, sup_high, last)
    near_res = near(res_low, res_high, last)
    if near_sup: cues.append("buyers defended support 🛡️")
    if near_res: cues.append("testing overhead supply 🧱")
    if bos_dir == "up": cues.append("breakout pressure building 🚀")
    if bos_dir == "down": cues.append("post-breakdown chop ⚠️")
    if not cues:
        cues = rnd.sample([
            "price action is steady","range bound but coiling",
            "watching for a decisive move soon","tightening ranges on the daily"
        ], k=1)

    # ending tilt
    endings_bull = [
        "could have more room if momentum sticks ✅",
        "setups lean constructive here 📈",
        "watch for follow-through on strength 🔎",
        "dips may get bought if tone stays positive 🧠"
    ]
    endings_bear = [
        "risk of rejection—watch reactions at key levels 👀",
        "tone is cautious; patience helps here 🧊",
        "relief bounces possible, trend still mixed ⚖️",
        "respect your stops if weakness persists 🛑"
    ]
    endings_neutral = [
        "neutral bias—let price confirm next leg 🎯",
        "waiting on a clean trigger ⚙️",
        "keep it on the radar; confirmation matters 🧭",
        "let volume lead the way 📊"
    ]
    bull_score = (1 if chg30 >= 5 else 0) + (1 if bos_dir == "up" else 0) + (1 if near_sup else 0)
    bear_score = (1 if chg30 <= -5 else 0) + (1 if bos_dir == "down" else 0) + (1 if near_res else 0)
    if bull_score > bear_score: ending = rnd.choice(endings_bull)
    elif bear_score > bull_score: ending = rnd.choice(endings_bear)
    else: ending = rnd.choice(endings_neutral)

    sector_emoji = {
        "AMD":"🖥️","NVDA":"🧠","TSM":"🔧","ASML":"🔬","QCOM":"📶","INTC":"💾","MU":"💽","TXN":"📟",
        "META":"🤖","GOOG":"🔎","AAPL":"📱","MSFT":"☁️","AMZN":"📦",
        "JNJ":"💊","UNH":"🏥","LLY":"🧪","ABBV":"🧬","MRK":"🧫",
        "MA":"💳","V":"💳","PYPL":"💸","SQ":"💸","SOFI":"🏦",
        "SNOW":"🧊","CRM":"📇","NOW":"🛠️","PLTR":"🛰️"
    }.get(ticker, "📈")

    joiners = [" — ", " · ", " — ", " • "]
    mid = rnd.choice(joiners)
    cue_txt = rnd.choice(["; ".join(cues), ", ".join(cues), " | ".join(cues)])
    line = f"{sector_emoji} {ticker}{mid}{lead}{mid}{cue_txt}; {ending}"
    return line[:280]

# CTA footer variants
CTA_POOL = [
    "Save for later 📌 · Comment your levels 💬 · See charts in carousel ➡️",
    "Tap save 📌 · Drop your take below 💬 · Full charts in carousel ➡️",
    "Save this post 📌 · Share your view 💬 · Swipe for charts ➡️",
    "Bookmark 📌 · What did we miss? 💬 · More charts inside ➡️"
]

# ------------------ FIX: define your missing picker ------------------
def choose_tickers_somehow():
    """
    Minimal deterministic picker from your defined pool.
    Keeps your existing structure; swap internals later if you have weighting elsewhere.
    """
    pool = list(COMPANY_QUERY.keys())
    rnd = random.Random(DATESTR)
    k = 6 if len(pool) >= 6 else len(pool)
    return rnd.sample(pool, k)

# ------------------ Ensure fetch/render exist (import shim) ------------------
_FETCH_RENDER_IMPORTED = False
try:
    # If you keep your real implementations elsewhere, set the correct module here:
    # from charting import fetch_one, render_single_post
    from charting import fetch_one, render_single_post   # change 'charting' if needed
    _FETCH_RENDER_IMPORTED = True
except Exception:
    try:
        from utils import fetch_one, render_single_post
        _FETCH_RENDER_IMPORTED = True
    except Exception:
        try:
            from renderer import fetch_one, render_single_post
            _FETCH_RENDER_IMPORTED = True
        except Exception:
            _FETCH_RENDER_IMPORTED = False

# ---- Fallbacks ONLY if real implementations aren't importable ----
if not _FETCH_RENDER_IMPORTED:
    # -------- robust yfinance fallback --------
    def _coerce_close_series(df: pd.DataFrame) -> pd.Series:
        """Get a 1D Close series from yfinance DataFrame (handles MultiIndex)."""
        if df is None or df.empty:
            return pd.Series(dtype=float)

        # Preferred columns
        for name in ("Close", "Adj Close", "close"):
            if name in df.columns:
                close = df[name]
                break
        else:
            close = None
            if isinstance(df.columns, pd.MultiIndex):
                for name in ("Close", "Adj Close"):
                    if name in df.columns.get_level_values(-1):
                        sub = df.xs(name, axis=1, level=-1, drop_level=False)
                        if isinstance(sub, pd.DataFrame) and sub.shape[1] >= 1:
                            close = sub.iloc[:, 0]
                            break

        if close is None:
            numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
            if not numeric_cols:
                return pd.Series(dtype=float)
            close = df[numeric_cols[0]]

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        close = pd.to_numeric(close, errors="coerce").dropna()
        return close

    def fetch_one(ticker):
        """
        Fallback: fetch yfinance data and compute the payload your code expects:
        (df,last,chg30,sup_low,sup_high,res_low,res_high,'Support','Resistance',bos_dir,bos_level,bos_idx,'D')
        """
        df = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True)
        if df is None or df.empty:
            return None

        close = _coerce_close_series(df)
        if close.empty:
            return None

        last = float(close.iloc[-1])
        if len(close) >= 30:
            base = float(close.iloc[-30])
            chg30 = float(100.0 * (last - base) / base) if base != 0 else 0.0
        else:
            chg30 = 0.0

        sup_low = float(close.min())
        sup_high = float(sup_low * 1.05)
        res_high = float(close.max())
        res_low = float(res_high * 0.95)
        bos_dir = "up" if chg30 > 5 else ("down" if chg30 < -5 else None)
        bos_level = last
        bos_idx = int(len(close) - 1)
        bos_tf = "D"

        # Align df to close index for downstream drawing
        df = df.loc[close.index]
        return (df, last, float(chg30), sup_low, sup_high, res_low, res_high,
                "Support", "Resistance", bos_dir, bos_level, bos_idx, bos_tf)

    # -------- styled candlestick fallback renderer (IBM reference) --------
    def render_single_post(out_path, ticker, payload):
        """
        Fallback renderer styled like your reference:
        - White card, rounded corners
        - Light grid
        - Candlesticks (green up / red down)
        - Shaded Support (blue) / Resistance (red) zones
        - Header: TICKER, last price, 30d change (colored), subtitle
        - Ticker logo (assets/logos/{ticker}.png) top-right (optional)
        - Brand logo bottom-right (optional)
        - Footer left: 'Resistance ~', 'Support ~', 'Not financial advice'
        """
        (df, last, chg30, sup_low, sup_high, res_low, res_high,
         sup_label, res_label, bos_dir, bos_level, bos_idx, bos_tf) = payload

        # canvas
        W, H = 1080, 1080
        BG = (245, 245, 245, 255)
        canvas = Image.new("RGBA", (W, H), BG)
        draw = ImageDraw.Draw(canvas)

        # fonts
        def _try_font(path, size):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                return None

        def _font(size, bold=False):
            # 1) Prefer your Grift family in repo
            grift_bold = _try_font("assets/fonts/Grift-Bold.ttf", size)
            grift_reg  = _try_font("assets/fonts/Grift-Regular.ttf", size)

            # 2) Then Roboto (repo or system)
            roboto_bold = (_try_font("assets/fonts/Roboto-Bold.ttf", size)
                           or _try_font("Roboto-Bold.ttf", size))
            roboto_reg  = (_try_font("assets/fonts/Roboto-Regular.ttf", size)
                           or _try_font("Roboto-Regular.ttf", size))

            if bold:
                return grift_bold or roboto_bold or ImageFont.load_default()
            else:
                return grift_reg or roboto_reg or ImageFont.load_default()

        # Size tune for IG (bigger, clean hierarchy)
       f_ticker = _font(100, bold=True)   # big ticker text
f_price  = _font(56,  bold=True)
f_delta  = _font(50,  bold=True)
f_sub    = _font(34)
f_sm     = _font(28)
f_badge  = _font(26, bold=True)

        # card
        margin = 40
        card = [margin, margin, W - margin, H - margin]
        draw.rounded_rectangle(card, radius=28, fill=(255, 255, 255, 255), outline=(230, 230, 230, 255), width=2)

        header_h = 160
        footer_h = 110
        pad_l, pad_r = 60, 50
        chart = [card[0] + pad_l, card[1] + header_h, card[2] - pad_r, card[3] - footer_h]
        cx1, cy1, cx2, cy2 = chart

        # header
        title_y = card[1] + 28
        title_x = card[0] + 28
        draw.text((title_x, title_y), ticker, fill=(20, 20, 20, 255), font=f_title)
        price_txt = f"{last:,.2f} USD"
        dy = title_y + 84
        draw.text((title_x, dy), price_txt, fill=(32, 32, 32, 255), font=f_price)
        sign_col = (18, 161, 74, 255) if chg30 >= 0 else (200, 60, 60, 255)
        change_txt = f"{chg30:+.2f}% past 30d"
        draw.text((title_x, dy + 46), change_txt, fill=sign_col, font=f_delta)
        sub_txt = "Daily chart • confluence S/R zones"
        draw.text((title_x, dy + 86), sub_txt, fill=(150, 150, 150, 255), font=f_sub)

        # ticker logo (optional)
        tlogo_path = os.path.join("assets", "logos", f"{ticker}.png")
        if os.path.exists(tlogo_path):
            try:
                tlogo = Image.open(tlogo_path).convert("RGBA")
                hmax = 80
                scale = min(1.0, hmax / max(1, tlogo.height))
                tlogo = tlogo.resize((int(tlogo.width * scale), int(tlogo.height * scale)))
                lx = card[2] - 28 - tlogo.width
                ly = title_y + 4
                canvas.alpha_composite(tlogo, (lx, ly))
            except Exception:
                pass

        # grid
        grid = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        g = ImageDraw.Draw(grid)
        for i in range(1, 5):
            y = cy1 + i * (cy2 - cy1) / 5.0
            g.line([(cx1, y), (cx2, y)], fill=(238, 238, 238, 255), width=1)
        for i in range(1, 6):
            x = cx1 + i * (cx2 - cx1) / 6.0
            g.line([(x, cy1), (x, cy2)], fill=(238, 238, 238, 255), width=1)
        canvas = Image.alpha_composite(canvas, grid)
        draw = ImageDraw.Draw(canvas)

        # prepare OHLC
        def _col(name):
            if name in df.columns: return df[name]
            if isinstance(df.columns, pd.MultiIndex) and name in df.columns.get_level_values(-1):
                sub = df.xs(name, axis=1, level=-1, drop_level=False)
                if isinstance(sub, pd.DataFrame) and sub.shape[1] >= 1:
                    return sub.iloc[:, 0]
            return None
        o = _col("Open"); h_ = _col("High"); l_ = _col("Low"); c = _col("Close")
        if c is None: c = _col("Adj Close")

        def _num(x):
            if x is None: return None
            if isinstance(x, pd.DataFrame): x = x.iloc[:, 0]
            return pd.to_numeric(x, errors="coerce")
        o, h_, l_, c = (_num(o), _num(h_), _num(l_), _num(c))

        if c is None or c.dropna().shape[0] < 2:
            out = canvas.convert("RGB"); out.save(out_path, quality=95); return

        df2 = pd.DataFrame({"O": o, "H": h_, "L": l_, "C": c}).dropna()
        if df2.empty:
            out = canvas.convert("RGB"); out.save(out_path, quality=95); return

        ymin = float(np.nanmin(df2["L"]))
        ymax = float(np.nanmax(df2["H"]))
        if not np.isfinite(ymin) or not np.isfinite(ymax) or abs(ymax - ymin) < 1e-9:
            ymin, ymax = (ymin - 0.5, ymax + 0.5) if np.isfinite(ymin) else (0, 1)

        def sx(i): return cx1 + (i / max(1, len(df2) - 1)) * (cx2 - cx1)
        def sy(v): return cy2 - ((float(v) - ymin) / (ymax - ymin)) * (cy2 - cy1)

        # shaded S/R zones
        sup_y1, sup_y2 = sy(sup_high), sy(sup_low)
        sup_rect = [cx1, min(sup_y1, sup_y2), cx2, max(sup_y1, sup_y2)]
        sup_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(sup_layer).rectangle(sup_rect, fill=(120, 160, 255, 60), outline=(120, 160, 255, 120), width=2)
        canvas = Image.alpha_composite(canvas, sup_layer)

        res_y1, res_y2 = sy(res_high), sy(res_low)
        res_rect = [cx1, min(res_y1, res_y2), cx2, max(res_y1, res_y2)]
        res_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(res_layer).rectangle(res_rect, fill=(255, 150, 150, 60), outline=(255, 150, 150, 120), width=2)
        canvas = Image.alpha_composite(canvas, res_layer)
        draw = ImageDraw.Draw(canvas)

        # candlesticks
        up_col = (26, 172, 93, 255)     # green
        dn_col = (214, 76, 76, 255)     # red
        wick_col = (150, 150, 150, 255)
        n = len(df2)
        body_px = max(3, int((cx2 - cx1) / max(60, n * 1.4)))
        half = body_px // 2
        for i, row in enumerate(df2.itertuples(index=False)):
            O, Hh, Ll, C = row
            x = sx(i)
            draw.line([(x, sy(Hh)), (x, sy(Ll))], fill=wick_col, width=1)
            col = up_col if C >= O else dn_col
            y1 = sy(max(O, C)); y2 = sy(min(O, C))
            if abs(y2 - y1) < 1: y2 = y1 + 1
            draw.rectangle([x - half, y1, x + half, y2], fill=col, outline=col)

        # BOS (optional)
        if bos_dir is not None and np.isfinite(bos_level):
            by = sy(bos_level)
            draw.line([(cx1, by), (cx2, by)], fill=(255, 210, 0, 255), width=3)

        # footer left
        footer_x = card[0] + 20
        footer_y = card[3] - 72
        draw.text((footer_x, footer_y), f"Resistance ~{res_high:.2f}", fill=(120, 120, 120, 255), font=f_sm)
        draw.text((footer_x, footer_y + 28), f"Support ~{sup_low:.2f}",  fill=(120, 120, 120, 255), font=f_sm)
        draw.text((footer_x, footer_y + 56), "Not financial advice",     fill=(160, 160, 160, 255), font=f_sm)

        # brand logo
        if BRAND_LOGO_PATH and os.path.exists(BRAND_LOGO_PATH):
            try:
                blogo = Image.open(BRAND_LOGO_PATH).convert("RGBA")
                maxh = 120
                scale = min(1.0, maxh / max(1, blogo.height))
                blogo = blogo.resize((int(blogo.width * scale), int(blogo.height * scale)))
                bx = card[2] - 24 - blogo.width
                by = card[3] - 24 - blogo.height
                canvas.alpha_composite(blogo, (bx, by))
            except Exception:
                pass

        # save
        out = canvas.convert("RGB")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out.save(out_path, quality=95)

# ------------------ Main ------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # You already choose your 6 tickers by weighted pools; keep that logic.
    tickers = choose_tickers_somehow()
    print("[info] selected tickers:", tickers)

    saved = 0
    captions = []
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
            print(f"Error:  failed for {t}: {e}")
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
