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

# ------------------ Ensure fetch/render exist (import shim, unchanged behavior if your real ones are present) ------------------
_FETCH_RENDER_IMPORTED = False
try:
    from charting import fetch_one, render_single_post  # if you have them
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

# ---- Fallbacks ONLY if your real implementations aren’t importable ----
if not _FETCH_RENDER_IMPORTED:
    def _coerce_close_series(df: pd.DataFrame) -> pd.Series:
        """
        Robustly get a 1D Close series from yfinance DataFrame, even if columns are MultiIndex.
        """
        if df is None or df.empty:
            return pd.Series(dtype=float)

        # Prefer the 'Close' column
        close = df.get("Close", None)
        if close is None:
            # some environments: lowercase or 'Adj Close' only
            if "Adj Close" in df.columns:
                close = df["Adj Close"]
            elif "close" in df.columns:
                close = df["close"]
            else:
                # last resort: try first numeric column
                numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
                if not numeric_cols:
                    return pd.Series(dtype=float)
                close = df[numeric_cols[0]]

        # If this is a DataFrame (MultiIndex case), take the first column
        if isinstance(close, pd.DataFrame):
            if close.shape[1] == 0:
                return pd.Series(dtype=float)
            close = close.iloc[:, 0]

        # ensure float dtype
        close = pd.to_numeric(close, errors="coerce").dropna()
        return close

    def fetch_one(ticker):
        """
        Basic yfinance fetch and trivial levels (stable scalar math).
        Returns tuple shaped exactly as your code expects.
        """
        # Make auto_adjust explicit to avoid FutureWarning variance
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

        # Ensure chg30 is scalar float before comparisons
        bos_dir = "up" if float(chg30) > 5 else ("down" if float(chg30) < -5 else None)
        bos_level = last
        bos_idx = int(len(close) - 1)
        bos_tf = "D"

        # Build a df aligned with close for downstream functions that expect df
        # (keep original df columns so your existing render code still works)
        df = df.loc[close.index]

        return (df, last, float(chg30), sup_low, sup_high, res_low, res_high,
                "Support", "Resistance", bos_dir, bos_level, bos_idx, bos_tf)

    def render_single_post(out_path, ticker, payload):
        """
        Minimal, safe 1080x1080 render so pipeline always completes if your real renderer isn't imported.
        """
        (df, last, chg30, sup_low, sup_high, res_low, res_high,
         sup_label, res_label, bos_dir, bos_level, bos_idx, bos_tf) = payload

        img = Image.new("RGB", (1080, 1080), (250, 250, 250))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 36)
        except Exception:
            font = ImageFont.load_default()

        draw.text((40, 40), f"{ticker}  {last:.2f}", fill=(0, 0, 0), font=font)
        draw.rectangle([30, 30, 1050, 1050], outline=(200,200,200), width=3)

        # Optional brand logo
        if BRAND_LOGO_PATH and os.path.exists(BRAND_LOGO_PATH):
            try:
                logo = Image.open(BRAND_LOGO_PATH).convert("RGBA").resize((120, 120))
                img.paste(logo, (930, 930), logo)
            except Exception:
                pass

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        img.save(out_path)

# ------------------ Main ------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tickers = choose_tickers_somehow()  # <- now defined
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
