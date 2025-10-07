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
BRAND_LOGO_PATH = os.getenv("BRAND_LOGO_PATH", "assets/brand_logo.png")  # optional

# ------------------ Ticker Pool ------------------
COMPANY_QUERY = {
    "META":"Meta Platforms", "AMD":"Advanced Micro Devices", "GOOG":"Google Alphabet", "GOOGL":"Alphabet",
    "AAPL":"Apple", "MSFT":"Microsoft", "TSM":"Taiwan Semiconductor", "TSLA":"Tesla",
    "JNJ":"Johnson & Johnson", "MA":"Mastercard", "V":"Visa", "NVDA":"NVIDIA",
    "AMZN":"Amazon", "SNOW":"Snowflake", "SQ":"Block Inc", "PYPL":"PayPal", "UNH":"UnitedHealth"
}

# ------------------ Ticker Picker ------------------
def choose_tickers_somehow():
    """
    Weighted, random selection of 6 tickers from sectors, seeded by date.
    """
    tech =     ["AAPL", "MSFT", "TSLA", "NVDA", "META", "AMD", "GOOG", "GOOGL", "AMZN", "SNOW"]
    fintech =  ["MA", "V", "PYPL", "SQ"]
    health =   ["JNJ", "UNH"]
    wildcard = ["TSM"]

    rnd = random.Random(DATESTR)

    pick = []
    pick.append(rnd.choice(tech)); tech.remove(pick[-1])
    pick.append(rnd.choice(tech)); tech.remove(pick[-1])
    pick.append(rnd.choice(fintech))
    pick.append(rnd.choice(health))
    pick.append(rnd.choice(tech + fintech + health))
    pick.append(rnd.choice(tech + fintech + health + wildcard))

    return pick

# ------------------ News Helper ------------------
import requests
SESS = requests.Session()
SESS.headers.update({"User-Agent":"TWD/1.0"})

def news_headline_for(ticker):
    name = COMPANY_QUERY.get(ticker, ticker)
    if NEWSAPI_KEY:
        try:
            r = SESS.get("https://newsapi.org/v2/everything",
                         params={"q": f'"{name}" OR {ticker}', "language":"en",
                                 "sortBy":"publishedAt", "pageSize":1},
                         headers={"X-Api-Key": NEWSAPI_KEY}, timeout=8)
            if r.ok:
                d = r.json().get("articles")
                if d:
                    title = d[0].get("title") or ""
                    src = d[0].get("source",{}).get("name","")
                    return f"{title} ({src})" if title else None
        except Exception:
            pass
    try:
        items = getattr(yf.Ticker(ticker), "news", []) or []
        if items:
            t = items[0].get("title") or ""
            p = items[0].get("publisher") or ""
            return f"{t} ({p})" if t else None
    except Exception:
        pass
    return None

# ------------------ Caption Generator ------------------
def plain_english_line(ticker, headline, payload, seed=None):
    (df,last,chg30,sup_low,sup_high,res_low,res_high,
     sup_label,res_label,bos_dir,bos_level,bos_idx,bos_tf) = payload

    if seed is None:
        seed = f"{ticker}-{DATESTR}"
    rnd = random.Random(str(seed))

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

    cues = []
    if chg30 >= 8: cues.append("momentum looks strong 🔥")
    elif chg30 <= -8: cues.append("recent pullback showing ⚠️")

    def near(l, h, p, k=1.0):
        if l is None or h is None: return False
        mid = 0.5*(l+h); rng = (h-l)*k + 1e-8
        return abs(p - mid) <= 0.6*rng

    if near(sup_low, sup_high, last): cues.append("buyers defended support 🛡️")
    if near(res_low, res_high, last): cues.append("testing overhead supply 🧱")
    if bos_dir == "up": cues.append("breakout pressure building 🚀")
    if bos_dir == "down": cues.append("post-breakdown chop ⚠️")
    if not cues:
        cues = rnd.sample([
            "price action is steady","range bound but coiling",
            "watching for a decisive move soon","tightening ranges on the daily"
        ], k=1)

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
    bull_score = (chg30 >= 5) + (bos_dir == "up") + near(sup_low, sup_high, last)
    bear_score = (chg30 <= -5) + (bos_dir == "down") + near(res_low, res_high, last)
    ending = rnd.choice(endings_bull if bull_score > bear_score else endings_bear if bear_score > bull_score else endings_neutral)

    sector_emoji = {
        "AMD":"🖥️","NVDA":"🧠","TSM":"🔧","META":"🤖","GOOG":"🔎","AAPL":"📱","MSFT":"☁️","AMZN":"📦",
        "JNJ":"💊","UNH":"🏥","MA":"💳","V":"💳","PYPL":"💸","SQ":"💸","SNOW":"🧊"
    }.get(ticker, "📈")

    joiners = [" — ", " · ", " — ", " • "]
    cue_txt = rnd.choice(["; ".join(cues), ", ".join(cues), " | ".join(cues)])
    return f"{sector_emoji} {ticker}{rnd.choice(joiners)}{lead}{rnd.choice(joiners)}{cue_txt}; {ending}"[:280]

CTA_POOL = [
    "Save for later 📌 · Comment your levels 💬 · See charts in carousel ➡️",
    "Tap save 📌 · Drop your take below 💬 · Full charts in carousel ➡️",
    "Save this post 📌 · Share your view 💬 · Swipe for charts ➡️",
    "Bookmark 📌 · What did we miss? 💬 · More charts inside ➡️"
]

# ------------------ Main Entry ------------------
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
                print(f"[warn] no data for {t}, skipping")
                continue
            out_path = os.path.join(OUTPUT_DIR, f"twd_{t}_{DATESTR}.png")
            render_single_post(out_path, t, payload)
            print("done:", out_path)

            headline = news_headline_for(t)
            line = plain_english_line(t, headline, payload, seed=DATESTR)
            captions.append(line)
            saved += 1
        except Exception as e:
            print(f"[error] failed for {t}: {e}")
            traceback.print_exc()

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
