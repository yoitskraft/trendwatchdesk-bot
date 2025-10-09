#!/usr/bin/env python3
import os, sys, io, math, json, random, datetime, traceback
import numpy as np
import pandas as pd
import yfinance as yf
from PIL import Image, ImageDraw, ImageFont

# ------------------ Config & knobs ------------------
OUTPUT_DIR = os.path.abspath("output")
TODAY = datetime.date.today()
DATESTR = TODAY.strftime("%Y%m%d")

# Env knobs (tweak via workflow env)
TWD_MODE = os.getenv("TWD_MODE", "charts").lower()            # charts | posters | all (charts)
TWD_TF = os.getenv("TWD_TF", "D").upper()                     # D or W (render timeframe)
TWD_DEBUG = os.getenv("TWD_DEBUG", "0").lower() in ("1","true","on","yes")

# UI scales
TWD_UI_SCALE = float(os.getenv("TWD_UI_SCALE", "0.90"))
TWD_TEXT_SCALE = float(os.getenv("TWD_TEXT_SCALE", "0.78"))
TWD_TLOGO_SCALE = float(os.getenv("TWD_TLOGO_SCALE", "0.55"))

# brand assets
BRAND_LOGO_PATH = os.getenv("BRAND_LOGO_PATH", "assets/brand_logo.png").strip()

# ------------------ Logging ------------------
def _dbg(msg: str):
    if TWD_DEBUG:
        print(f"[debug] {msg}")

# ------------------ Utilities (columns, fonts) ------------------
def _find_col(df: pd.DataFrame, key: str):
    if df is None or df.empty: return None
    cols = df.columns
    if key in cols: return df[key]
    if isinstance(cols, pd.MultiIndex):
        for lvl in reversed(range(cols.nlevels)):
            try:
                sub = df.xs(key, axis=1, level=lvl, drop_level=False)
                if isinstance(sub, pd.DataFrame) and sub.shape[1] >= 1:
                    return sub.iloc[:, 0]
            except Exception:
                pass
    return None

def _load_font(size, bold=False):
    # Try Grift → Roboto → default
    font_candidates = []
    if bold:
        font_candidates += ["assets/fonts/Grift-Bold.ttf", "assets/fonts/Roboto-Bold.ttf"]
    else:
        font_candidates += ["assets/fonts/Grift-Regular.ttf", "assets/fonts/Roboto-Regular.ttf"]
    for p in font_candidates:
        if os.path.exists(p):
            try: return ImageFont.truetype(p, size)
            except Exception: pass
    return ImageFont.load_default()

# ------------------ Ticker selection (keep your pools later) ------------------
DEFAULT_TICKERS = [
    "AAPL","MSFT","META","AMZN","GOOGL","TSLA","NVDA",
    "AMD","TSM","ASML","QCOM","INTC","MU","TXN",
    "JNJ","UNH","LLY","ABBV","MRK",
    "MA","V","PYPL","SQ","SOFI",
    "SNOW","CRM","NOW","PLTR",
    "NFLX","ADBE","NKE","COST","PEP","KO","XOM","CVX"
]

def choose_tickers_somehow(n=6, seed=None):
    rnd = random.Random(seed or DATESTR)
    return rnd.sample(DEFAULT_TICKERS, k=n)

# ------------------ OHLC normalization ------------------
def _get_ohlc_df(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    _dbg(f"_get_ohlc_df in: cols={list(df.columns)} len={len(df)}")

    o = _find_col(df,"Open"); h = _find_col(df,"High"); l = _find_col(df,"Low")
    c = _find_col(df,"Close")
    if c is None or (hasattr(c, "dropna") and c.dropna().empty):
        c = _find_col(df,"Adj Close")

    if c is None or (hasattr(c, "dropna") and c.dropna().empty):
        _dbg("Close/Adj Close missing; _get_ohlc_df -> None")
        return None

    idx = c.index
    def _al(x):
        return pd.to_numeric(x, errors="coerce").reindex(idx) if x is not None else None

    c = _al(c).astype(float)
    # Synthesize missing from Close so we can still render
    o = _al(o) if o is not None else c.copy()
    h = _al(h) if h is not None else c.copy()
    l = _al(l) if l is not None else c.copy()

    out = pd.DataFrame({"Open":o,"High":h,"Low":l,"Close":c}).dropna()
    if out.empty:
        _dbg("_get_ohlc_df constructed empty")
        return None
    _dbg(f"_get_ohlc_df out len={len(out)}")
    return out

# ------------------ Swings & support (4H driven) ------------------
def _swing_points(series: pd.Series, w=3):
    s = series.values
    idxs = series.index.to_list()
    highs, lows = [], []
    for i in range(w, len(s)-w):
        left = s[i-w:i]; right = s[i+1:i+1+w]
        if s[i] >= max(left) and s[i] >= max(right): highs.append((idxs[i], s[i]))
        if s[i] <= min(left) and s[i] <= min(right): lows.append((idxs[i], s[i]))
    return highs, lows

def pick_support_level_from_4h(df_4h: pd.DataFrame, trend_bullish: bool, w=3, pct_tol=0.004, atr_len=14):
    if df_4h is None or df_4h.empty: return (None, None)
    cl = pd.to_numeric(df_4h["Close"], errors="coerce").dropna()
    if cl.empty: return (None, None)

    last = float(cl.iloc[-1])
    H, L = _swing_points(cl, w=w)  # swing points based on close
    swing_highs = sorted([v for (_, v) in H if v <= last], reverse=True)
    swing_lows  = sorted([v for (_, v) in L if v <= last], reverse=True)

    # ATR-ish zone thickness
    rng = (pd.to_numeric(df_4h["High"], errors="coerce") - pd.to_numeric(df_4h["Low"], errors="coerce")).rolling(atr_len).mean().dropna()
    atr = float(rng.iloc[-1]) if not rng.empty else last * pct_tol
    thickness = max(atr, last * pct_tol)

    if trend_bullish:
        if swing_highs:
            lvl = swing_highs[0]
            return (lvl - thickness*0.5, lvl + thickness*0.5)
    else:
        # bearish: last swing low as possible bounce
        if swing_lows:
            lvl = swing_lows[0]
            return (lvl - thickness*0.5, lvl + thickness*0.5)

    return (None, None)

# ------------------ Fetch pack ------------------
def fetch_one(ticker):
    # Daily 1y
    try:
        df_d = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=True)
    except Exception as e:
        _dbg(f"{ticker} history daily exception: {e}")
        df_d = None
    if df_d is None or df_d.empty:
        _dbg(f"{ticker} history empty; download fallback")
        df_d = yf.download(ticker, period="1y", interval="1d", auto_adjust=True)

    _dbg(f"{ticker}: daily len={0 if df_d is None else len(df_d)}")
    ohlc_d = _get_ohlc_df(df_d)
    if ohlc_d is None or ohlc_d.empty:
        _dbg(f"{ticker}: ohlc_d empty → skip")
        return None

    close_d = ohlc_d["Close"].dropna()
    if close_d.shape[0] < 2:
        _dbg(f"{ticker}: too few daily closes")
        return None

    last = float(close_d.iloc[-1])
    base_val = float(close_d.iloc[-31]) if close_d.shape[0] > 30 else float(close_d.iloc[0])
    chg30 = 100.0 * (last - base_val) / base_val if base_val != 0 else 0.0
    prev = float(close_d.iloc[-2])
    chg1d = 100.0 * (last - prev) / prev if prev != 0 else 0.0

    # 60m → 4H for support
    sup_low, sup_high = (None, None)
    try:
        df_60 = yf.Ticker(ticker).history(period="6mo", interval="60m", auto_adjust=True)
        if df_60 is None or df_60.empty:
            _dbg(f"{ticker}: 60m empty; download fallback")
            df_60 = yf.download(ticker, period="6mo", interval="60m", auto_adjust=True)

        if df_60 is not None and not df_60.empty:
            ohlc_60 = _get_ohlc_df(df_60)
            if ohlc_60 is not None and not ohlc_60.empty:
                cutoff = ohlc_60.index.max() - pd.Timedelta(days=120)
                df_60_clip = ohlc_60.loc[ohlc_60.index >= cutoff].copy()
                if not df_60_clip.empty:
                    df_4h = df_60_clip.resample("4H").agg(
                        {"Open":"first","High":"max","Low":"min","Close":"last"}
                    ).dropna()
                    _dbg(f"{ticker}: 4H len={len(df_4h)}")
                    if not df_4h.empty:
                        trend_bullish = (chg30 > 0)
                        sup_low, sup_high = pick_support_level_from_4h(
                            df_4h, trend_bullish, w=3, pct_tol=0.004, atr_len=14
                        )
    except Exception as e:
        _dbg(f"{ticker}: support calc error (ignored): {e}")

    # Render timeframe
    if TWD_TF == "W":
        df_render = ohlc_d.resample("W-FRI").agg(
            {"Open":"first","High":"max","Low":"min","Close":"last"}
        ).dropna().tail(60)
        tf_tag = "W"
    else:
        df_render = ohlc_d.dropna().tail(260)
        tf_tag = "D"

    if df_render is None or df_render.empty:
        _dbg(f"{ticker}: df_render empty after timeframe transform")
        return None

    return (df_render, last, float(chg30), sup_low, sup_high, tf_tag, float(chg1d))

# ------------------ Caption builder ------------------
SECTOR_EMOJI = {
    "AMD":"🖥️","NVDA":"🧠","TSM":"🔧","ASML":"🔬","QCOM":"📶","INTC":"💾","MU":"💽","TXN":"📟",
    "META":"🤖","GOOG":"🔎","GOOGL":"🔎","AAPL":"📱","MSFT":"☁️","AMZN":"📦","NFLX":"🎬","ADBE":"🎨",
    "JNJ":"💊","UNH":"🏥","LLY":"🧪","ABBV":"🧬","MRK":"🧫",
    "MA":"💳","V":"💳","PYPL":"💸","SQ":"💸","SOFI":"🏦",
    "SNOW":"🧊","CRM":"📇","NOW":"🛠️","PLTR":"🛰️"
}

CTA_POOL = [
    "Save for later 📌 · Comment your levels 💬 · See charts in carousel ➡️",
    "Tap save 📌 · Drop your take below 💬 · Full charts in carousel ➡️",
    "Bookmark 📌 · What did we miss? 💬 · More charts inside ➡️"
]

def caption_line(ticker, last, chg30, chg1d, sup_low, sup_high, seed=None):
    rnd = random.Random(seed or DATESTR)
    cues = []
    if chg30 >= 8: cues.append("momentum looks strong 🔥")
    elif chg30 <= -8: cues.append("recent pullback showing ⚠️")

    # proximity heuristic to support
    def near_sup(l, h, p):
        if l is None or h is None: return False
        mid = 0.5*(l+h); rng=(h-l)+1e-8
        return abs(p-mid) <= 0.6*rng
    if near_sup(sup_low, sup_high, last): cues.append("buyers defended support 🛡️")

    if not cues:
        cues = rnd.sample([
            "price action is steady", "range bound but coiling",
            "watching for a decisive move soon", "tightening ranges on the daily"
        ], k=1)

    endings_bull = [
        "could have more room if momentum sticks ✅",
        "setups lean constructive here 📈",
        "watch for follow-through on strength 🔎"
    ]
    endings_bear = [
        "risk of rejection—watch reactions at key levels 👀",
        "tone is cautious; patience helps here 🧊",
        "relief bounces possible, trend still mixed ⚖️"
    ]
    ending = random.choice(endings_bull if chg30 > 0 else endings_bear)

    emj = SECTOR_EMOJI.get(ticker, "📈")
    parts = [
        f"{emj} {ticker}",
        f"{last:,.2f} USD",
        f"{chg30:+.2f}% (30d)"
    ]
    head = " — ".join(parts)
    cue_txt = rnd.choice(["; ".join(cues), ", ".join(cues), " · ".join(cues)])
    return f"{head} — {cue_txt}; {ending}"

# ------------------ Rendering ------------------
def render_single_post(out_path, ticker, payload):
    (df, last, chg30, sup_low, sup_high, tf_tag, chg1d) = payload

    # canvas
    W, H = 1080, 1080
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # scaled layout
    margin = int(40 * TWD_UI_SCALE)
    header_h = int(190 * TWD_UI_SCALE)
    footer_h = int(90 * TWD_UI_SCALE)
    pad_l, pad_r = int(58 * TWD_UI_SCALE), int(52 * TWD_UI_SCALE)

    card = [margin, margin, W - margin, H - margin]
    chart = [card[0] + pad_l, card[1] + header_h, card[2] - pad_r, card[3] - footer_h]
    cx1, cy1, cx2, cy2 = chart

    # fonts
    f_ticker = _load_font(int(76 * TWD_TEXT_SCALE), bold=True)
    f_price  = _load_font(int(44 * TWD_TEXT_SCALE), bold=False)
    f_delta  = _load_font(int(38 * TWD_TEXT_SCALE), bold=True)
    f_sub    = _load_font(int(26 * TWD_TEXT_SCALE), bold=False)
    f_sm     = _load_font(int(24 * TWD_TEXT_SCALE), bold=False)

    # header text block (left)
    title_x = card[0] + int(22 * TWD_UI_SCALE)
    title_y = card[1] + int(20 * TWD_UI_SCALE)

    draw.text((title_x, title_y), ticker, fill=(20,20,20,255), font=f_ticker)
    dy = title_y + int(70 * TWD_TEXT_SCALE) + int(12 * TWD_UI_SCALE)
    draw.text((title_x, dy), f"{last:,.2f} USD", fill=(30,30,30,255), font=f_price)
    dy2 = dy + int(44 * TWD_TEXT_SCALE)
    sign_col = (18, 161, 74, 255) if chg30 >= 0 else (200, 60, 60, 255)
    draw.text((title_x, dy2), f"{chg30:+.2f}% past 30d", fill=sign_col, font=f_delta)
    dy3 = dy2 + int(40 * TWD_TEXT_SCALE)
    sub_txt = "Weekly chart • key zones" if tf_tag == "W" else "Daily chart • key zones"
    draw.text((title_x, dy3), sub_txt, fill=(140,140,140,255), font=f_sub)

    # ticker logo (optional)
    tlogo_path = os.path.join("assets","logos",f"{ticker}.png")
    if os.path.exists(tlogo_path):
        try:
            tlogo = Image.open(tlogo_path).convert("RGBA")
            hmax = int(72 * TWD_TLOGO_SCALE)
            scale = min(1.0, hmax / max(1, tlogo.height))
            tlogo = tlogo.resize((int(tlogo.width*scale), int(tlogo.height*scale)))
            lx = card[2] - int(24 * TWD_UI_SCALE) - tlogo.width
            ly = title_y + 4
            canvas.alpha_composite(tlogo, (lx, ly))
        except Exception:
            pass

    # grid
    grid = Image.new("RGBA", (W, H), (0,0,0,0))
    g = ImageDraw.Draw(grid)
    for i in range(1, 5):
        y = cy1 + i * (cy2 - cy1) / 5.0
        g.line([(cx1, y), (cx2, y)], fill=(238,238,238,255), width=1)
    for i in range(1, 6):
        x = cx1 + i * (cx2 - cx1) / 6.0
        g.line([(x, cy1), (x, cy2)], fill=(238,238,238,255), width=1)
    canvas = Image.alpha_composite(canvas, grid)
    draw = ImageDraw.Draw(canvas)

    # OHLC to draw
    df2 = df[["Open","High","Low","Close"]].dropna()
    if df2.empty:
        out = canvas.convert("RGB"); os.makedirs(os.path.dirname(out_path), exist_ok=True); out.save(out_path, quality=92); return

    # y-range
    ymin = float(np.nanmin(df2["Low"])); ymax = float(np.nanmax(df2["High"]))
    if not np.isfinite(ymin) or not np.isfinite(ymax) or abs(ymax-ymin) < 1e-9:
        ymin, ymax = (ymin - 0.5, ymax + 0.5) if np.isfinite(ymin) else (0, 1)

    def sx(i): return cx1 + (i / max(1, len(df2)-1)) * (cx2 - cx1)
    def sy(v): return cy2 - ((float(v) - ymin) / (ymax - ymin)) * (cy2 - cy1)

    # support zone (blue), only if present & inside y-range
    if (sup_low is not None) and (sup_high is not None):
        sup_y1, sup_y2 = sy(sup_high), sy(sup_low)
        sup_rect = [cx1, min(sup_y1, sup_y2), cx2, max(sup_y1, sup_y2)]
        sup_layer = Image.new("RGBA", (W, H), (0,0,0,0))
        ImageDraw.Draw(sup_layer).rectangle(sup_rect, fill=(120,160,255,60), outline=(120,160,255,120), width=2)
        canvas = Image.alpha_composite(canvas, sup_layer)
        draw = ImageDraw.Draw(canvas)

    # candlesticks
    up_col, dn_col, wick_col = (26,172,93,255), (214,76,76,255), (150,150,150,255)
    n = len(df2)
    body_px = max(3, int((cx2 - cx1) / max(60, n * 1.35)))
    half = body_px // 2
    for i, row in enumerate(df2.itertuples(index=False)):
        O, Hh, Ll, C = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        x = sx(i)
        draw.line([(x, sy(Hh)), (x, sy(Ll))], fill=wick_col, width=1)
        col = up_col if C >= O else dn_col
        y1 = sy(max(O, C)); y2 = sy(min(O, C))
        if abs(y2 - y1) < 1: y2 = y1 + 1
        draw.rectangle([x - half, y1, x + half, y2], fill=col, outline=col)

    # footer left (simple)
    foot_x = card[0] + int(18 * TWD_UI_SCALE)
    foot_y = card[3] - int(70 * TWD_UI_SCALE)
    if (sup_low is not None) and (sup_high is not None):
        draw.text((foot_x, foot_y), f"Support zone shown", fill=(120,120,120,255), font=f_sm)
    else:
        draw.text((foot_x, foot_y), f"Support zone n/a", fill=(160,160,160,255), font=f_sm)
    draw.text((foot_x, foot_y + int(26*TWD_UI_SCALE)), "Not financial advice", fill=(160,160,160,255), font=f_sm)

    # brand logo bottom-right
    if BRAND_LOGO_PATH and os.path.exists(BRAND_LOGO_PATH):
        try:
            blogo = Image.open(BRAND_LOGO_PATH).convert("RGBA")
            maxh = int(120 * TWD_UI_SCALE)
            scale = min(1.0, maxh / max(1, blogo.height))
            blogo = blogo.resize((int(blogo.width*scale), int(blogo.height*scale)))
            bx = card[2] - int(22*TWD_UI_SCALE) - blogo.width
            by = card[3] - int(22*TWD_UI_SCALE) - blogo.height
            canvas.alpha_composite(blogo, (bx, by))
        except Exception:
            pass

    # save
    out = canvas.convert("RGB")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.save(out_path, quality=92)

# ------------------ Main ------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # pick 6 (seeded by date for variety + reproducibility)
    tickers = choose_tickers_somehow(n=6, seed=DATESTR)
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
            _dbg(f"saving {out_path}")
            try:
                render_single_post(out_path, t, payload)
            except Exception as re:
                # Diagnostic image if render fails
                _dbg(f"render error for {t}: {re}")
                img = Image.new("RGB", (1080,1080), "white")
                d = ImageDraw.Draw(img)
                d.text((40,40), f"Render error: {t}\n{re}", fill="black")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                img.save(out_path, quality=85)
            saved += 1

            # caption
            (df, last, chg30, sup_low, sup_high, tf_tag, chg1d) = payload
            line = caption_line(t, last, chg30, chg1d, sup_low, sup_high, seed=DATESTR)
            captions.append(line)

        except Exception as e:
            print(f"Error: failed for {t}: {e}")
            traceback.print_exc()

    print(f"[info] saved images: {saved}")

    # caption file
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
