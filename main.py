#!/usr/bin/env python3
import os, sys, io, math, json, random, datetime, traceback, re, glob, hashlib
import numpy as np
import pandas as pd
import yfinance as yf
from PIL import Image, ImageDraw, ImageFont

# ------------------ Config ------------------
OUTPUT_DIR = os.path.abspath("output")
TODAY = datetime.date.today()
DATESTR = TODAY.strftime("%Y%m%d")
UTC_NOW = datetime.datetime.utcnow()

# Modes
TWD_MODE = os.getenv("TWD_MODE", "charts").lower()            # charts | posters | all
TWD_TF = os.getenv("TWD_TF", "D").upper()                     # D or W
TWD_DEBUG = os.getenv("TWD_DEBUG", "0").lower() in ("1","true","on","yes")

# UI scales (charts)
TWD_UI_SCALE = float(os.getenv("TWD_UI_SCALE", "0.90"))
TWD_TEXT_SCALE = float(os.getenv("TWD_TEXT_SCALE", "0.78"))
TWD_TLOGO_SCALE = float(os.getenv("TWD_TLOGO_SCALE", "0.55"))

# brand
BRAND_LOGO_PATH = os.getenv("BRAND_LOGO_PATH", "assets/brand_logo.png").strip()

# Posters knobs
TWD_BREAKING_ON = os.getenv("TWD_BREAKING_ON", "on").lower() in ("on","1","true","yes")
TWD_BREAKING_RECENCY_MIN = int(os.getenv("TWD_BREAKING_RECENCY_MIN", "90"))
TWD_BREAKING_MIN_SOURCES = int(os.getenv("TWD_BREAKING_MIN_SOURCES", "3"))
TWD_BREAKING_MIN_INTERVAL_MIN = int(os.getenv("TWD_BREAKING_MIN_INTERVAL_MIN", "30"))
TWD_BREAKING_FALLBACK = os.getenv("TWD_BREAKING_FALLBACK", "off").lower() in ("on","1","true","yes")

# Paths
STATE_DIR = os.path.abspath("state")
SEEN_FILE = os.path.join(STATE_DIR, "seen_stories.json")
POSTERS_DIR = os.path.join(OUTPUT_DIR, "posters")

def _dbg(msg): 
    if TWD_DEBUG: print(f"[debug] {msg}")

# ------------------ Fonts ------------------
def _load_font(size, bold=False):
    candidates = []
    if bold:
        candidates += ["assets/fonts/Grift-Bold.ttf", "assets/fonts/Roboto-Bold.ttf"]
    else:
        candidates += ["assets/fonts/Grift-Regular.ttf", "assets/fonts/Roboto-Regular.ttf"]
    for p in candidates:
        if os.path.exists(p):
            try: return ImageFont.truetype(p, size)
            except Exception: pass
    return ImageFont.load_default()

# ------------------ Ticker selection ------------------
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

# ------------------ OHLC helpers ------------------
def _find_col(df: pd.DataFrame, key: str):
    if df is None or df.empty: return None
    if key in df.columns: return df[key]
    if isinstance(df.columns, pd.MultiIndex):
        for lvl in reversed(range(df.columns.nlevels)):
            try:
                sub = df.xs(key, axis=1, level=lvl, drop_level=False)
                if isinstance(sub, pd.DataFrame) and sub.shape[1] >= 1:
                    return sub.iloc[:, 0]
            except Exception:
                pass
    return None

def _get_ohlc_df(df: pd.DataFrame):
    if df is None or df.empty: return None
    o = _find_col(df,"Open"); h = _find_col(df,"High"); l = _find_col(df,"Low")
    c = _find_col(df,"Close") or _find_col(df,"Adj Close")
    if c is None: return None
    idx = c.index
    def _al(x): return pd.to_numeric(x, errors="coerce").reindex(idx) if x is not None else None
    c = _al(c).astype(float)
    o = _al(o) if o is not None else c.copy()
    h = _al(h) if h is not None else c.copy()
    l = _al(l) if l is not None else c.copy()
    out = pd.DataFrame({"Open":o,"High":h,"Low":l,"Close":c}).dropna()
    return out if not out.empty else None

# ------------------ Swings & support (4H) ------------------
def _swing_points(series: pd.Series, w=3):
    s = series.values; idxs = series.index.to_list()
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
    H, L = _swing_points(cl, w=w)
    swing_highs = sorted([v for (_, v) in H if v <= last], reverse=True)
    swing_lows  = sorted([v for (_, v) in L if v <= last], reverse=True)
    rng = (pd.to_numeric(df_4h["High"], errors="coerce") - pd.to_numeric(df_4h["Low"], errors="coerce")).rolling(atr_len).mean().dropna()
    atr = float(rng.iloc[-1]) if not rng.empty else last * pct_tol
    thickness = max(atr, last * pct_tol)
    if trend_bullish and swing_highs:
        lvl = swing_highs[0]; return (lvl - thickness*0.5, lvl + thickness*0.5)
    if (not trend_bullish) and swing_lows:
        lvl = swing_lows[0]; return (lvl - thickness*0.5, lvl + thickness*0.5)
    return (None, None)

# ------------------ Charts fetch ------------------
def fetch_one_chart(ticker):
    try:
        df_d = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=True)
    except Exception as e:
        _dbg(f"{ticker} daily history error: {e}"); df_d = None
    if df_d is None or df_d.empty:
        df_d = yf.download(ticker, period="1y", interval="1d", auto_adjust=True)

    ohlc_d = _get_ohlc_df(df_d)
    if ohlc_d is None or ohlc_d.empty: return None

    close_d = ohlc_d["Close"].dropna()
    if close_d.shape[0] < 2: return None
    last = float(close_d.iloc[-1])
    base_val = float(close_d.iloc[-31]) if close_d.shape[0] > 30 else float(close_d.iloc[0])
    chg30 = 100.0 * (last - base_val) / base_val if base_val != 0 else 0.0
    prev = float(close_d.iloc[-2])
    chg1d = 100.0 * (last - prev) / prev if prev != 0 else 0.0

    sup_low = sup_high = None
    try:
        df_60 = yf.Ticker(ticker).history(period="6mo", interval="60m", auto_adjust=True)
        if df_60 is None or df_60.empty:
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
                    if not df_4h.empty:
                        trend_bullish = (chg30 > 0)
                        sup_low, sup_high = pick_support_level_from_4h(df_4h, trend_bullish)
    except Exception as e:
        _dbg(f"{ticker} support calc err: {e}")

    if TWD_TF == "W":
        df_render = ohlc_d.resample("W-FRI").agg({"Open":"first","High":"max","Low":"min","Close":"last"}).dropna().tail(60)
        tf_tag = "W"
    else:
        df_render = ohlc_d.dropna().tail(260)
        tf_tag = "D"

    if df_render is None or df_render.empty: return None
    return (df_render, last, float(chg30), sup_low, sup_high, tf_tag, float(chg1d))

# ------------------ Captions ------------------
SECTOR_EMOJI = {
    "AMD":"🖥️","NVDA":"🧠","TSM":"🔧","ASML":"🔬","QCOM":"📶","INTC":"💾","MU":"💽","TXN":"📟",
    "META":"🤖","GOOG":"🔎","GOOGL":"🔎","AAPL":"📱","MSFT":"☁️","AMZN":"📦","NFLX":"🎬","ADBE":"🎨",
    "JNJ":"💊","UNH":"🏥","LLY":"🧪","ABBV":"🧬","MRK":"🧫","MA":"💳","V":"💳","PYPL":"💸","SQ":"💸","SOFI":"🏦",
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
    def near_sup(l,h,p):
        if l is None or h is None: return False
        mid=0.5*(l+h); rng=(h-l)+1e-8
        return abs(p-mid)<=0.6*rng
    if near_sup(sup_low, sup_high, last): cues.append("buyers defended support 🛡️")
    if not cues:
        cues = rnd.sample(["price action is steady","range bound but coiling","watching for a decisive move soon","tightening ranges on the daily"], k=1)
    ending = random.choice(["could have more room if momentum sticks ✅","setups lean constructive here 📈","watch for follow-through on strength 🔎"] if chg30>0 else ["risk of rejection—watch reactions at key levels 👀","tone is cautious; patience helps here 🧊","relief bounces possible, trend still mixed ⚖️"])
    emj = SECTOR_EMOJI.get(ticker, "📈")
    head = " — ".join([f"{emj} {ticker}", f"{last:,.2f} USD", f"{chg30:+.2f}% (30d)"])
    cue_txt = rnd.choice(["; ".join(cues), ", ".join(cues), " · ".join(cues)])
    return f"{head} — {cue_txt}; {ending}"

# ------------------ Chart renderer ------------------
def render_single_post(out_path, ticker, payload):
    (df, last, chg30, sup_low, sup_high, tf_tag, chg1d) = payload
    W, H = 1080, 1080
    canvas = Image.new("RGBA", (W, H), (255,255,255,255))
    draw = ImageDraw.Draw(canvas)

    margin = int(40 * TWD_UI_SCALE)
    header_h = int(190 * TWD_UI_SCALE)
    footer_h = int(90 * TWD_UI_SCALE)
    pad_l, pad_r = int(58 * TWD_UI_SCALE), int(52 * TWD_UI_SCALE)

    card = [margin, margin, W - margin, H - margin]
    chart = [card[0] + pad_l, card[1] + header_h, card[2] - pad_r, card[3] - footer_h]
    cx1, cy1, cx2, cy2 = chart

    f_ticker = _load_font(int(76 * TWD_TEXT_SCALE), bold=True)
    f_price  = _load_font(int(44 * TWD_TEXT_SCALE), bold=False)
    f_delta  = _load_font(int(38 * TWD_TEXT_SCALE), bold=True)
    f_sub    = _load_font(int(26 * TWD_TEXT_SCALE), bold=False)
    f_sm     = _load_font(int(24 * TWD_TEXT_SCALE), bold=False)

    title_x = card[0] + int(22 * TWD_UI_SCALE)
    title_y = card[1] + int(20 * TWD_UI_SCALE)
    draw.text((title_x, title_y), ticker, fill=(20,20,20,255), font=f_ticker)
    dy = title_y + int(70 * TWD_TEXT_SCALE) + int(12 * TWD_UI_SCALE)
    draw.text((title_x, dy), f"{last:,.2f} USD", fill=(30,30,30,255), font=f_price)
    dy2 = dy + int(44 * TWD_TEXT_SCALE)
    sign_col = (18,161,74,255) if chg30 >= 0 else (200,60,60,255)
    draw.text((title_x, dy2), f"{chg30:+.2f}% past 30d", fill=sign_col, font=f_delta)
    dy3 = dy2 + int(40 * TWD_TEXT_SCALE)
    draw.text((title_x, dy3), ("Weekly chart" if tf_tag=="W" else "Daily chart") + " • key zones", fill=(140,140,140,255), font=f_sub)

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
        except Exception: pass

    grid = Image.new("RGBA", (W, H), (0,0,0,0)); g = ImageDraw.Draw(grid)
    for i in range(1, 5):
        y = cy1 + i*(cy2-cy1)/5.0; g.line([(cx1,y),(cx2,y)], fill=(238,238,238,255), width=1)
    for i in range(1, 6):
        x = cx1 + i*(cx2-cx1)/6.0; g.line([(x,cy1),(x,cy2)], fill=(238,238,238,255), width=1)
    canvas = Image.alpha_composite(canvas, grid); draw = ImageDraw.Draw(canvas)

    df2 = df[["Open","High","Low","Close"]].dropna()
    if df2.empty:
        out = canvas.convert("RGB"); os.makedirs(os.path.dirname(out_path), exist_ok=True); out.save(out_path, quality=92); return

    ymin = float(np.nanmin(df2["Low"])); ymax = float(np.nanmax(df2["High"]))
    if not np.isfinite(ymin) or not np.isfinite(ymax) or abs(ymax-ymin) < 1e-9: ymin, ymax = (0,1)

    def sx(i): return cx1 + (i / max(1, len(df2)-1)) * (cx2 - cx1)
    def sy(v): return cy2 - ((float(v) - ymin) / (ymax - ymin)) * (cy2 - cy1)

    if (sup_low is not None) and (sup_high is not None):
        sup_y1, sup_y2 = sy(sup_high), sy(sup_low)
        sup_rect = [cx1, min(sup_y1, sup_y2), cx2, max(sup_y1, sup_y2)]
        sup_layer = Image.new("RGBA", (W, H), (0,0,0,0))
        ImageDraw.Draw(sup_layer).rectangle(sup_rect, fill=(120,160,255,60), outline=(120,160,255,120), width=2)
        canvas = Image.alpha_composite(canvas, sup_layer); draw = ImageDraw.Draw(canvas)

    up_col, dn_col, wick_col = (26,172,93,255),(214,76,76,255),(150,150,150,255)
    n = len(df2); body_px = max(3, int((cx2 - cx1) / max(60, n * 1.35))); half = body_px // 2
    for i, row in enumerate(df2.itertuples(index=False)):
        O,Hh,Ll,C = float(row[0]),float(row[1]),float(row[2]),float(row[3])
        x = sx(i)
        draw.line([(x, sy(Hh)), (x, sy(Ll))], fill=wick_col, width=1)
        col = up_col if C >= O else dn_col
        y1 = sy(max(O, C)); y2 = sy(min(O, C)); 
        if abs(y2-y1) < 1: y2 = y1+1
        draw.rectangle([x-half, y1, x+half, y2], fill=col, outline=col)

    foot_x = card[0] + int(18*TWD_UI_SCALE); foot_y = card[3] - int(70*TWD_UI_SCALE)
    draw.text((foot_x, foot_y), "Support zone shown" if (sup_low is not None and sup_high is not None) else "Support zone n/a", fill=(120,120,120,255), font=_load_font(int(24*TWD_TEXT_SCALE)))
    draw.text((foot_x, foot_y + int(26*TWD_UI_SCALE)), "Not financial advice", fill=(160,160,160,255), font=_load_font(int(24*TWD_TEXT_SCALE)))

    if BRAND_LOGO_PATH and os.path.exists(BRAND_LOGO_PATH):
        try:
            blogo = Image.open(BRAND_LOGO_PATH).convert("RGBA")
            maxh = int(120 * TWD_UI_SCALE); scale = min(1.0, maxh / max(1, blogo.height))
            blogo = blogo.resize((int(blogo.width*scale), int(blogo.height*scale)))
            bx = card[2] - int(22*TWD_UI_SCALE) - blogo.width; by = card[3] - int(22*TWD_UI_SCALE) - blogo.height
            canvas.alpha_composite(blogo, (bx, by))
        except Exception: pass

    out = canvas.convert("RGB"); os.makedirs(os.path.dirname(out_path), exist_ok=True); out.save(out_path, quality=92)

# ------------------ Posters: news fetch & cluster ------------------
# BASE watchlist (overridable via env)
WATCHLIST = [
    "AAPL","MSFT","META","AMZN","GOOGL","TSLA","NVDA","AMD",
    "SPY","QQQ","DIA","IWM","GLD","SLV","USO","UNG"
]
# allow env override
WATCHLIST_ENV = os.getenv("TWD_WATCHLIST", "").strip()
if WATCHLIST_ENV:
    WATCHLIST = [s.strip().upper() for s in WATCHLIST_ENV.split(",") if s.strip()]
    _dbg(f"Using env watchlist: {WATCHLIST}")

# Fallback “general news” symbols to mimic Yahoo top headlines
FALLBACK_SYMBOLS = [
    # Indices / macro
    "^GSPC","^IXIC","^DJI","^VIX",
    # Commodities / FX / crypto proxies
    "GC=F","SI=F","CL=F","DX-Y.NYB","BTC-USD","ETH-USD",
    # Mega-caps & bellwethers
    "META","NVDA","AMD","AAPL","MSFT","AMZN","TSLA",
    "JPM","BAC","XOM","CVX","WMT","TGT","DIS","NFLX","NKE"
]

PREFERRED_SOURCES = [
    "Reuters","Bloomberg","Financial Times","The Wall Street Journal","WSJ",
    "CNBC","MarketWatch","Barron's","Yahoo Finance","The Verge","The New York Times"
]

def _norm_headline(h: str) -> str:
    h = (h or "").strip().lower()
    h = re.sub(r"[^a-z0-9 ]+", "", h)
    return " ".join(h.split()[:10])

def _news_age_minutes(ts_epoch: float) -> float:
    dt = datetime.datetime.utcfromtimestamp(ts_epoch)
    return (UTC_NOW - dt).total_seconds()/60.0

def collect_news_for_symbols(symbols):
    items = []
    for sym in symbols:
        try:
            news = getattr(yf.Ticker(sym), "news", []) or []
            for it in news:
                title = it.get("title") or ""
                pub = it.get("publisher") or ""
                link = it.get("link") or ""
                ts = it.get("providerPublishTime") or it.get("published") or 0
                age = _news_age_minutes(float(ts)) if ts else 1e9
                if age <= TWD_BREAKING_RECENCY_MIN:
                    items.append({"symbol": sym, "title": title, "publisher": pub, "link": link, "age": age, "ts": float(ts)})
        except Exception as e:
            _dbg(f"news error {sym}: {e}")
    return items

def collect_news_candidates():
    # Pass 1: user/watchlist
    items = collect_news_for_symbols(WATCHLIST)
    if items:
        return items
    # Pass 2: broaden to macro + mega-caps (mimics “general headlines” on Yahoo)
    _dbg("watchlist empty → trying FALLBACK_SYMBOLS for general headlines")
    return collect_news_for_symbols(FALLBACK_SYMBOLS)

def cluster_and_filter(items):
    clusters = {}
    for it in items:
        key = _norm_headline(it["title"])
        if not key: 
            continue
        clusters.setdefault(key, []).append(it)
    kept = []
    for key, arr in clusters.items():
        pubs = set([a["publisher"] for a in arr if a.get("publisher")])
        if len(pubs) >= TWD_BREAKING_MIN_SOURCES:
            arr_sorted = sorted(arr, key=lambda x: (x["publisher"] in PREFERRED_SOURCES, -x["ts"]), reverse=True)
            kept.append((key, arr_sorted[0], arr))
    kept.sort(key=lambda x: -x[1]["ts"])
    return kept

# ------------------ Poster state ------------------
def load_seen():
    os.makedirs(STATE_DIR, exist_ok=True)
    if os.path.exists(SEEN_FILE):
        try:
            with open(SEEN_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"hashes": [], "last_post_ts": 0}
    return {"hashes": [], "last_post_ts": 0}

def save_seen(state):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(SEEN_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def headline_hash(key: str) -> str:
    return hashlib.md5(key.encode("utf-8")).hexdigest()

# ------------------ Poster background ------------------
def _pick_story_background_path(symbol: str, title: str):
    base = "assets/backgrounds"
    sym_dir = os.path.join(base, symbol.upper())
    candidates = sorted(glob.glob(os.path.join(sym_dir, "*.*")))
    if candidates: return candidates[0]
    title_l = (title or "").lower()
    themes = []
    if any(k in title_l for k in ["gold","bullion"]): themes.append("GOLD")
    if any(k in title_l for k in ["oil","opec","brent","wti"]): themes.append("OIL")
    if any(k in title_l for k in ["chip","chips","semiconductor","gpu","ai"]): themes.append("CHIPS")
    if any(k in title_l for k in ["auto","ev","tesla"]): themes.append("AUTO")
    if any(k in title_l for k in ["health","drug","trial","fda"]): themes.append("HEALTH")
    if not themes: themes = ["GENERIC"]
    for th in themes:
        th_dir = os.path.join(base, th)
        cand = sorted(glob.glob(os.path.join(th_dir, "*.*")))
        if cand: return cand[0]
    return None

# ------------------ Poster render ------------------
def render_news_poster(out_path, symbol, title, body, publisher=None):
    W, H = 1080, 1080
    canvas = Image.new("RGBA", (W, H), (255,255,255,255))
    draw = ImageDraw.Draw(canvas)

    bg_path = _pick_story_background_path(symbol, title)
    if bg_path and os.path.exists(bg_path):
        try:
            bg = Image.open(bg_path).convert("RGBA")
            scale = max(W/bg.width, H/bg.height)
            bg = bg.resize((int(bg.width*scale), int(bg.height*scale)))
            bx = (bg.width - W)//2; by = (bg.height - H)//2
            bg = bg.crop((bx, by, bx+W, by+H))
            alpha = Image.new("L", (W,H), int(255*0.15))
            bg.putalpha(alpha)
            canvas = Image.alpha_composite(canvas, bg); draw = ImageDraw.Draw(canvas)
        except Exception as e:
            _dbg(f"bg load fail: {e}")

    f_news = _load_font(28, bold=True)
    f_head = _load_font(64, bold=True)
    f_body = _load_font(34, bold=False)
    f_meta = _load_font(24, bold=False)

    pad = 60; x1,y1,x2,y2 = pad,pad,W-pad,H-pad
    draw.text((x1, y1), "NEWS", fill=(30,30,30,255), font=f_news)

    def wrap_text(text, font, max_width):
        words = (text or "").split()
        lines, cur = [], ""
        for w in words:
            test = f"{cur} {w}".strip()
            if draw.textlength(test, font=font) <= max_width: cur = test
            else:
                if cur: lines.append(cur)
                cur = w
        if cur: lines.append(cur)
        return lines

    head = (title or "MARKET UPDATE").upper()
    head_lines = wrap_text(head, f_head, x2-x1)[:3]
    head_y = y1 + 70
    for i, line in enumerate(head_lines):
        draw.text((x1, head_y + i*72), line, fill=(10,10,10,255), font=f_head)
    body_top = head_y + len(head_lines)*72 + 24

    body_lines = wrap_text((body or "").strip(), f_body, x2-x1)[:8]
    for i, line in enumerate(body_lines):
        draw.text((x1, body_top + i*46), line, fill=(35,35,35,255), font=f_body)

    if publisher:
        draw.text((x1, y2 - 30), f"Source: {publisher}", fill=(90,90,90,255), font=f_meta)

    tlogo_path = os.path.join("assets","logos",f"{symbol.upper()}.png")
    if os.path.exists(tlogo_path):
        try:
            tlogo = Image.open(tlogo_path).convert("RGBA")
            hmax = 80; scale = min(1.0, hmax / max(1, tlogo.height))
            tlogo = tlogo.resize((int(tlogo.width*scale), int(tlogo.height*scale)))
            lx = x2 - tlogo.width; ly = y1
            canvas.alpha_composite(tlogo, (lx, ly))
        except Exception: pass

    if BRAND_LOGO_PATH and os.path.exists(BRAND_LOGO_PATH):
        try:
            blogo = Image.open(BRAND_LOGO_PATH).convert("RGBA")
            maxh = 130; scale = min(1.0, maxh / max(1, blogo.height))
            blogo = blogo.resize((int(blogo.width*scale), int(blogo.height*scale)))
            bx = x2 - blogo.width; by = y2 - blogo.height
            canvas.alpha_composite(blogo, (bx, by))
        except Exception: pass

    out = canvas.convert("RGB")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.save(out_path, quality=92)

# ------------------ Posters driver ------------------
def run_posters_once():
    if not TWD_BREAKING_ON:
        print("[info] Breaking posters OFF by env")
        return 0

    os.makedirs(POSTERS_DIR, exist_ok=True); os.makedirs(STATE_DIR, exist_ok=True)
    seen = load_seen()
    last_post_ts = seen.get("last_post_ts", 0)
    mins_since_last = (UTC_NOW - datetime.datetime.utcfromtimestamp(last_post_ts)).total_seconds()/60.0 if last_post_ts else 1e9
    _dbg(f"mins since last poster: {mins_since_last:.1f}")

    items = collect_news_candidates()
    print(f"[info] news items within window: {len(items)}")
    if TWD_DEBUG:
        for it in items[:10]:
            print(f"[debug] item: {it.get('publisher')} | {it.get('symbol')} | {it.get('title')[:90]}... age={it.get('age'):.1f}m")

    # Cluster by headline and require distinct sources
    def cluster_and_emit(items_local):
        clusters = cluster_and_filter(items_local)
        print(f"[info] clusters (>= {TWD_BREAKING_MIN_SOURCES} sources): {len(clusters)}")
        for key, best, arr in clusters:
            h = headline_hash(key)
            pubs = sorted(list(set([a.get("publisher","") for a in arr if a.get("publisher")])))
            print(f"[info] candidate cluster: pubs={len(pubs)} key='{key}' best_pub={best.get('publisher')}")
            if h in set(seen.get("hashes", [])):
                print("[info]  -> skipped (already seen)")
                continue
            if mins_since_last < TWD_BREAKING_MIN_INTERVAL_MIN:
                print(f"[info] Cooldown active ({mins_since_last:.1f}m < {TWD_BREAKING_MIN_INTERVAL_MIN}m). No poster.")
                return 0
            symbol = best.get("symbol") or "MARKET"
            title = best.get("title") or "MARKET UPDATE"
            pub = best.get("publisher") or "Multiple"
            paragraph = f"{title}. Coverage from {', '.join(pubs[:5])}" + (" and others." if len(pubs) > 5 else ".")
            ts_str = datetime.datetime.utcfromtimestamp(best["ts"]).strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(POSTERS_DIR, f"news_{ts_str}.png")
            try:
                render_news_poster(out_path, symbol, title, paragraph, publisher=pub)
                print("poster:", out_path)
            except Exception as e:
                print(f"[warn] poster render failed: {e}"); traceback.print_exc(); return 0
            seen.setdefault("hashes", []).append(h); seen["last_post_ts"] = best["ts"]; save_seen(seen); return 1
        return 0

    made = cluster_and_emit(items)
    if made: return made

    # Fallback: if no clusters, allow newest single item (from watchlist or fallback symbols)
    if TWD_BREAKING_FALLBACK:
        if not items:
            # widen to 24h to ensure at least one story
            global TWD_BREAKING_RECENCY_MIN
            TWD_BREAKING_RECENCY_MIN = max(TWD_BREAKING_RECENCY_MIN, 24*60)
            print("[info] No items in window; widened to 24h.")
            items = collect_news_candidates()
            print(f"[info] widened window items: {len(items)}")

        if items:
            best = sorted(items, key=lambda x: -x["ts"])[0]
            symbol = best.get("symbol") or "MARKET"
            title = best.get("title") or "MARKET UPDATE"
            pub = best.get("publisher") or "Source"
            paragraph = f"{title}. Reported by {pub}."
            key = _norm_headline(title); h = headline_hash(key)
            if h in set(seen.get("hashes", [])):
                print("[info] fallback skipped (already seen)"); return 0
            ts_str = datetime.datetime.utcfromtimestamp(best["ts"]).strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(POSTERS_DIR, f"news_{ts_str}.png")
            try:
                render_news_poster(out_path, symbol, title, paragraph, publisher=pub)
                print("poster (fallback):", out_path)
            except Exception as e:
                print(f"[warn] fallback poster render failed: {e}"); traceback.print_exc(); return 0
            seen.setdefault("hashes", []).append(h); seen["last_post_ts"] = best["ts"]; save_seen(seen); return 1

    print("[info] No cross-confirmed clusters and fallback disabled or no items.")
    return 0

# ------------------ Charts driver ------------------
def run_charts_once():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tickers = choose_tickers_somehow(n=6, seed=DATESTR)
    print("[info] selected tickers:", tickers)
    saved = 0; captions = []
    for t in tickers:
        try:
            payload = fetch_one_chart(t)
            if not payload:
                print(f"[warn] no data for {t}, skipping"); continue
            out_path = os.path.join(OUTPUT_DIR, f"twd_{t}_{DATESTR}.png")
            try:
                render_single_post(out_path, t, payload)
            except Exception as re:
                _dbg(f"render error for {t}: {re}")
                img = Image.new("RGB", (1080,1080), "white"); d = ImageDraw.Draw(img)
                d.text((40,40), f"Render error: {t}\n{re}", fill="black")
                os.makedirs(os.path.dirname(out_path), exist_ok=True); img.save(out_path, quality=85)
            saved += 1
            (df,last,chg30,sup_low,sup_high,tf_tag,chg1d) = payload
            captions.append(caption_line(t, last, chg30, chg1d, sup_low, sup_high, seed=DATESTR))
        except Exception as e:
            print(f"Error: failed for {t}: {e}"); traceback.print_exc()
    print(f"[info] saved images: {saved}")
    if saved > 0:
        caption_path = os.path.join(OUTPUT_DIR, f"caption_{DATESTR}.txt")
        now_str = TODAY.strftime("%d %b %Y")
        header = f"Ones to Watch – {now_str}\n\n"
        footer = f"\n\n{random.choice(CTA_POOL)}\n\nIdeas only — not financial advice"
        with open(caption_path, "w", encoding="utf-8") as f:
            f.write(header); f.write("\n\n".join(captions)); f.write(footer)
        print("[info] wrote caption:", caption_path)
    return saved

# ------------------ Main ------------------
def main():
    charts = posters = 0
    if TWD_MODE in ("charts","all"):
        charts = run_charts_once()
    if TWD_MODE in ("posters","all"):
        posters = run_posters_once()
    print(f"[info] charts_generated={charts}, posters_generated={posters}")

if __name__ == "__main__":
    main()
