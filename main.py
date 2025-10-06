import os, io, math, random, datetime, pytz, traceback
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import yfinance as yf

# -------- CONFIG --------
BRAND_NAME = "TrendWatchDesk"
TIMEZONE   = "Europe/London"
CANVAS_W, CANVAS_H = 1080, 1080
MARGIN = 40

# Visuals
BG        = (255,255,255)
TEXT_MAIN = (20,20,22)
TEXT_MUT  = (145,150,160)
GRID      = (230,233,237)
UP_COL    = (20,170,90)
DOWN_COL  = (230,70,70)

SUPPORT_FILL = (40,120,255,44)
SUPPORT_EDGE = (40,120,255,120)
RESIST_FILL  = (230,70,70,40)
RESIST_EDGE  = (230,70,70,120)

CHART_LOOKBACK   = 120
SUMMARY_LOOKBACK = 30
YAHOO_PERIOD     = "2y"
STOOQ_MAX_DAYS   = 400

# -------- CATEGORY POOLS --------
POOLS = {
    "AI": ["NVDA","MSFT","GOOG","META","AMD","AVGO","CRM","SNOW","PLTR","NOW"],
    "QUANTUM": ["IONQ","IBM","RGTI","AMZN","MSFT"],
    "MAG7": ["AAPL","MSFT","GOOG","AMZN","META","NVDA","TSLA"],
    "HEALTHCARE": ["UNH","LLY","JNJ","ABBV","MRK"],
    "FINTECH": ["V","MA","PYPL","SQ","SOFI"],
    "SEMIS": ["TSM","ASML","QCOM","INTC","AMD","MU","TXN"]
}
# Quotas + 2 wildcards -> total 8 per run
QUOTAS = [("AI", 2), ("MAG7", 1), ("HEALTHCARE", 1), ("FINTECH", 1), ("SEMIS", 1)]
WILDCARDS = 2
N_TICKERS = sum(q for _, q in QUOTAS) + WILDCARDS

OUTPUT_DIR = "output"
LOGO_DIR   = "assets/logos"
BRAND_DIR  = "assets"

# -------- HTTP session (with retry) --------
def make_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    retry = Retry(total=5, backoff_factor=0.7,
                  status_forcelist=[429,500,502,503,504],
                  allowed_methods=["GET","POST"])
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s
SESS = make_session()

# -------- Fonts --------
def load_font(size=42, bold=False):
    pref = "fonts/Roboto-Bold.ttf" if bold else "fonts/Roboto-Regular.ttf"
    if os.path.exists(pref):
        try: return ImageFont.truetype(pref, size)
        except: pass
    fam = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try: return ImageFont.truetype(fam, size)
    except: return ImageFont.load_default()
F_TITLE = load_font(70,  bold=True)
F_PRICE = load_font(46,  bold=True)
F_CHG   = load_font(34,  bold=True)
F_SUB   = load_font(28,  bold=False)
F_META  = load_font(20,  bold=False)  # tiny caption font

# -------- Helpers --------
def y_map(v, vmin, vmax, y0, y1):
    if vmax - vmin < 1e-6: return (y0 + y1)//2
    return int(y1 - (v - vmin) * (y1 - y0) / (vmax - vmin))

def case_insensitive_find(directory, base_no_ext):
    if not os.path.isdir(directory): return None
    for fn in os.listdir(directory):
        stem, ext = os.path.splitext(fn)
        if stem.upper() == base_no_ext.upper() and ext.lower() in (".png",".jpg",".jpeg",".webp"):
            return os.path.join(directory, fn)
    return None

def find_ticker_logo_path(ticker):
    exact = os.path.join(LOGO_DIR, f"{ticker}.png")
    if os.path.exists(exact): return exact
    return case_insensitive_find(LOGO_DIR, ticker)

def find_brand_logo_path():
    candidates = [os.path.join(BRAND_DIR, "brand_logo.png"), os.path.join(BRAND_DIR, "logo.png")]
    for p in candidates:
        if os.path.exists(p): return p
    if os.path.isdir(BRAND_DIR):
        for fn in os.listdir(BRAND_DIR):
            low = fn.lower()
            if ("logo" in low) and os.path.splitext(low)[1] in (".png",".jpg",".jpeg"):
                return os.path.join(BRAND_DIR, fn)
    return None

def sample_with_quotas_and_wildcards(quotas, wildcards, pools, seed):
    rnd = random.Random(seed)
    chosen = []
    def pick_from(category, k):
        src = list(pools.get(category, []))
        rnd.shuffle(src)
        for t in src:
            if len(chosen) >= sum(q for _, q in quotas) and k <= 0: break
            if t not in chosen:
                chosen.append(t); k -= 1
                if k == 0: break
    for cat, q in quotas:
        pick_from(cat, q)
    universe = []
    for arr in pools.values():
        for t in arr:
            if t not in universe: universe.append(t)
    rnd.shuffle(universe)
    for t in universe:
        if len(chosen) >= sum(q for _, q in quotas) + wildcards: break
        if t not in chosen: chosen.append(t)
    return chosen

# -------- Indicators & S/R --------
def atr(df, n=14):
    high = df["High"]; low = df["Low"]; close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def sma(series, n):
    return series.rolling(n).mean()

def swing_points(df, w=2):
    highs, lows = [], []
    H, L = df["High"], df["Low"]
    for i in range(w, len(df)-w):
        if H.iloc[i] == H.iloc[i-w:i+w+1].max():
            highs.append((i, df.index[i], float(H.iloc[i])))
        if L.iloc[i] == L.iloc[i-w:i+w+1].min():
            lows.append((i, df.index[i], float(L.iloc[i])))
    return lows, highs

def recent_major_swings(df, lookback=120):
    lows, highs = swing_points(df.tail(lookback), w=2)
    if not lows or not highs: return None, None
    low_level  = min([p[2] for p in lows])
    high_level = max([p[2] for p in highs])
    return low_level, high_level

def fibonacci_levels(low, high):
    if low is None or high is None or high <= low: return []
    span = high - low
    return [low + 0.382*span, low + 0.500*span, low + 0.618*span]

def floor_pivots(df):
    last = df.iloc[-2] if len(df) >= 2 else df.iloc[-1]
    H, L, C = float(last["High"]), float(last["Low"]), float(last["Close"])
    P = (H + L + C) / 3.0
    R1 = 2*P - L; S1 = 2*P - H
    R2 = P + (H - L); S2 = P - (H - L)
    return [P, S1, R1, S2, R2]

def volume_percentile(series, p=0.6):
    return series.quantile(p)

def cluster_levels(levels, tol):
    if not levels: return []
    levels = sorted(levels)
    clusters = []
    cluster = [levels[0]]
    for x in levels[1:]:
        if abs(x - cluster[-1]) <= tol:
            cluster.append(x)
        else:
            clusters.append((sum(cluster)/len(cluster), len(cluster)))
            cluster = [x]
    clusters.append((sum(cluster)/len(cluster), len(cluster)))
    return clusters

def confluence_support_resistance(df):
    df = df.copy()
    vol = df["Volume"].fillna(0)
    vol_bar = volume_percentile(vol.tail(120), 0.6)

    df["SMA50"]  = sma(df["Close"], 50)
    df["SMA200"] = sma(df["Close"], 200)
    low_major, high_major = recent_major_swings(df, lookback=120)
    fibs = fibonacci_levels(low_major, high_major)
    pivs = floor_pivots(df)

    lows, highs = swing_points(df.tail(180), w=2)
    swing_low_lvls  = [p[2] for p in lows]
    swing_high_lvls = [p[2] for p in highs]

    cand = swing_low_lvls + swing_high_lvls
    for col in ["SMA50","SMA200"]:
        if not df[col].dropna().empty:
            cand.append(float(df[col].iloc[-1]))
    cand += fibs + pivs
    cand = [float(x) for x in cand if x and not math.isnan(x) and math.isfinite(x)]
    if not cand:
        return None, None, None, None, None, None, (lows, highs)

    last = float(df["Close"].iloc[-1])
    atr_series = atr(df)
    atr_val = float(atr_series.iloc[-1]) if not atr_series.dropna().empty else max(last*0.005, 0.5)
    tol_abs = max(last * 0.003, 0.2)

    clusters = cluster_levels(cand, tol_abs)

    scores = []
    H, L = df["High"], df["Low"]
    for level, hits in clusters:
        touch_mask = (L <= level) & (H >= level)
        touches = int(touch_mask.tail(180).sum())
        vol_conf = int(((touch_mask) & (vol >= vol_bar)).tail(180).sum() > 0)
        prox = max(0.0, 1.0 - abs(level - last) / max(atr_val*5, tol_abs*3))
        score = hits*1.0 + touches*0.5 + vol_conf*0.5 + prox*0.5
        scores.append((level, score))

    supports    = [(lv, sc) for lv, sc in scores if lv <= last]
    resistances = [(lv, sc) for lv, sc in scores if lv >= last]

    sup_low = sup_high = res_low = res_high = None
    sup_label = res_label = None

    if supports:
        supports.sort(key=lambda x: (-x[1], last - x[0]))
        s_level = supports[0][0]
        half = max(atr_val*0.25, last*0.002, 0.2)
        sup_low, sup_high = s_level - half, s_level + half
        sup_label = f"Support ~{s_level:.2f}"

    if resistances:
        resistances.sort(key=lambda x: (-x[1], x[0] - last))
        r_level = resistances[0][0]
        half = max(atr_val*0.25, last*0.002, 0.2)
        res_low, res_high = r_level - half, r_level + half
        res_label = f"Resistance ~{r_level:.2f}"

    return (sup_low, sup_high, res_low, res_high, sup_label, res_label, (lows, highs))

def detect_bos(df, swings, buffer_pct=0.0005, vol_threshold_pct=0.4):
    """
    BoS with relaxed thresholds:
      - close > last swing high*(1+buffer) AND vol ≥ percentile -> BoS↑
      - close < last swing low*(1-buffer)  AND vol ≥ percentile -> BoS↓
    Returns: ("up"/"down"/None, swing_price, swing_index)
    """
    lows, highs = swings
    if df is None or len(df) < 5:
        return (None, None, None)

    close_last = float(df["Close"].iloc[-1])
    vol_last   = float(df["Volume"].iloc[-1])
    vol_bar = volume_percentile(df["Volume"].tail(120), vol_threshold_pct)

    if highs:
        i_hi, _, hi_price = highs[-1]
        if (close_last > hi_price * (1 + buffer_pct)) and (vol_last >= vol_bar):
            return ("up", hi_price, i_hi)

    if lows:
        i_lo, _, lo_price = lows[-1]
        if (close_last < lo_price * (1 - buffer_pct)) and (vol_last >= vol_bar):
            return ("down", lo_price, i_lo)

    return (None, None, None)

# -------- Data --------
def to_stooq_symbol(t): return f"{t.lower()}.us"

def fetch_stooq_daily(t):
    try:
        url = f"https://stooq.com/q/d/l/?s={to_stooq_symbol(t)}&i=d"
        r = SESS.get(url, timeout=10); r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if df is None or df.empty: return None
        df = df.rename(columns=str.title)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna().set_index("Date").sort_index()
        return df.iloc[-STOOQ_MAX_DAYS:]
    except Exception:
        return None

def fetch_yahoo_daily(t):
    try:
        df = yf.download(tickers=t, period=YAHOO_PERIOD, interval="1d",
                         auto_adjust=False, progress=False, session=SESS)
        if df is not None and not df.empty: return df
    except Exception:
        return None
    return None

def clean_and_summarize(df):
    if df is None or df.empty: return None
    df = df[["Open","High","Low","Close","Volume"]].dropna()
    dfc = df.iloc[-CHART_LOOKBACK:].copy()
    if dfc.empty: return None
    dfs = df.iloc[-SUMMARY_LOOKBACK:].copy()
    last = float(dfc["Close"].iloc[-1])
    chg30 = (last/float(dfs["Close"].iloc[0]) - 1.0)*100 if len(dfs)>1 else 0.0

    sup_low, sup_high, res_low, res_high, sup_label, res_label, swings = confluence_support_resistance(dfc)
    bos_dir, bos_level, bos_idx = detect_bos(dfc, swings, buffer_pct=0.0005, vol_threshold_pct=0.4)

    # Fallback: show latest swing level if no BoS
    bos_fallback = False
    if bos_dir is None:
        lows, highs = swings
        if highs:
            bos_dir, bos_level, bos_idx = "up", highs[-1][2], highs[-1][0]
            bos_fallback = True
        elif lows:
            bos_dir, bos_level, bos_idx = "down", lows[-1][2], lows[-1][0]
            bos_fallback = True

    return (dfc,last,chg30,sup_low,sup_high,res_low,res_high,
            sup_label,res_label,bos_dir,bos_level,bos_idx,bos_fallback)

def fetch_one(t):
    df = fetch_stooq_daily(t)
    if df is None: df = fetch_yahoo_daily(t)
    return clean_and_summarize(df)

# -------- Drawing --------
def render_single_post(path, ticker, payload, brand_logo_path):
    (df,last,chg30,sup_low,sup_high,res_low,res_high,
     sup_label,res_label,bos_dir,bos_level,bos_idx,bos_fallback) = payload

    img = Image.new("RGBA", (CANVAS_W, CANVAS_H), BG + (255,))
    d = ImageDraw.Draw(img)

    # Header
    d.text((MARGIN, MARGIN), ticker, fill=TEXT_MAIN, font=F_TITLE)
    d.text((MARGIN, MARGIN+72), f"{last:,.2f} USD", fill=TEXT_MAIN, font=F_PRICE)
    chg_col = UP_COL if chg30>=0 else DOWN_COL
    d.text((MARGIN, MARGIN+122), f"{chg30:+.2f}% past {SUMMARY_LOOKBACK}d", fill=chg_col, font=F_CHG)
    d.text((MARGIN, MARGIN+122+38), "Daily chart • confluence S/R zones", fill=TEXT_MUT, font=F_SUB)

    # Ticker logo top-right
    t_logo = find_ticker_logo_path(ticker)
    if t_logo:
        try:
            logo = Image.open(t_logo).convert("RGBA")
            logo.thumbnail((140,140))
            img.alpha_composite(logo, (CANVAS_W - logo.width - MARGIN, MARGIN))
        except: pass

    # Chart area
    top = 260; bot = CANVAS_H - 100; left = MARGIN; right = CANVAS_W - MARGIN
    d.rectangle((left, top, right, bot), fill=(255,255,255))
    for gy in (0.25, 0.5, 0.75):
        y = int(top + gy*(bot-top)); d.line([(left+4,y),(right-4,y)], fill=GRID, width=1)

    vmin = float(df["Low"].min()); vmax = float(df["High"].max())
    n = len(df); step = (right-left)/max(1,n)
    wick = max(1, int(step*0.12)); body = max(3, int(step*0.4))

    # Candles
    for i,row in enumerate(df.itertuples(index=False)):
        o,h,l,c = float(row.Open), float(row.High), float(row.Low), float(row.Close)
        cx = int(left + i*step + step*0.5)
        yH = y_map(h, vmin, vmax, top, bot); yL = y_map(l, vmin, vmax, top, bot)
        yO = y_map(o, vmin, vmax, top, bot); yC = y_map(c, vmin, vmax, top, bot)
        col = UP_COL if c>=o else DOWN_COL
        d.line([(cx,yH),(cx,yL)], fill=col, width=wick)
        t, b = min(yO,yC), max(yO,yC)
        if b-t < 2: b = t+2
        d.rectangle((cx-body//2, t, cx+body//2, b), fill=col)

    # Zones overlay
    overlay = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0,0,0,0))
    od = ImageDraw.Draw(overlay)
    if sup_low and sup_high:
        y1 = y_map(sup_low, vmin, vmax, top, bot)
        y2 = y_map(sup_high, vmin, vmax, top, bot)
        od.rectangle((left, min(y1,y2), right, max(y1,y2)),
                     fill=SUPPORT_FILL, outline=SUPPORT_EDGE, width=1)
    if res_low and res_high:
        y1 = y_map(res_low, vmin, vmax, top, bot)
        y2 = y_map(res_high, vmin, vmax, top, bot)
        od.rectangle((left, min(y1,y2), right, max(y1,y2)),
                     fill=RESIST_FILL, outline=RESIST_EDGE, width=1)
    img.alpha_composite(overlay)

    # --- BoS / Swing marker: strong highlight + dotted line + arrow + label ---
    bos_caption = None
    if bos_level is not None and bos_idx is not None:
        try:
            cx_center = int(left + bos_idx*step + step*0.5)
            x0 = int(left + bos_idx*step)
            x1 = int(x0 + step)
            yL = y_map(bos_level, vmin, vmax, top, bot)

            is_bos = (not bos_fallback)
            bos_col = UP_COL if bos_dir == "up" else DOWN_COL
            bos_fill = (bos_col[0], bos_col[1], bos_col[2], 100 if is_bos else 70)
            bos_line = (bos_col[0], bos_col[1], bos_col[2], 255 if is_bos else 200)

            bos_overlay = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0,0,0,0))
            bod = ImageDraw.Draw(bos_overlay)

            # 1) highlight column (always, but slightly softer if fallback)
            bod.rectangle((max(left, x0-1), top, min(right, x1+1), bot), fill=bos_fill)

            # 2) dotted horizontal line
            dot_len, gap = 12, 4
            xx = left
            while xx < right:
                x_end = min(xx + dot_len, right)
                bod.line([(xx, yL), (x_end, yL)], fill=bos_line, width=3)
                xx += dot_len + gap

            # 3) arrow marker
            tri = 12
            if bos_dir == "up":
                bod.polygon([(cx_center, yL-16-tri), (cx_center-tri, yL-16), (cx_center+tri, yL-16)], fill=bos_line)
            else:
                bod.polygon([(cx_center, yL+16+tri), (cx_center-tri, yL+16), (cx_center+tri, yL+16)], fill=bos_line)

            # 4) label at right edge
            lbl = "BoS↑" if (is_bos and bos_dir == "up") else ("BoS↓" if (is_bos and bos_dir == "down") else "Swing ref")
            ly = yL - 22 if bos_dir == "up" else yL + 6
            lx = right - 110
            pad = 6
            tw, th = d.textbbox((0,0), lbl, font=F_META)[2:]
            bod.rectangle((lx - pad, ly - pad, lx + tw + pad, ly + th + pad),
                          fill=(245,245,245,220))
            img.alpha_composite(bos_overlay)
            d.text((lx, ly), lbl, fill=bos_col, font=F_META)

            bos_caption = ("BoS↑ (close+vol)" if bos_dir == "up" else "BoS↓ (close+vol)") if is_bos else "Swing reference (no break)"
        except Exception:
            bos_caption = None

    # Tiny captions just above footer
    caption_y = CANVAS_H - 68
    if sup_label:
        d.text((MARGIN, caption_y), sup_label, fill=TEXT_MUT, font=F_META); caption_y -= 18
    if res_label:
        d.text((MARGIN, caption_y), res_label, fill=TEXT_MUT, font=F_META); caption_y -= 18
    if bos_caption:
        d.text((MARGIN, caption_y), bos_caption, fill=TEXT_MUT, font=F_META)

    # Footer
    d.text((MARGIN, CANVAS_H-40), "Not financial advice", fill=TEXT_MUT, font=F_META)

    # Brand logo (max 120px width)
    brand_logo_path = brand_logo_path or find_brand_logo_path()
    if brand_logo_path and os.path.exists(brand_logo_path):
        try:
            brand = Image.open(brand_logo_path).convert("RGBA")
            w,h = brand.size
            if w > 120:
                scale = 120 / float(w)
                brand = brand.resize((120, int(h*scale)), Image.LANCZOS)
            bx = CANVAS_W - brand.width - MARGIN
            by = CANVAS_H - brand.height - MARGIN
            img.alpha_composite(brand, (bx, by))
        except Exception as e:
            print(f"[warn] brand logo draw failed: {brand_logo_path} ({e})")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = Image.new("RGB", img.size, (255,255,255))
    out.paste(img, mask=img.split()[-1])
    out.save(path, "PNG", optimize=True)

# -------- Main --------
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    now = datetime.datetime.now(pytz.timezone(TIMEZONE))
    datestr = now.strftime("%Y%m%d")

    tickers = sample_with_quotas_and_wildcards(QUOTAS, WILDCARDS, POOLS, seed=datestr)
    print("[info] selected tickers:", tickers)

    BRAND_LOGO_PATH = find_brand_logo_path()
    print("[info] resolved brand logo path:", BRAND_LOGO_PATH)

    for t in tickers:
        try:
            payload = fetch_one(t)
            out_path = os.path.join(OUTPUT_DIR, f"twd_{t}_{datestr}.png")
            if not payload:
                print(f"[warn] no data for {t}, skipping")
                continue
            render_single_post(out_path, t, payload, BRAND_LOGO_PATH)
            print("done:", out_path)
        except Exception as e:
            print(f"[error] failed for {t}: {e}")
            traceback.print_exc()
