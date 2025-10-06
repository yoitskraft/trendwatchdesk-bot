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
# Quotas + wildcards -> total 8 per run
QUOTAS = [("AI", 2), ("MAG7", 1), ("HEALTHCARE", 1), ("FINTECH", 1), ("SEMIS", 1)]
WILDCARDS = 2
N_TICKERS = sum(q for _, q in QUOTAS) + WILDCARDS  # 8

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
F_META  = load_font(22,  bold=False)

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
    # quotas first
    for cat, q in quotas:
        pick_from(cat, q)
    # wildcards
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
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def sma(series, n):
    return series.rolling(n).mean()

def swing_points(df, w=2):
    """Find swing highs/lows using simple fractals with window w."""
    highs, lows = [], []
    H, L = df["High"], df["Low"]
    for i in range(w, len(df)-w):
        if H.iloc[i] == H.iloc[i-w:i+w+1].max():
            highs.append((df.index[i], float(H.iloc[i])))
        if L.iloc[i] == L.iloc[i-w:i+w+1].min():
            lows.append((df.index[i], float(L.iloc[i])))
    return lows, highs

def recent_major_swings(df, lookback=90):
    """Pick most recent prominent swing high/low over lookback using extremes of swing points."""
    lows, highs = swing_points(df.tail(lookback), w=2)
    if not lows or not highs: return None, None
    # Use most recent lowest low and highest high
    low_level  = min([p[1] for p in lows])
    high_level = max([p[1] for p in highs])
    return low_level, high_level

def fibonacci_levels(low, high):
    if low is None or high is None or high <= low: return []
    span = high - low
    levels = [
        low + 0.382*span,
        low + 0.500*span,
        low + 0.618*span,
    ]
    return levels

def floor_pivots(df):
    """Classic floor pivots from last completed session."""
    last = df.iloc[-2] if len(df) >= 2 else df.iloc[-1]
    H, L, C = float(last["High"]), float(last["Low"]), float(last["Close"])
    P = (H + L + C) / 3.0
    R1 = 2*P - L; S1 = 2*P - H
    R2 = P + (H - L); S2 = P - (H - L)
    return [P, S1, R1, S2, R2]

def volume_percentile(series, p=0.6):
    return series.quantile(p)

def cluster_levels(levels, tol):
    """Group nearby levels within tolerance tol; return cluster centers and hit counts."""
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
    """
    Build candidate levels from:
      - swing point prices (last ~CHART_LOOKBACK)
      - SMA50, SMA200
      - Fibonacci (from recent major swing low/high)
      - Floor pivots (P, S1, R1, S2, R2)
    Confirm with volume: bars touching level with volume > 60th percentile.
    Convert top support/resistance into zones using ATR.
    """
    df = df.copy()
    vol = df["Volume"].fillna(0)
    vol_bar = volume_percentile(vol.tail(120), 0.6)

    # indicators
    df["SMA50"]  = sma(df["Close"], 50)
    df["SMA200"] = sma(df["Close"], 200)
    low_major, high_major = recent_major_swings(df, lookback=120)
    fibs = fibonacci_levels(low_major, high_major)
    pivs = floor_pivots(df)

    # swing points
    lows, highs = swing_points(df.tail(180), w=2)
    swing_low_lvls  = [p[1] for p in lows]
    swing_high_lvls = [p[1] for p in highs]

    # candidate levels pool
    cand = []
    cand += swing_low_lvls + swing_high_lvls
    for col in ["SMA50","SMA200"]:
        if not df[col].dropna().empty:
            cand.append(float(df[col].iloc[-1]))
    cand += fibs
    cand += pivs

    # Clean
    cand = [float(x) for x in cand if x and not math.isnan(x) and math.isfinite(x)]
    if not cand:
        return None, None

    last = float(df["Close"].iloc[-1])
    # Tolerance for clustering: 0.3% of price (or wider for low-priced stocks)
    tol_abs = max(last * 0.003, 0.2)

    # cluster nearby levels, count confluence
    clusters = cluster_levels(cand, tol_abs)

    # score clusters with touches and volume confirmation
    scores = []
    H, L, C = df["High"], df["Low"], df["Close"]
    atr_val = float(atr(df).iloc[-1]) if not atr(df).dropna().empty else max(last*0.005, 0.5)

    for level, hits in clusters:
        # touches: bars where level lies within wick
        touch_mask = (L <= level) & (H >= level)
        touches = int(touch_mask.tail(180).sum())
        # volume confirmation: any of those touches with volume > 60th pct
        vol_conf = int(((touch_mask) & (vol >= vol_bar)).tail(180).sum() > 0)
        # proximity component (closer to current price = slightly higher score)
        prox = max(0.0, 1.0 - abs(level - last) / max(atr_val*5, tol_abs*3))
        score = hits*1.0 + touches*0.5 + vol_conf*0.5 + prox*0.5
        scores.append((level, score))

    # Split into support (<= last) and resistance (>= last)
    supports   = [(lv, sc) for lv, sc in scores if lv <= last]
    resistances= [(lv, sc) for lv, sc in scores if lv >= last]

    if supports:
        supports.sort(key=lambda x: (-x[1], last - x[0]))  # higher score, then closest below
        s_level = supports[0][0]
        # zone thickness from ATR (quarter ATR) with bounds
        half = max(atr_val*0.25, last*0.002, 0.2)
        sup_low, sup_high = s_level - half, s_level + half
    else:
        sup_low = sup_high = None

    if resistances:
        resistances.sort(key=lambda x: (-x[1], x[0] - last))  # higher score, then closest above
        r_level = resistances[0][0]
        half = max(atr_val*0.25, last*0.002, 0.2)
        res_low, res_high = r_level - half, r_level + half
    else:
        res_low = res_high = None

    return (sup_low, sup_high, res_low, res_high)

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

    # ---- NEW: confluence-based zones
    sup_low, sup_high, res_low, res_high = confluence_support_resistance(dfc)

    return (dfc,last,chg30,sup_low,sup_high,res_low,res_high)

def fetch_one(t):
    df = fetch_stooq_daily(t)
    if df is None: df = fetch_yahoo_daily(t)
    return clean_and_summarize(df)

# -------- Drawing --------
def render_single_post(path, ticker, payload, brand_logo_path):
    df,last,chg30,sup_low,sup_high,res_low,res_high = payload
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
