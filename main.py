import os, random, time, datetime, pytz, io
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, Image
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import yfinance as yf

# -------- CONFIG --------
BRAND_NAME = "TrendWatchDesk"
TIMEZONE   = "Europe/London"
CANVAS_W, CANVAS_H = 1080, 1350

# Visuals
BG        = (255,255,255)
CARD_BG   = (250,250,250)
TEXT_MAIN = (20,20,22)
TEXT_MUT  = (92,95,102)
GRID      = (225,228,232)
UP_COL    = (20,170,90)     # support
DOWN_COL  = (230,70,70)     # resistance
ACCENT    = (40,120,255)    # neutral left strip

# Data windows (DAILY)
CHART_LOOKBACK   = 90     # bars shown & used for pivots
SUMMARY_LOOKBACK = 30     # for % text only
YAHOO_PERIOD     = "1y"
STOOQ_MAX_DAYS   = 250

# Weighted random ticker pool
POOL = {
    "NVDA": 5, "MSFT": 4, "TSLA": 3, "AMZN": 5, "META": 4, "GOOG": 4, "AMD": 3,
    "UNH": 2, "AAPL": 5, "NFLX": 2, "BABA": 2, "JPM": 2, "DIS": 2, "BA": 1,
    "ORCL": 2, "NKE": 1, "PYPL": 1, "INTC": 2, "CRM": 2, "KO": 2
}
N_TICKERS = 3

today_seed = datetime.date.today().strftime("%Y%m%d")

def weighted_sample(pool: dict, n: int, seed: str):
    tickers = list(pool.keys()); weights = list(pool.values())
    expanded = [t for t,w in zip(tickers,weights) for _ in range(max(1,int(w)))]
    rnd = random.Random(seed)
    n = min(n, len(set(expanded)))
    picked = []
    while len(picked) < n and expanded:
        t = rnd.choice(expanded)
        if t not in picked: picked.append(t)
    return picked

TICKERS = weighted_sample(POOL, N_TICKERS, seed=today_seed)

OUTPUT_DIR= "output"
DOCS_DIR  = "docs"
LOGO_PATH = "assets/logo.png"
PAGES_URL = "https://<your-username>.github.io/trendwatchdesk-bot/"

# -------- HTTP session --------
def make_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    retry = Retry(
        total=5, connect=5, read=5, backoff_factor=0.6,
        status_forcelist=[429,500,502,503,504],
        allowed_methods=["GET","POST"]
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("http://", adapter); s.mount("https://", adapter)
    return s

SESS = make_session()

# -------- FONTS --------
def load_font(size=42, bold=False):
    pref = "fonts/Roboto-Bold.ttf" if bold else "fonts/Roboto-Regular.ttf"
    if os.path.exists(pref): return ImageFont.truetype(pref, size)
    fam = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try: return ImageFont.truetype(fam, size)
    except: return ImageFont.load_default()

F_TITLE = load_font(65,  bold=True)
F_SUB   = load_font(30,  bold=False)   # smaller subtitle
F_TICK  = load_font(54,  bold=True)
F_NUM   = load_font(46,  bold=True)
F_CHG   = load_font(34,  bold=True)
F_META  = load_font(24,  bold=False)

# -------- Utilities --------
def y_map(v, vmin, vmax, y0, y1):
    if vmax - vmin < 1e-6: return (y0 + y1)//2
    return int(y1 - (v - vmin) * (y1 - y0) / (vmax - vmin))

def clamp(v, a, b): return max(a, min(b, v))

# -------- Pivots & Support/Resistance --------
def _pivot_points(arr, window=3, mode="high"):
    """Unique local extrema pivots."""
    arr = np.asarray(arr, dtype=float); n = len(arr)
    idxs, vals = [], []
    for i in range(window, n-window):
        seg = arr[i-window:i+window+1]; v = arr[i]
        if mode == "high":
            if v == seg.max() and np.count_nonzero(seg==v)==1:
                idxs.append(i); vals.append(v)
        else:
            if v == seg.min() and np.count_nonzero(seg==v)==1:
                idxs.append(i); vals.append(v)
    if not idxs: return np.array([],dtype=int), np.array([],dtype=float)
    return np.array(idxs,dtype=int), np.array(vals,dtype=float)

def pick_key_levels(values, last_close, max_levels=3, min_sep_ratio=0.007):
    """
    From a list of candidate levels (floats), select up to max_levels,
    ensuring each chosen level differs from others by at least min_sep_ratio (0.7% by default).
    Preference: most recent (we assume values are passed in recency order), then proximity to price.
    """
    if len(values) == 0: return []
    # First: dedupe by proximity (iterate from most recent end)
    chosen = []
    for v in values[::-1]:  # recent first
        if all(abs(v - c)/((v+c)/2.0) >= min_sep_ratio for c in chosen):
            chosen.append(float(v))
        if len(chosen) >= max_levels:
            break
    # If we picked more than needed (unlikely), keep closest to last_close
    chosen.sort(key=lambda x: abs(x - last_close))
    return chosen[:max_levels]

def get_support_resistance(df, look=CHART_LOOKBACK, window=3, max_levels=3):
    use = df.iloc[-look:].copy()
    highs = use["High"].values
    lows  = use["Low"].values
    closes = use["Close"].values
    last_close = float(closes[-1])
    # pivots
    h_idx, h_val = _pivot_points(highs, window=window, mode="high")
    l_idx, l_val = _pivot_points(lows,  window=window, mode="low")
    # Order by recency (use indices as x)
    h_order = np.argsort(h_idx) if len(h_idx)>0 else []
    l_order = np.argsort(l_idx) if len(l_idx)>0 else []
    h_vals_ordered = h_val[h_order] if len(h_idx)>0 else np.array([])
    l_vals_ordered = l_val[l_order] if len(l_idx)>0 else np.array([])
    # pick levels
    res_levels = pick_key_levels(h_vals_ordered, last_close, max_levels=max_levels)
    sup_levels = pick_key_levels(l_vals_ordered, last_close, max_levels=max_levels)
    return sup_levels, res_levels

# -------- Data (DAILY) --------
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
    except Exception as e:
        print(f"[warn] Stooq daily failed {t}: {e}")
        return None

def fetch_yahoo_daily(t, tries=4, base_sleep=1.5):
    for k in range(tries):
        try:
            df = yf.download(tickers=t, period=YAHOO_PERIOD, interval="1d",
                             auto_adjust=False, progress=False, session=SESS)
            if df is not None and not df.empty: return df
        except Exception as e:
            msg = str(e)
            if "429" in msg or "Too Many Requests" in msg:
                wait = base_sleep * (k+1) * 2
                print(f"[429] Yahoo daily rate limited {t}; wait {wait:.1f}s")
                time.sleep(wait)
            else:
                print(f"[warn] Yahoo daily {t} try {k+1}/{tries}: {repr(e)}")
        time.sleep(base_sleep)
    return None

def clean_and_summarize(df):
    cols = [c for c in ["Open","High","Low","Close"] if c in df.columns]
    if len(cols) < 4: return None
    df = df[["Open","High","Low","Close"]].dropna()
    if df.empty: return None
    dfc = df.iloc[-CHART_LOOKBACK:].copy()
    if dfc.empty or len(dfc) < 10: return None
    dfs = df.iloc[-SUMMARY_LOOKBACK:].copy()
    last = float(dfc["Close"].iloc[-1])
    chg30 = (last/float(dfs["Close"].iloc[0]) - 1.0) * 100.0 if len(dfs)>1 else 0.0
    # compute support/resistance here to avoid recomputation
    sup_levels, res_levels = get_support_resistance(dfc, look=CHART_LOOKBACK, window=3, max_levels=3)
    return (dfc, last, chg30, sup_levels, res_levels)

def fetch_all_daily(tickers):
    out = {t: None for t in tickers}
    for t in tickers:
        df = fetch_stooq_daily(t)
        if df is None: df = fetch_yahoo_daily(t)
        out[t] = clean_and_summarize(df) if df is not None else None
        time.sleep(2.0)
    return out

# -------- Draw card --------
def draw_card(d, box, ticker, df, last, chg30, sup_levels, res_levels):
    x0,y0,x1,y1 = box
    # container
    d.rounded_rectangle((x0+6,y0+6,x1+6,y1+6), radius=14, fill=(230,230,230))
    d.rounded_rectangle((x0,y0,x1,y1), radius=14, fill=CARD_BG)
    # neutral accent strip
    d.rectangle((x0,y0,x0+12,y1), fill=ACCENT)

    pad=24; info_x=x0+pad; info_y=y0+pad
    d.text((info_x,info_y), ticker, fill=TEXT_MAIN, font=F_TICK)
    d.text((info_x,info_y+60), f"{last:,.2f} USD", fill=TEXT_MAIN, font=F_NUM)
    d.text((info_x,info_y+105), f"{chg30:+.2f}% past {SUMMARY_LOOKBACK}d", fill=TEXT_MUT, font=F_CHG)

    # chart area
    cx0=x0+380; cx1=x1-pad; cy0=y0+pad; cy1=y1-pad-18
    d.rectangle((cx0,cy0,cx1,cy1), fill=(255,255,255))

    vmin=float(df["Low"].min()); vmax=float(df["High"].max())
    n=len(df)
    step_candle=(cx1-cx0)/max(1,n); wick=max(1,int(step_candle*0.12)); body=max(3,int(step_candle*0.4))

    # grid
    for gy in (0.25,0.5,0.75):
        y=int(cy0 + gy*(cy1-cy0))
        d.line([(cx0+4,y),(cx1-4,y)], fill=GRID, width=1)

    # candles
    for i,row in enumerate(df.itertuples(index=False)):
        o,h,l,c = float(row.Open), float(row.High), float(row.Low), float(row.Close)
        cx=int(cx0+i*step_candle+step_candle*0.5)
        yH=clamp(y_map(h,vmin,vmax,cy0,cy1), cy0,cy1)
        yL=clamp(y_map(l,vmin,vmax,cy0,cy1), cy0,cy1)
        yO=clamp(y_map(o,vmin,vmax,cy0,cy1), cy0,cy1)
        yC=clamp(y_map(c,vmin,vmax,cy0,cy1), cy0,cy1)
        col=UP_COL if c>=o else DOWN_COL
        d.line([(cx,yH),(cx,yL)], fill=col, width=wick)
        top,bot=min(yO,yC),max(yO,yC)
        if bot-top<2: bot=top+2
        d.rectangle((cx-body//2, top, cx+body//2, bot), fill=col)

    # --- Support & Resistance lines (no price labels) ---
    def draw_level(y_val, color, width=3):
        y = clamp(y_map(y_val, vmin, vmax, cy0, cy1), cy0, cy1)
        d.line([(cx0, y), (cx1, y)], fill=color, width=width)

    for r in res_levels:  # resistance first (red)
        draw_level(r, DOWN_COL, width=3)
    for s in sup_levels:  # support (green)
        draw_level(s, UP_COL, width=3)

    d.text((cx0, cy1+4), f"{CHART_LOOKBACK} daily bars • support/resistance", fill=TEXT_MUT, font=F_META)

# -------- Page render --------
def render_image(path, data_map):
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), BG); d = ImageDraw.Draw(img)
    d.text((64,50), "ONES TO WATCH", fill=TEXT_MAIN, font=F_TITLE)
    d.text((64,120), "Daily charts • key support & resistance", fill=TEXT_MUT, font=F_SUB)

    if os.path.exists(LOGO_PATH):
        try:
            logo = Image.open(LOGO_PATH).convert("RGBA"); logo.thumbnail((200,200))
            img.paste(logo, (CANVAS_W - logo.width - 56, 44), logo)
        except: pass

    cont=(48,220,CANVAS_W-48,CANVAS_H-60); margin=28
    card_h=int((cont[3]-cont[1]-margin*4)/3)
    x=cont[0]+margin; w=(cont[2]-cont[0])-margin*2; y=cont[1]+margin

    for t in TICKERS:
        payload = data_map.get(t)
        if not payload:
            d.rounded_rectangle((x+6,y+6,x+w+6,y+card_h+6),14,fill=(230,230,230))
            d.rounded_rectangle((x,y,x+w,y+card_h),14,fill=CARD_BG)
            d.rectangle((x,y,x+12,y+card_h), fill=ACCENT)
            d.text((x+40,y+40), f"{t} – data unavailable", fill=DOWN_COL, font=F_TICK)
        else:
            df,last,chg30,sups,ress = payload
            draw_card(d,(x,y,x+w,y+card_h), t, df, last, chg30, sups, ress)
        y += card_h + margin

    d.text((64, CANVAS_H-30), "Ideas only – Not financial advice", fill=TEXT_MUT, font=F_META)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img.save(path, "PNG", optimize=True)

# -------- Docs/RSS --------
def write_docs(latest_filename, ts_str):
    os.makedirs(DOCS_DIR, exist_ok=True)
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{BRAND_NAME} – Ones to Watch</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;
      margin:0;padding:24px;background:#fff;color:#111}}
.wrapper{{max-width:1080px;margin:0 auto;text-align:center}}
img{{max-width:100%;height:auto;border-radius:12px;
     box-shadow:0 10px 30px rgba(0,0,0,.08)}}
</style></head>
<body>
<div class="wrapper">
<h1>{BRAND_NAME} – Ones to Watch</h1>
<p>Latest post image below. Subscribe via <a href="feed.xml">RSS</a>.</p>
<img src="../output/{latest_filename}" alt="daily image"/>
<p style="color:#666;font-size:14px">Ideas only – Not financial advice</p>
</div>
</body></html>"""
    with open(os.path.join(DOCS_DIR, "index.html"), "w", encoding="utf-8") as f: f.write(html)

    feed = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>{BRAND_NAME} – Daily</title>
    <link>{PAGES_URL}</link>
    <description>Daily image for Instagram automation.</description>
    <item>
      <title>Ones to Watch {ts_str}</title>
      <link>{PAGES_URL}output/{latest_filename}</link>
      <guid isPermaLink="false">{ts_str}</guid>
      <pubDate>{ts_str}</pubDate>
      <enclosure url="{PAGES_URL}output/{latest_filename}" type="image/png" />
      <description>Daily watchlist image.</description>
    </item>
  </channel>
</rss>"""
    with open(os.path.join(DOCS_DIR, "feed.xml"), "w", encoding="utf-8") as f: f.write(feed)

# -------- Main --------
if __name__ == "__main__":
    now = datetime.datetime.now(pytz.timezone(TIMEZONE))
    datestr = now.strftime("%Y%m%d")
    out_name = f"twd_{datestr}.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    data_map = fetch_all_daily(TICKERS)
    render_image(out_path, data_map)

    ts_str = now.strftime("%a, %d %b %Y %H:%M:%S %z")
    write_docs(out_name, ts_str)
    print("done:", out_path)
