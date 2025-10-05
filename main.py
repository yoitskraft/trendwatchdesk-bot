import os, random, time, datetime, pytz
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import yfinance as yf
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# -------- CONFIG --------
BRAND_NAME = "TrendWatchDesk"
TIMEZONE   = "Europe/London"
CANVAS_W, CANVAS_H = 1080, 1350

BG        = (255,255,255)
CARD_BG   = (250,250,250)
TEXT_MAIN = (20,20,22)
TEXT_MUT  = (92,95,102)
GRID      = (225,228,232)
UP_COL    = (20,170,90)
DOWN_COL  = (230,70,70)
HOLD_COL  = (130,130,130)
SUPPORT   = (0,0,0)

# ---------- Weighted random ticker pool ----------
POOL = {
    "NVDA": 5, "MSFT": 4, "TSLA": 3, "AMZN": 5, "META": 4, "GOOG": 4, "AMD": 3,
    "UNH": 2, "AAPL": 5, "NFLX": 2, "BABA": 2, "JPM": 2, "DIS": 2, "BA": 1,
    "ORCL": 2, "NKE": 1, "PYPL": 1, "INTC": 2, "CRM": 2, "KO": 2
}
N_TICKERS = 3

def weighted_sample(pool: dict, n: int, seed: str):
    tickers = list(pool.keys())
    weights = list(pool.values())
    expanded = [t for t,w in zip(tickers,weights) for _ in range(max(1,int(w)))]
    rnd = random.Random(seed)
    n = min(n, len(set(expanded)))
    picked = []
    while len(picked) < n and expanded:
        t = rnd.choice(expanded)
        if t not in picked:
            picked.append(t)
    return picked

today_seed = datetime.date.today().strftime("%Y%m%d")
TICKERS = weighted_sample(POOL, N_TICKERS, seed=today_seed)

OUTPUT_DIR= "output"
DOCS_DIR  = "docs"
LOGO_PATH = "assets/logo.png"            # optional
PAGES_URL = "https://<your-username>.github.io/trendwatchdesk-bot/"

# -------- FONTS --------
def load_font(size=42, bold=False):
    pref = "fonts/Roboto-Bold.ttf" if bold else "fonts/Roboto-Regular.ttf"
    if os.path.exists(pref):
        return ImageFont.truetype(pref, size)
    fam = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(fam, size)
    except:
        return ImageFont.load_default()

F_TITLE = load_font(65,  bold=True)
F_SUB   = load_font(38,  bold=False)
F_TICK  = load_font(54,  bold=True)
F_NUM   = load_font(46,  bold=True)
F_CHG   = load_font(34,  bold=True)
F_META  = load_font(24,  bold=False)
F_TAG   = load_font(30,  bold=True)

# -------- TECH --------
def rsi(series, period=14):
    s = pd.Series(series).dropna()
    if len(s) < period + 1: return None
    delta = s.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain).rolling(period).mean()
    roll_dn = pd.Series(loss).rolling(period).mean()
    rs = roll_up / (roll_dn + 1e-9)
    out = 100.0 - (100.0 / (1.0 + rs))
    return float(out.iloc[-1])

def ema_series(series, span):
    return pd.Series(series).ewm(span=span, adjust=False).mean()

def support_level(df, look=30):
    if df is None or df.empty: return None
    return float(df["Low"].iloc[-look:].min())

def recommend_signal(close_series, sup):
    s = pd.Series(close_series).dropna()
    if len(s) < 25: return "HOLD"
    ema10 = ema_series(s, 10).iloc[-1]
    ema20 = ema_series(s, 20).iloc[-1]
    last  = float(s.iloc[-1])
    r     = rsi(s, 14)
    prox = None if not sup else (last - sup) / sup * 100.0
    up, dn = (ema10 > ema20), (ema10 < ema20)
    ob, ok = (r is not None and r >= 70), (r is not None and 40 <= r <= 65)
    if up and ok and (prox is None or prox <= 1.5): return "BUY"
    if dn or ob or (prox is not None and prox < -1.5): return "SELL"
    return "HOLD"

# -------- Robust DATA fetch --------
def _make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    })
    retry = Retry(
        total=5, connect=5, read=5, backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

def _clean_frame(df):
    if df is None or df.empty: return None
    cols = [c for c in ["Open","High","Low","Close"] if c in df.columns]
    if len(cols) < 4: return None
    df = df[["Open","High","Low","Close"]].dropna()
    if df.empty: return None
    return df.iloc[-30:]

def _summarize(df):
    df30 = _clean_frame(df)
    if df30 is None: return None
    last = float(df30["Close"].iloc[-1])
    chg30 = (last/float(df30["Close"].iloc[0]) - 1.0) * 100.0
    sup = support_level(df30, 30)
    sig = recommend_signal(df30["Close"].values, sup)
    return (df30, last, chg30, sup, sig)

def _fetch_single(t, tries=3, sleep_sec=1.2):
    sess = _make_session()
    for k in range(tries):
        try:
            df = yf.download(
                tickers=t, period="45d", interval="1d",
                auto_adjust=False, progress=False, session=sess
            )
            if df is not None and not df.empty:
                out = _summarize(df)
                if out: return out
            else:
                print(f"[warn] {t} returned empty frame (try {k+1}/{tries})")
        except Exception as e:
            print(f"[warn] single fetch {t} failed (try {k+1}/{tries}):", repr(e))
        time.sleep(sleep_sec * (k+1))
    return None

def fetch_all_30d(tickers):
    out = {t: None for t in tickers}
    for t in tickers:
        out[t] = _fetch_single(t)
    return out

def y_map(v, vmin, vmax, y0, y1):
    if vmax - vmin < 1e-6: return (y0+y1)//2
    return int(y1 - (v - vmin) * (y1 - y0) / (vmax - vmin))

# -------- DRAW --------
def draw_card(d, box, ticker, df, last, chg30, sup, sig):
    x0,y0,x1,y1 = box
    d.rounded_rectangle((x0+6,y0+6,x1+6,y1+6), radius=14, fill=(230,230,230))
    d.rounded_rectangle((x0,y0,x1,y1), radius=14, fill=CARD_BG)
    strip_col = {"BUY":UP_COL, "SELL":DOWN_COL, "HOLD":HOLD_COL}[sig]
    d.rectangle((x0,y0,x0+12,y1), fill=strip_col)

    pad=24
    info_x=x0+pad; info_y=y0+pad
    d.text((info_x, info_y), ticker, fill=TEXT_MAIN, font=F_TICK)
    d.text((info_x, info_y+60), f"{last:,.2f} USD", fill=TEXT_MAIN, font=F_NUM)
    sign = "+" if chg30>=0 else ""
    d.text((info_x, info_y+105), f"{sign}{chg30:.2f}% past 30d",
           fill=(UP_COL if chg30>=0 else DOWN_COL), font=F_CHG)

    cx0=x0+380; cx1=x1-pad; cy0=y0+pad; cy1=y1-pad-18
    d.rectangle((cx0,cy0,cx1,cy1), fill=(255,255,255))
    vmin=float(df["Low"].min()); vmax=float(df["High"].max())
    n=len(df); step=(cx1-cx0)/max(1,n); wick=max(1,int(step*0.12)); body=max(3,int(step*0.4))
    for gy in (0.25,0.5,0.75):
        y=int(cy0 + gy*(cy1-cy0))
        d.line([(cx0+4,y),(cx1-4,y)], fill=GRID, width=1)
    for i,row in enumerate(df.itertuples(index=False)):
        o,h,l,c = float(row.Open), float(row.High), float(row.Low), float(row.Close)
        cx=int(cx0+i*step+step*0.5)
        yH=y_map(h,vmin,vmax,cy0,cy1); yL=y_map(l,vmin,vmax,cy0,cy1)
        yO=y_map(o,vmin,vmax,cy0,cy1); yC=y_map(c,vmin,vmax,cy0,cy1)
        col = UP_COL if c>=o else DOWN_COL
        d.line([(cx,yH),(cx,yL)], fill=col, width=wick)
        top=min(yO,yC); bot=max(yO,yC); 
        if bot-top<2: bot=top+2
        d.rectangle((cx-body//2, top, cx+body//2, bot), fill=col)
    if sup:
        ys=y_map(sup, vmin, vmax, cy0, cy1)
        d.line([(cx0+4,ys),(cx1-60,ys)], fill=SUPPORT, width=3)
        label=f"{sup:,.2f}"
        tw,th = d.textbbox((0,0), label, font=F_CHG)[2:]
        bx0=cx1-(tw+20)-8
        by0=max(ys-th//2-6, cy0+6); by0=min(by0, cy1-th-6)
        d.rounded_rectangle((bx0,by0,bx0+tw+20,by0+th+12), radius=8, fill=SUPPORT)
        d.text((bx0+10,by0+6), label, fill=(255,255,255), font=F_CHG)
    d.text((cx0, cy1+4), "30 days", fill=TEXT_MUT, font=F_META)

def render_image(path, data_map):
    img = Image.new("RGB",(CANVAS_W,CANVAS_H), BG)
    d   = ImageDraw.Draw(img)
    d.text((64,50), "ONES TO WATCH", fill=TEXT_MAIN, font=F_TITLE)
    d.text((64,120),"KEY BUY LEVELS THIS WEEK", fill=TEXT_MUT, font=F_SUB)
    d.text((64,165),"BUY",  fill=UP_COL,   font=F_TAG)
    d.text((160,165),"SELL", fill=DOWN_COL, font=F_TAG)
    d.text((290,165),"HOLD", fill=HOLD_COL, font=F_TAG)

    if os.path.exists(LOGO_PATH):
        try:
            logo = Image.open(LOGO_PATH).convert("RGBA")
            logo.thumbnail((200,200))
            img.paste(logo, (CANVAS_W - logo.width - 56, 44), logo)
        except: pass

    cont=(48,220,CANVAS_W-48,CANVAS_H-60)
    margin=28
    card_h=int((cont[3]-cont[1]-margin*4)/3)
    x=cont[0]+margin; w=(cont[2]-cont[0])-margin*2; y=cont[1]+margin

    for t in TICKERS:
        payload = data_map.get(t)
        if not payload:
            d.rounded_rectangle((x+6,y+6,x+w+6,y+card_h+6), radius=14, fill=(230,230,230))
            d.rounded_rectangle((x,y,x+w,y+card_h), radius=14, fill=CARD_BG)
            d.rectangle((x,y,x+12,y+card_h), fill=HOLD_COL)
            d.text((x+40,y+40), f"{t} – data unavailable", fill=DOWN_COL, font=F_TICK)
        else:
            df,last,chg30,sup,sig = payload
            draw_card(d,(x,y,x+w,y+card_h),t,df,last,chg30,sup,sig)
        y+=card_h+margin

    d.text((64, CANVAS_H-30), "Ideas only – Not financial advice", fill=TEXT_MUT, font=F_META)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img.save(path, "PNG", optimize=True)

def write_docs(latest_filename, ts_str):
    os.makedirs(DOCS_DIR, exist_ok=True)
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{BRAND_NAME} – Ones to Watch</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;margin:0;padding:24px;background:#fff;color:#111}}
.wrapper{{max-width:1080px;margin:0 auto;text-align:center}}
img{{max-width:100%;height:auto;border-radius:12px;box-shadow:0 10px 30px rgba(0,0,0,.08)}}</style></head>
<body><div class="wrapper">
<h1>{BRAND_NAME} – Ones to Watch</h1>
<p>Latest post image below. Subscribe via <a href="feed.xml">RSS</a>.</p>
<img src="../output/{latest_filename}" alt="daily image"/>
<p style="color:#666;font-size:14px">Ideas only – Not financial advice</p>
</div></body></html>"""
    with open(os.path.join(DOCS_DIR,"index.html"),"w",encoding="utf-8") as f: f.write(html)

    feed = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
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
</channel></rss>"""
    with open(os.path.join(DOCS_DIR,"feed.xml"),"w",encoding="utf-8") as f: f.write(feed)

if __name__ == "__main__":
    now = datetime.datetime.now(pytz.timezone(TIMEZONE))
    datestr = now.strftime("%Y%m%d")
    out_name = f"twd_{datestr}.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    data_map = fetch_all_30d(TICKERS)
    render_image(out_path, data_map)

    ts_str = now.strftime("%a, %d %b %Y %H:%M:%S %z")
    write_docs(out_name, ts_str)
    print("done:", out_path)
