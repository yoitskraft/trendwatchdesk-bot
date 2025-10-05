import os, datetime, pytz, random, time, math
import pandas as pd
import yfinance as yf
from PIL import Image, ImageDraw, ImageFont
from xml.etree import ElementTree as ET

BRAND = "TrendWatchDesk"
TZ = os.getenv("TZ", "Europe/London")
W, H = 1080, 1350

# Neon dark palette
BG  = (8,10,14)
FG  = (230,240,255)
SUB = (160,176,196)
CYAN = (0, 220, 255)
GREEN = (0, 190, 140)
RED = (240, 80, 80)
CARD_BG = (15,18,24)
STROKE = (30,40,60)

TICKERS = ["NVDA","MSFT","TSLA","AAPL","GOOGL","AMZN"]

def load_font(size=42):
    for path in ["/System/Library/Fonts/SFNS.ttf",
                 "/Library/Fonts/Arial Bold.ttf",
                 "/Library/Fonts/Arial.ttf",
                 "/Library/Fonts/Helvetica.ttc"]:
        if os.path.exists(path):
            try: return ImageFont.truetype(path,size)
            except Exception: pass
    return ImageFont.load_default()

F_HUGE = load_font(120)
F_H1 = load_font(80)
F_H2 = load_font(50)
F_B  = load_font(40)
F_S  = load_font(28)

# ---------- Data ----------
def pct_change_5d(t):
    try:
        df = yf.Ticker(t).history(period="10d", interval="1d").dropna()
        if len(df) < 6: return None
        last = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-6])
        return last, (last - prev)/prev*100.0
    except Exception:
        return None

def score(max_retries=2, sleep_s=2):
    # simple ranking by 5d change
    rows=[]
    for t in TICKERS:
        out = pct_change_5d(t)
        if out:
            last, pc = out
            rows.append({"ticker":t,"last":last,"chg5":pc})
    df = pd.DataFrame(rows).sort_values("chg5", ascending=False)
    return df

def ohlc(t, days=30):
    try:
        df = yf.Ticker(t).history(period=f"{days+10}d", interval="1d").dropna().iloc[-days:]
        return df[["Open","High","Low","Close"]]
    except Exception:
        return pd.DataFrame(columns=["Open","High","Low","Close"])

# ---------- Drawing helpers ----------
def draw_header(img, d):
    # Title
    d.text((60, 40), "ONES TO", fill=CYAN, font=F_HUGE)
    d.text((60, 150), "WATCH", fill=CYAN, font=F_HUGE)
    d.text((60, 270), "KEY BUY LEVELS THIS WEEK", fill=FG, font=F_H2)
    # Logo (optional)
    try:
        logo = Image.open("logo.png").convert("RGBA")
        logo.thumbnail((220,220))
        img.paste(logo, (W - logo.width - 40, 60), logo)
        d.text((W - logo.width - 40, 60 + logo.height + 10), BRAND, fill=FG, font=F_B)
    except Exception:
        d.text((W-480, 80), BRAND, fill=FG, font=F_B)

def fmt_price(x):
    if x is None: return ""
    if abs(x) >= 1000: return f"{x:,.1f}"
    if abs(x) >= 100: return f"{x:,.2f}"
    if abs(x) >= 10: return f"{x:,.2f}"
    return f"{x:,.3f}"

def recent_support(df, lookback=20):
    if df is None or df.empty: return None
    lows = df["Low"].dropna()
    if len(lows) < 3: return None
    return float(lows.iloc[-lookback:].min())

def recent_resistance(df, lookback=20):
    if df is None or df.empty: return None
    highs = df["High"].dropna()
    if len(highs) < 3: return None
    return float(highs.iloc[-lookback:].max())

def normalize_y(v, vmin, vmax, y0, y1):
    if vmax - vmin < 1e-9: return (y0+y1)//2
    return int(y1 - (v - vmin) * (y1 - y0) / (vmax - vmin))

def draw_candles(d, box, df):
    x0,y0,x1,y1 = box
    d.rounded_rectangle(box, radius=24, fill=CARD_BG, outline=CYAN, width=2)
    if df is None or df.empty: return
    # padding
    pad=20
    x0+=pad; x1-=pad; y0+=pad; y1-=pad
    vmin = float(df["Low"].min()); vmax=float(df["High"].max())
    n=len(df)
    if n<2: return
    step = (x1-x0)/n
    wick_w = max(1,int(step*0.15))
    body_w = max(4,int(step*0.45))
    for i,row in enumerate(df.itertuples(index=False)):
        o,h,l,c = float(row.Open), float(row.High), float(row.Low), float(row.Close)
        cx = int(x0 + i*step + step*0.5)
        y_high = normalize_y(h,vmin,vmax,y0,y1)
        y_low  = normalize_y(l,vmin,vmax,y0,y1)
        y_open = normalize_y(o,vmin,vmax,y0,y1)
        y_close= normalize_y(c,vmin,vmax,y0,y1)
        col = GREEN if c>=o else RED
        # wick
        d.line([(cx, y_high), (cx, y_low)], fill=col, width=wick_w)
        # body
        top = min(y_open, y_close); bot = max(y_open, y_close)
        if abs(bot-top) < 2: bot = top+2
        d.rectangle([cx-body_w//2, top, cx+body_w//2, bot], fill=col)

def draw_level_with_label(d, box, level, color):
    if level is None: return
    x0,y0,x1,y1 = box
    pad=20
    x0+=pad; x1-=pad; y0+=pad; y1-=pad
    # map to y
    # we don't have vmin/vmax here, so store them during candles draw? Instead compute outside; pass via closure.
    # For simplicity, caller ensures y mapping by passing y via normalized value
    pass

# ---------- Compose ----------
def draw_card_block(img, d, x, y, w, h, t, last_price=None, chg5=None):
    # Outer rounded container (thin cyan stroke)
    d.rounded_rectangle([x,y,x+w,y+h], radius=30, outline=CYAN, width=3, fill=BG)
    # Header row
    d.ellipse([x+22, y+22, x+22+46, y+22+46], fill=(24,150,255))  # avatar dot
    d.text((x+80, y+20), f"{t}", fill=FG, font=F_B)
    if last_price is not None:
        d.text((x+80, y+62), f"{fmt_price(last_price)} USD", fill=FG, font=F_H2)
    if chg5 is not None:
        sign = "+" if chg5>=0 else ""
        col = GREEN if chg5>=0 else RED
        d.text((x+80, y+110), f"{sign}{chg5:.2f}% past 5d", fill=col, font=F_S)

    # Candlestick area
    chart_box = (x+40, y+160, x+w-40, y+h-80)
    df = ohlc(t, days=30)
    draw_candles(d, chart_box, df)

    # Support label (KEY BUY LEVEL)
    sup = recent_support(df, lookback=20)
    if sup:
        # horizontal line + label at right
        sx0, sy0, sx1, sy1 = chart_box
        vmin = float(df["Low"].min()); vmax=float(df["High"].max())
        y_sup = normalize_y(sup, vmin, vmax, sy0+20, sy1-20)
        d.line([(sx0+10, y_sup), (sx1-60, y_sup)], fill=CYAN, width=3)
        txt = fmt_price(sup)
        tw, th = d.textbbox((0,0), txt, font=F_B)[2:]
        bx0 = sx1 - (tw+28) - 10
        by0 = max(y_sup - th//2 - 8, sy0+10)
        by0 = min(by0, sy1 - th - 10)
        d.rounded_rectangle([bx0, by0, bx0+tw+28, by0+th+16], radius=12, fill=CYAN)
        d.text((bx0+14, by0+8), txt, fill=(10,14,20), font=F_B)
        d.text((sx0+10, sy1+8), "5 days", fill=SUB, font=F_S)

def draw_image(longs, shorts, out_path):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    draw_header(img, d)

    # Big card container similar to reference
    container = (50, 360, W-50, H-40)
    d.rounded_rectangle(container, radius=36, outline=CYAN, width=3, fill=BG)

    # We'll render three cards stacked inside container: top 3 "longs"
    margin = 30
    card_h = int((container[3]-container[1] - margin*4)/3)
    x = container[0]+margin; w = (container[2]-container[0]) - margin*2
    y = container[1]+margin
    for i, t in enumerate(longs[:3]):
        last=None; chg=None
        row = None
        try:
            df = score()
            row = df[df["ticker"]==t].iloc[0] if not df.empty else None
        except Exception:
            pass
        if row is not None:
            last = float(row["last"]); chg = float(row["chg5"])
        draw_card_block(img, d, x, y, w, card_h, t, last_price=last, chg5=chg)
        y += card_h + margin

    img.save(out_path,"PNG")

# ---------- RSS + Main ----------
def append_rss(image_url,caption,feed_path="docs/feed.xml"):
    now=datetime.datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")
    if not os.path.exists(feed_path):
        rss = ET.Element("rss",version="2.0")
        ch = ET.SubElement(rss,"channel")
        ET.SubElement(ch,"title").text=BRAND
        ET.SubElement(ch,"link").text=image_url
        ET.SubElement(ch,"description").text="Daily Stocks to Watch"
        tree=ET.ElementTree(rss); tree.write(feed_path, encoding="utf-8", xml_declaration=True)
    tree=ET.parse(feed_path); ch=tree.getroot().find("channel")
    item=ET.SubElement(ch,"item")
    ET.SubElement(item,"title").text=caption.splitlines()[0]
    ET.SubElement(item,"link").text=image_url
    ET.SubElement(item,"description").text=caption
    ET.SubElement(item,"pubDate").text=now
    tree.write(feed_path, encoding="utf-8", xml_declaration=True)

def main():
    df = score()
    longs = df.ticker.tolist()[:3] if df is not None and not df.empty else TICKERS[:3]
    shorts = TICKERS[-3:]  # not used in this layout
    os.makedirs("docs", exist_ok=True)
    fname = f"twd_{datetime.datetime.utcnow().strftime('%Y%m%d')}.png"
    path = os.path.join("docs", fname)
    draw_image(longs, shorts, path)

    caption = "\n".join([
        f"{BRAND} — Ones to Watch {datetime.datetime.now().strftime('%d %b %Y')}",
        f"Upside focus: {', '.join(longs)}",
        "Graphic includes last 30d candlesticks and key buy level.",
        "Not financial advice."
    ])

    base = os.getenv("PAGES_BASE", "https://<your-username>.github.io/trendwatchdesk-bot")
    image_url = f"{base}/{fname}"
    append_rss(image_url, caption, "docs/feed.xml")
    print("Generated:", image_url)

if __name__=="__main__":
    main()
