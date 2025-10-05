import os, datetime, pytz, random, time, math
import pandas as pd
import yfinance as yf
from PIL import Image, ImageDraw, ImageFont
from xml.etree import ElementTree as ET

BRAND = "TrendWatchDesk"
TZ = os.getenv("TZ", "Europe/London")
W, H = 1080, 1350
BG = (11,15,20); FG = (234,242,255)

TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","AMD","TSLA",
    "JPM","UNH","XOM","COST","WMT","ADBE","ORCL","INTC","MRK","BA","PFE"
]

# ----------------------------
# Fonts
# ----------------------------
def load_font(size=42):
    for path in ["/System/Library/Fonts/SFNS.ttf",
                 "/Library/Fonts/Arial.ttf",
                 "/Library/Fonts/Helvetica.ttc"]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()

F_H1=load_font(72); F_H2=load_font(54); F_B=load_font(38); F_S=load_font(30)

# ----------------------------
# Data fetching (robust)
# ----------------------------
def score(max_retries=3, sleep_s=3):
    """Fetch 1-day % change for all tickers, robust to both yfinance shapes."""
    df = None
    for attempt in range(max_retries):
        try:
            df = yf.download(
                tickers=" ".join(TICKERS),
                period="5d", interval="1d",
                group_by="ticker", auto_adjust=False,
                threads=True, progress=False
            )
            break
        except Exception as e:
            print(f"[score] batch download attempt {attempt+1} failed: {e}")
            time.sleep(sleep_s)

    rows = []

    if df is not None and len(df) > 0:
        for t in TICKERS:
            try:
                if isinstance(df.columns, pd.MultiIndex):
                    close = df.loc[:, (t, "Close")] if (t, "Close") in df.columns else pd.Series(dtype=float)
                else:
                    # older shape (single level) - try df[t]['Close'] or df['Close']
                    if t in df.columns:
                        sub = df[t]
                        close = sub["Close"] if "Close" in sub else pd.Series(dtype=float)
                    else:
                        close = df["Close"] if "Close" in df else pd.Series(dtype=float)
                close = close.dropna()
                if len(close) >= 2:
                    last, prev = float(close.iloc[-1]), float(close.iloc[-2])
                    rows.append({"ticker": t, "change": (last - prev) / prev * 100.0})
            except Exception as e:
                print(f"[score] parse skipped {t}: {e}")

    if not rows:
        print("[score] no rows from batch; falling back per-ticker…")
        for t in TICKERS:
            try:
                h = yf.Ticker(t).history(period="5d", interval="1d")
                close = h["Close"].dropna()
                if len(close) >= 2:
                    last, prev = float(close.iloc[-1]), float(close.iloc[-2])
                    rows.append({"ticker": t, "change": (last - prev) / prev * 100.0})
            except Exception as e:
                print(f"[score] fallback fetch failed for {t}: {e}")

    out = pd.DataFrame(rows).sort_values("change", ascending=False)
    if out.empty:
        print("[score] still empty; using random fallback")
    return out

def select_lists(df, n=3):
    pool = TICKERS.copy()
    if df is None or df.empty or len(df) < 2*n:
        random.shuffle(pool)
        return pool[:n], pool[n:2*n]
    return df.head(n).ticker.tolist(), df.tail(n).ticker.tolist()[::-1]

# ----------------------------
# Chart helpers
# ----------------------------
def ema(series, span=20):
    return series.ewm(span=span, adjust=False).mean()

def recent_levels(series, lookback=30):
    s = series.dropna()
    if len(s) < lookback: lookback = len(s)
    if lookback < 5:
        return (None, None)
    window = s.iloc[-lookback:]
    return (float(window.min()), float(window.max()))

def normalize_points(vals, box_w, box_h, pad=8):
    if len(vals) < 2:
        return ([(pad, box_h//2), (box_w-pad, box_h//2)], 0, 1)
    vmin, vmax = min(vals), max(vals)
    if math.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    pts = []
    n = len(vals)
    for i, v in enumerate(vals):
        x = pad + int((i/(n-1))*(box_w-2*pad))
        y = pad + int((1 - (v - vmin)/(vmax - vmin))*(box_h-2*pad))
        pts.append((x, y))
    return pts, vmin, vmax

def draw_sparkline(d, xy, w, h, closes, support=None, resistance=None, ema_vals=None,
                   line=(230,230,230), ema_color=(120,170,255),
                   sup_color=(0,200,140), res_color=(240,90,90)):
    x0, y0 = xy
    # frame
    d.rounded_rectangle([x0, y0, x0+w, y0+h], radius=18, outline=(55,65,80), width=2, fill=(20,26,34))
    if len(closes) < 2:
        return
    pts, vmin, vmax = normalize_points(closes, w, h)
    pts = [(x0+px, y0+py) for (px,py) in pts]
    # price line
    for i in range(1, len(pts)):
        d.line([pts[i-1], pts[i]], fill=line, width=3)
    # EMA
    if ema_vals is not None and len(ema_vals) == len(closes):
        epts, _, _ = normalize_points(ema_vals, w, h)
        epts = [(x0+px, y0+py) for (px,py) in epts]
        for i in range(1, len(epts)):
            d.line([epts[i-1], epts[i]], fill=ema_color, width=2)
    # support/resistance
    def y_for_level(level):
        if level is None: return None
        if math.isclose(vmin, vmax):
            return y0 + h//2
        yrel = (1 - (level - vmin)/(vmax - vmin))
        return y0 + 8 + int(yrel*(h-16))
    ys = y_for_level(support)
    yr = y_for_level(resistance)
    if ys is not None:
        d.line([(x0+8, ys), (x0+w-8, ys)], fill=sup_color, width=2)
    if yr is not None:
        d.line([(x0+8, yr), (x0+w-8, yr)], fill=res_color, width=2)

# ----------------------------
# Painter
# ----------------------------
def draw_image(longs, shorts, out_path):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # Top banner
    banner_h = 160
    d.rectangle([0,0,W,banner_h], fill=(28,40,64))
    title = f"{BRAND} — Ones to Watch"
    d.text((60, 42), title, fill=(255,255,255), font=F_H1)
    date = datetime.datetime.now(pytz.timezone(TZ)).strftime("%d %b %Y")
    d.text((60, 120), date, fill=(200,220,255), font=F_S)

    # Logo (optional)
    try:
        logo = Image.open("logo.png").convert("RGBA")
        logo.thumbnail((220,220))
        img.paste(logo, (W - logo.width - 40, 24), logo)
    except Exception as e:
        print("Logo not found/used:", e)

    # Section titles
    left_x, right_x = 60, W//2 + 20
    top_y = banner_h + 20
    d.text((left_x, top_y), "📈 Upside", fill=(0, 224, 164), font=F_H2)
    d.text((right_x, top_y), "📉 Possible Sell", fill=(244, 92, 92), font=F_H2)

    # Fetch chart data helper
    def get_prices(t):
        try:
            h = yf.Ticker(t).history(period="180d", interval="1d")["Close"].dropna()
            if len(h) > 60:
                h = h.iloc[-60:]
            return h
        except Exception as e:
            print("history fail", t, e)
            return pd.Series(dtype=float)

    # Layout for 3 cards per column
    card_w, card_h = 460, 240
    ygap = 26
    start_y = top_y + 70

    # Left column (Upside)
    y = start_y
    for t in longs[:3]:
        closes = get_prices(t)
        ema20 = ema(closes, 20) if len(closes) else None
        sup, res = recent_levels(closes, lookback=30)
        d.text((left_x, y - 34), f"${t}", fill=FG, font=F_B)
        draw_sparkline(d, (left_x, y), card_w, card_h, list(closes.values), support=sup, resistance=res,
                       ema_vals=(list(ema20.values) if ema20 is not None else None))
        y += card_h + ygap

    # Right column (Possible Sell)
    y = start_y
    for t in shorts[:3]:
        closes = get_prices(t)
        ema20 = ema(closes, 20) if len(closes) else None
        sup, res = recent_levels(closes, lookback=30)
        d.text((right_x, y - 34), f"${t}", fill=FG, font=F_B)
        draw_sparkline(d, (right_x, y), card_w, card_h, list(closes.values), support=sup, resistance=res,
                       ema_vals=(list(ema20.values) if ema20 is not None else None))
        y += card_h + ygap

    # Footer
    d.rectangle([0, H-110, W, H], fill=(22, 22, 26))
    d.text((60, H-86), "Auto-generated • Not financial advice", fill=(170,182,197), font=F_S)
    d.text((60, H-52), "#stocks #trading #marketwatch", fill=(130,140,150), font=F_S)

    img.save(out_path, "PNG")

# ----------------------------
# RSS feed
# ----------------------------
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

# ----------------------------
# Main
# ----------------------------
def main():
    df = score()
    print("[debug] scored rows:", 0 if df is None else len(df))
    if df is not None and not df.empty:
        print(df.head(5))

    longs, shorts = select_lists(df)
    os.makedirs("docs", exist_ok=True)
    fname = f"twd_{datetime.datetime.utcnow().strftime('%Y%m%d')}.png"
    path = os.path.join("docs", fname)
    draw_image(longs, shorts, path)

    caption = "\n".join([
        f"{BRAND} — Ones to Watch {datetime.datetime.now().strftime('%d %b %Y')}",
        f"Upside: {', '.join(longs)}",
        f"Possible Sell: {', '.join(shorts)}",
        "Not financial advice."
    ])

    base = os.getenv("PAGES_BASE", "https://<your-username>.github.io/trendwatchdesk-bot")
    image_url = f"{base}/{fname}"
    append_rss(image_url, caption, "docs/feed.xml")

    print("Generated:", image_url)

if __name__=="__main__":
    main()
