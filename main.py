import os, datetime, pytz, random, time
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
        # Two possible shapes:
        # 1) MultiIndex columns: (ticker, field)
        # 2) Column-per-field with top-level tickers (older behavior)
        for t in TICKERS:
            try:
                if isinstance(df.columns, pd.MultiIndex):
                    close = df.loc[:, (t, "Close")].dropna()
                else:
                    # older shape: df[t] is a subframe
                    sub = df[t] if t in df.columns.get_level_values(0) else df
                    close = (sub["Close"] if "Close" in sub else pd.Series(dtype=float)).dropna()

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

def draw_image(longs, shorts, out_path):
    img=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(img)
    now=datetime.datetime.now(pytz.timezone(TZ))
    title=f"{BRAND} — Stocks to Watch"
    d.text((60,50),title,fill=FG,font=F_H1)
    date=now.strftime("%d %b %Y"); d.text((60,140),date,fill=FG,font=F_S)

    try:
        logo=Image.open("logo.png").convert("RGBA")
        logo.thumbnail((220,220))
        img.paste(logo,(W-logo.width-40,40),logo)
    except Exception as e:
        print("Logo not found/used:", e)

    y0=240
    d.text((60,y0),"📈 Upside",fill=FG,font=F_H2); y=y0+70
    for t in longs:
        d.text((80,y),f"${t}",fill=FG,font=F_B); y+=60

    d.text((560,y0),"📉 Possible Sell",fill=FG,font=F_H2); y=y0+70
    for t in shorts:
        d.text((580,y),f"${t}",fill=FG,font=F_B); y+=60

    foot="Auto-generated • Not financial advice"
    d.text((60,H-80),foot,fill=(170,182,197),font=F_S)
    img.save(out_path,"PNG")

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
