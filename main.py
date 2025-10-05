import os, datetime, pytz, random
import pandas as pd
import yfinance as yf
from PIL import Image, ImageDraw, ImageFont
from xml.etree import ElementTree as ET

BRAND = "TrendWatchDesk"
TZ = os.getenv("TZ", "Europe/London")
W, H = 1080, 1350
BG = (11,15,20); FG = (234,242,255)

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

def universe():
    return ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","AMD","TSLA","JPM","UNH","XOM","COST","WMT","ADBE","ORCL","INTC","MRK","BA","PFE"]

def fetch_change(t):
    try:
        df = yf.Ticker(t).history(period="5d", interval="1d")
        if len(df)<2: return None
        last, prev = df["Close"].iloc[-1], df["Close"].iloc[-2]
        return float((last-prev)/prev*100.0)
    except Exception:
        return None

def score():
    rows=[]
    for t in universe():
        ch=fetch_change(t)
        if ch is not None: rows.append({"ticker":t,"change":ch})
    df=pd.DataFrame(rows).sort_values("change", ascending=False)
    return df

def select_lists(df,n=3):
    if df.empty or len(df)<6:
        u=universe(); import random as _r; _r.shuffle(u)
        return u[:n], u[n:n*2]
    return df.head(n).ticker.tolist(), df.tail(n).ticker.tolist()[::-1]

def draw_image(longs, shorts, out_path):
    img=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(img)

    # Title & date
    now=datetime.datetime.now(pytz.timezone(TZ))
    title=f"{BRAND} — Stocks to Watch"
    d.text((60,50),title,fill=FG,font=F_H1)
    date=now.strftime("%d %b %Y"); d.text((60,140),date,fill=FG,font=F_S)

    # Logo (optional)
    try:
        logo=Image.open("logo.png").convert("RGBA")
        logo.thumbnail((220,220))
        img.paste(logo,(W-logo.width-40,40),logo)
    except Exception as e:
        print("Logo not found/used:", e)

    # Sections
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
    df=score(); longs,shorts=select_lists(df)
    os.makedirs("docs",exist_ok=True)
    fname=f"twd_{datetime.datetime.utcnow().strftime('%Y%m%d')}.png"
    path=os.path.join("docs",fname)
    draw_image(longs,shorts,path)
    caption="\n".join([
        f"{BRAND} — Ones to Watch {datetime.datetime.now().strftime('%d %b %Y')}",
        f"Upside: {', '.join(longs)}",
        f"Possible Sell: {', '.join(shorts)}",
        "Not financial advice."
    ])
    base=os.getenv("PAGES_BASE","https://<your-username>.github.io/trendwatchdesk-bot")
    image_url=f"{base}/{fname}"
    append_rss(image_url,caption,"docs/feed.xml")
    print("Generated:",image_url)

if __name__=="__main__": main()
