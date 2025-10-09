#!/usr/bin/env python3
import os, io, sys, time, shutil
from typing import Dict
import requests
from PIL import Image, ImageDraw, ImageFont

LOGO_DIR = "assets/logos"
FONT_DIR = "assets/fonts"
os.makedirs(LOGO_DIR, exist_ok=True)

MIRRORS = {"GOOGL": "GOOG"}  # if GOOG exists but GOOGL doesn't

TICKER_DOMAIN: Dict[str, str] = {
    "AAPL":"apple.com","MSFT":"microsoft.com","GOOGL":"google.com","GOOG":"google.com","META":"about.facebook.com",
    "AMZN":"amazon.com","NVDA":"nvidia.com","TSLA":"tesla.com","NFLX":"netflix.com",
    "AMD":"amd.com","AVGO":"broadcom.com","TSM":"tsmc.com","ASML":"asml.com","ARM":"arm.com",
    "SMCI":"supermicro.com","INTC":"intel.com","MU":"micron.com","TER":"teradyne.com",
    "CRM":"salesforce.com","NOW":"servicenow.com","SNOW":"snowflake.com","DDOG":"datadoghq.com","MDB":"mongodb.com","PLTR":"palantir.com",
    "PANW":"paloaltonetworks.com","CRWD":"crowdstrike.com","ZS":"zscaler.com",
    "V":"visa.com","MA":"mastercard.com","PYPL":"paypal.com","SQ":"squareup.com",
    "COIN":"coinbase.com","HOOD":"robinhood.com","SOFI":"sofi.com",
    "XOM":"exxonmobil.com","CVX":"chevron.com","SLB":"slb.com","OXY":"oxy.com","COP":"conocophillips.com",
    "GLD":"spdrs.com","SLV":"ishares.com","USO":"uscfinvestments.com","DBC":"invesco.com","UNG":"uscfinvestments.com",
    "SPY":"spdrs.com","QQQ":"invesco.com","DIA":"spdrs.com","IWM":"ishares.com",
    "LMT":"lockheedmartin.com","RTX":"rtx.com","NOC":"northropgrumman.com","GD":"gd.com","BA":"boeing.com",
    "LLY":"lilly.com","REGN":"regeneron.com","MRK":"merck.com","PFE":"pfizer.com","JNJ":"jnj.com","ISRG":"intuitive.com",
    "SHOP":"shopify.com","COST":"costco.com","WMT":"walmart.com",
    "BOTZ":"globalxetfs.com","ROBO":"roboglobal.com","IRBT":"irobot.com","FANUY":"fanuc.co.jp",
    "IONQ":"ionq.com","RGTI":"rigetti.com","QBTS":"dwavesys.com","IBM":"ibm.com",
    "BABA":"alibaba.com","DIS":"thewaltdisneycompany.com","JPM":"jpmorganchase.com",
}

WATCHLIST = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","NFLX","META",
    "AMD","AVGO","TSM","ASML","ARM","SMCI","INTC","MU","TER",
    "CRM","NOW","SNOW","DDOG","MDB","PLTR","PANW","CRWD","ZS",
    "V","MA","PYPL","SQ","COIN","HOOD","SOFI",
    "XOM","CVX","SLB","OXY","COP","GLD","SLV","USO","DBC","UNG",
    "LMT","RTX","NOC","GD","BA",
    "LLY","REGN","MRK","PFE","JNJ","ISRG",
    "SHOP","COST","WMT",
    "BOTZ","ROBO","IRBT","FANUY","TER",
    "IONQ","RGTI","QBTS","IBM",
    "SPY","QQQ","DIA","IWM",
    "BABA","DIS","JPM","GOOG"
]

def have_logo(tkr: str) -> bool:
    return os.path.isfile(os.path.join(LOGO_DIR, f"{tkr}.png"))

def fetch_logo(domain: str):
    url = f"https://logo.clearbit.com/{domain}"
    try:
        r = requests.get(url, timeout=12)
        if r.status_code == 200 and r.content:
            return Image.open(io.BytesIO(r.content)).convert("RGBA")
    except Exception:
        return None
    return None

def save_png(img: Image.Image, path: str):
    img.save(path, "PNG")

def ticker_badge(tkr: str, w=512, h=256) -> Image.Image:
    img = Image.new("RGBA", (w, h), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0,0,w-1,h-1], radius=32, fill=(255,255,255,18), outline=(255,255,255,80), width=2)
    font_path = None
    for cand in [os.path.join(FONT_DIR,"Grift-Bold.ttf"), os.path.join(FONT_DIR,"Grift-Regular.ttf")]:
        if os.path.isfile(cand): font_path = cand; break
    try:
        font = ImageFont.truetype(font_path, 140) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0,0), tkr, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    x = (w - tw)//2; y = (h - th)//2
    draw.text((x,y), tkr, font=font, fill=(255,255,255,230))
    return img

def main():
    os.makedirs(LOGO_DIR, exist_ok=True)

    # mirror GOOG→GOOGL if needed
    for dst, src in MIRRORS.items():
        srcp = os.path.join(LOGO_DIR, f"{src}.png")
        dstp = os.path.join(LOGO_DIR, f"{dst}.png")
        if not os.path.isfile(dstp) and os.path.isfile(srcp):
            shutil.copy2(srcp, dstp)
            print(f"[mirror] {src}.png → {dst}.png")

    downloaded = []; badged = []; skipped = []; failed = []

    sess = requests.Session()
    sess.headers.update({"User-Agent":"TrendWatchDesk/1.0"})

    for t in WATCHLIST:
        out = os.path.join(LOGO_DIR, f"{t}.png")
        if os.path.isfile(out):
            skipped.append(t); continue

        domain = TICKER_DOMAIN.get(t)
        img = fetch_logo(domain) if domain else None
        if img is not None:
            save_png(img, out)
            downloaded.append(t)
            print(f"[ok] {t} ← {domain}")
            time.sleep(0.25)
            continue

        try:
            save_png(ticker_badge(t), out)
            badged.append(t)
            print(f"[badge] {t} (fallback)")
        except Exception:
            failed.append(t)
            print(f"[fail] {t} (no logo, badge failed)")

    print("\n=== Summary ===")
    print("Downloaded :", len(downloaded), downloaded)
    print("Badged     :", len(badged), badged)
    print("Skipped    :", len(skipped), "(already present)")
    print("Failed     :", len(failed), failed)
    if failed:
        sys.exit(2)

if __name__ == "__main__":
    main()
