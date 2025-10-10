# controls.py
"""
TrendWatchDesk controls — safe config overrides for charts & posters.
"""

SETTINGS = {
    # scale both W & H together
    "CHART_SCALE": 0.9,        # 1.0 = normal, <1 smaller, >1 larger
    "POSTER_SCALE": None,      # None = follow CHART_SCALE, or set e.g. 1.0

    # fine-tune logo sizes independently
    "CHART_LOGO_SCALE": 0.8,
    "POSTER_LOGO_SCALE": 0.9,
}
    # ---- Support zone ----
    "SUPPORT_FILL_ALPHA": 110,   # whiter/stronger
    "SUPPORT_BLUR_RADIUS": 5,    # crisper
    "SUPPORT_OUTLINE_ALPHA": 160,
    "SUPPORT_MIN_PX": 28,

    # ---- Candlesticks ----
    "CANDLE_BODY_RATIO": 0.35,
    "CANDLE_BODY_MAX": 12,
    "CANDLE_WICK_RATIO": 0.15,
    "CANDLE_UP_RGBA":  (90, 230, 150, 255),
    "CANDLE_DN_RGBA": (245, 110, 110, 255),
    "WICK_RGBA":      (245, 250, 255, 170),

    # ---- Watchlist extensions ----
    "WATCHLIST_EXTEND": ["SOFI", "IONQ", "ISRG"],

    # ---- Pick distribution ----
    "PICK_COUNTS": {
        "AI": 2,
        "MAG7": 2,
        "Semis": 1,
        "Other": 1,
    },

    # ---- Posters ----
    "POSTER_COUNT": 2,
    "POSTERS_ENABLED": True,

    # ---- Captions ----
    "CAPTION_DECIMALS_30D": 1,
}

def apply_overrides(g):
    """Apply SETTINGS into main.py globals safely."""
    S = SETTINGS
    g["CHART_W"] = S.get("CHART_WIDTH", g.get("CHART_W", 1080))
    g["CHART_H"] = S.get("CHART_HEIGHT", g.get("CHART_H", 720))

    # logos
    g["CHART_LOGO_SCALE"]  = S.get("CHART_LOGO_SCALE", g.get("CHART_LOGO_SCALE", 1.0))
    g["POSTER_LOGO_SCALE"] = S.get("POSTER_LOGO_SCALE", g.get("POSTER_LOGO_SCALE", 1.0))

    # support zone
    g["SUPPORT_FILL_ALPHA"]   = S.get("SUPPORT_FILL_ALPHA",   g.get("SUPPORT_FILL_ALPHA", 96))
    g["SUPPORT_BLUR_RADIUS"]  = S.get("SUPPORT_BLUR_RADIUS",  g.get("SUPPORT_BLUR_RADIUS", 6))
    g["SUPPORT_OUTLINE_ALPHA"]= S.get("SUPPORT_OUTLINE_ALPHA",g.get("SUPPORT_OUTLINE_ALPHA", 140))
    g["SUPPORT_MIN_PX"]       = S.get("SUPPORT_MIN_PX",       g.get("SUPPORT_MIN_PX", 26))

    # candles
    g["CANDLE_BODY_RATIO"] = S.get("CANDLE_BODY_RATIO", g.get("CANDLE_BODY_RATIO", 0.35))
    g["CANDLE_BODY_MAX"]   = S.get("CANDLE_BODY_MAX",   g.get("CANDLE_BODY_MAX", 12))
    g["CANDLE_WICK_RATIO"] = S.get("CANDLE_WICK_RATIO", g.get("CANDLE_WICK_RATIO", 0.15))
    g["CANDLE_UP_RGBA"]    = S.get("CANDLE_UP_RGBA",    g.get("CANDLE_UP_RGBA", (90,230,150,255)))
    g["CANDLE_DN_RGBA"]    = S.get("CANDLE_DN_RGBA",    g.get("CANDLE_DN_RGBA", (245,110,110,255)))
    g["WICK_RGBA"]         = S.get("WICK_RGBA",         g.get("WICK_RGBA", (245,250,255,170)))

    # watchlist
    extra = S.get("WATCHLIST_EXTEND", [])
    g["WATCHLIST_EXTRA"] = list({*(g.get("WATCHLIST_EXTRA", [])), *extra})

    # picks
    g["PICK_COUNTS"] = {**g.get("PICK_COUNTS", {"AI":2,"MAG7":2,"Semis":1,"Other":1}),
                        **S.get("PICK_COUNTS", {})}

    # posters
    g["POSTER_COUNT"]    = S.get("POSTER_COUNT",    g.get("POSTER_COUNT", 2))
    g["POSTERS_ENABLED"] = S.get("POSTERS_ENABLED", g.get("POSTERS_ENABLED", True))

    # captions
    g["CAPTION_DECIMALS_30D"] = S.get("CAPTION_DECIMALS_30D", g.get("CAPTION_DECIMALS_30D", 1))
