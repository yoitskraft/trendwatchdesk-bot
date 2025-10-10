# controls.py
# TrendWatchDesk controls — safe overrides for main.py
# Tip: use spaces only (no tabs). UTF-8, no BOM.

SETTINGS = {
    # ---- Global scale (resizes EVERYTHING) ----
    # 1.0 = default size, <1 smaller, >1 larger
    "CHART_SCALE": 1.0,
    # Posters follow CHART_SCALE if None; set a number to override (e.g., 1.0)
    "POSTER_SCALE": None,

    # ---- Logo scaling (multiplies on top of the canvas scale) ----
    "CHART_LOGO_SCALE": 0.80,
    "POSTER_LOGO_SCALE": 0.90,

    # ---- Support zone look ----
    "SUPPORT_FILL_ALPHA": 96,
    "SUPPORT_BLUR_RADIUS": 6,
    "SUPPORT_OUTLINE_ALPHA": 140,
    "SUPPORT_MIN_PX": 26,

    # ---- Candles ----
    "CANDLE_BODY_RATIO": 0.35,
    "CANDLE_BODY_MAX": 12,
    "CANDLE_WICK_RATIO": 0.15,
    "CANDLE_UP_RGBA":  (90, 230, 150, 255),
    "CANDLE_DN_RGBA": (245, 110, 110, 255),
    "WICK_RGBA":      (245, 250, 255, 170),

    # ---- Posters behavior ----
    "POSTER_COUNT": 2,
    "POSTERS_ENABLED": True,

    # ---- Captions ----
    "CAPTION_DECIMALS_30D": 1,
}

def apply_overrides(g):
    """Apply SETTINGS into main.py globals. Safe if keys are missing."""
    S = SETTINGS

    # Scales
    g["CHART_SCALE"]  = float(S.get("CHART_SCALE",  g.get("CHART_SCALE", 1.0)))
    _poster_scale = S.get("POSTER_SCALE", None)
    g["POSTER_SCALE"] = float(_poster_scale) if _poster_scale is not None else float(g.get("CHART_SCALE", 1.0))

    # Logo scales
    g["CHART_LOGO_SCALE"]  = float(S.get("CHART_LOGO_SCALE",  g.get("CHART_LOGO_SCALE", 1.0)))
    g["POSTER_LOGO_SCALE"] = float(S.get("POSTER_LOGO_SCALE", g.get("POSTER_LOGO_SCALE", 1.0)))

    # Support zone
    g["SUPPORT_FILL_ALPHA"]     = int(S.get("SUPPORT_FILL_ALPHA",     g.get("SUPPORT_FILL_ALPHA", 96)))
    g["SUPPORT_BLUR_RADIUS"]    = int(S.get("SUPPORT_BLUR_RADIUS",    g.get("SUPPORT_BLUR_RADIUS", 6)))
    g["SUPPORT_OUTLINE_ALPHA"]  = int(S.get("SUPPORT_OUTLINE_ALPHA",  g.get("SUPPORT_OUTLINE_ALPHA", 140)))
    g["SUPPORT_MIN_PX"]         = int(S.get("SUPPORT_MIN_PX",         g.get("SUPPORT_MIN_PX", 26)))

    # Candles
    g["CANDLE_BODY_RATIO"] = float(S.get("CANDLE_BODY_RATIO", g.get("CANDLE_BODY_RATIO", 0.35)))
    g["CANDLE_BODY_MAX"]   = int(S.get("CANDLE_BODY_MAX",     g.get("CANDLE_BODY_MAX", 12)))
    g["CANDLE_WICK_RATIO"] = float(S.get("CANDLE_WICK_RATIO", g.get("CANDLE_WICK_RATIO", 0.15)))
    g["CANDLE_UP_RGBA"]    = tuple(S.get("CANDLE_UP_RGBA",    g.get("CANDLE_UP_RGBA", (90,230,150,255))))
    g["CANDLE_DN_RGBA"]    = tuple(S.get("CANDLE_DN_RGBA",    g.get("CANDLE_DN_RGBA", (245,110,110,255))))
    g["WICK_RGBA"]         = tuple(S.get("WICK_RGBA",         g.get("WICK_RGBA", (245,250,255,170))))

    # Posters
    g["POSTER_COUNT"]    = int(S.get("POSTER_COUNT",    g.get("POSTER_COUNT", 2)))
    g["POSTERS_ENABLED"] = bool(S.get("POSTERS_ENABLED", g.get("POSTERS_ENABLED", True)))

    # Captions
    g["CAPTION_DECIMALS_30D"] = int(S.get("CAPTION_DECIMALS_30D", g.get("CAPTION_DECIMALS_30D", 1)))
