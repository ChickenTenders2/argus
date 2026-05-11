import os
import requests
import logging

logger = logging.getLogger("Argus.Macro")

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


def _fetch_fred_series(series_id, limit=5):
    """Fetch the latest N observations for a FRED series. Returns list newest-first."""
    fred_api_key = os.environ.get("FRED_API_KEY", "")
    if not fred_api_key:
        return None
    try:
        r = requests.get(
            _FRED_BASE,
            params={
                "series_id": series_id,
                "api_key": fred_api_key,
                "file_type": "json",
                "limit": limit,
                "sort_order": "desc",
            },
            timeout=10,
        )
        if r.status_code != 200:
            logger.warning(f"FRED {series_id} returned {r.status_code}")
            return None
        obs = r.json().get("observations", [])
        vals = [float(d["value"]) for d in obs if d.get("value") not in (".", None, "")]
        return vals if vals else None
    except Exception as e:
        logger.warning(f"FRED fetch failed for {series_id}: {e}")
        return None


def get_fear_greed():
    """
    Fetch Alternative.me Fear & Greed Index (no API key needed).
    Returns dict with value (0=Extreme Fear, 100=Extreme Greed), label, week trend.
    """
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=7", timeout=8)
        if r.status_code != 200:
            return None
        data = r.json().get("data", [])
        if not data:
            return None
        latest = data[0]
        week_ago = data[-1] if len(data) >= 7 else data[0]
        val = int(latest["value"])
        val_prev = int(week_ago["value"])
        return {
            "value": val,
            "label": latest["value_classification"],
            "week_ago": val_prev,
            "trend": "rising" if val > val_prev + 5 else ("falling" if val < val_prev - 5 else "stable"),
        }
    except Exception as e:
        logger.warning(f"Fear & Greed fetch failed: {e}")
        return None


def get_fred_macro():
    """
    Fetch three FRED macro indicators:
      - T10Y2Y  : 10Y-2Y Treasury spread (yield curve). Negative = inverted.
      - CPIAUCSL: CPI (monthly). Compute annualised 3-month % change.
      - FEDFUNDS: Effective Federal Funds Rate (monthly).

    Returns a dict of sub-dicts, each containing 'value' and 'signal'.
    Returns empty dict silently if FRED_API_KEY is not set.
    """
    result = {}

    # ── Yield Curve (T10Y2Y) ──────────────────────────────
    t10y2y = _fetch_fred_series("T10Y2Y", limit=3)
    if t10y2y:
        val = t10y2y[0]
        result["yield_curve"] = {
            "value": round(val, 2),
            "inverted": val < 0,
            "signal": "⚠️ Inverted" if val < 0 else ("⚡ Flat" if val < 0.5 else "✅ Normal"),
        }

    # ── CPI (CPIAUCSL) ────────────────────────────────────
    cpi = _fetch_fred_series("CPIAUCSL", limit=4)
    if cpi and len(cpi) >= 2:
        n = len(cpi) - 1
        annualised = ((cpi[0] / cpi[-1]) ** (12 / n) - 1) * 100
        result["cpi"] = {
            "value": round(annualised, 1),
            "accelerating": annualised > 3.5,
            "signal": "🔴 Hot" if annualised > 3.5 else ("🟡 Elevated" if annualised > 2.5 else "✅ Contained"),
        }

    # ── Fed Funds Rate (FEDFUNDS) ─────────────────────────
    fedfunds = _fetch_fred_series("FEDFUNDS", limit=2)
    if fedfunds:
        val = fedfunds[0]
        prev = fedfunds[1] if len(fedfunds) >= 2 else val
        result["fed_funds"] = {
            "value": round(val, 2),
            "rising": val > prev,
            "signal": "🔴 Tight" if val >= 5.0 else ("🟡 Elevated" if val >= 3.0 else "✅ Accommodative"),
        }

    return result


def build_macro_context():
    """
    Combine FRED macro indicators with the Fear & Greed index into one dict.
    All fields are optional — callers must use .get() with defaults.
    """
    macro = get_fred_macro()
    fg = get_fear_greed()
    if fg:
        macro["fear_greed"] = fg
    return macro
