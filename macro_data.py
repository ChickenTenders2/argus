import os
import requests
import logging

logger = logging.getLogger("Argus.Macro")

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


def _fetch_fred_series(series_id, limit=10):
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


def _get_yield_curve_yfinance():
    """
    Fallback: compute 10Y-3M Treasury spread from yfinance (^TNX minus ^IRX).
    Used when FRED_API_KEY is not set.
    """
    try:
        import yfinance as yf
        tnx = yf.Ticker("^TNX").history(period="5d")["Close"].dropna()
        irx = yf.Ticker("^IRX").history(period="5d")["Close"].dropna()
        if tnx.empty or irx.empty:
            return None
        spread = round(float(tnx.iloc[-1]) - float(irx.iloc[-1]), 2)
        return {
            "value": spread,
            "inverted": spread < 0,
            "signal": "⚠️ Inverted" if spread < 0 else ("⚡ Flat" if spread < 0.5 else "✅ Normal"),
            "source": "yfinance (10Y-3M)",
        }
    except Exception as e:
        logger.debug(f"yfinance yield curve fallback failed: {e}")
        return None


def _get_fear_greed_vix_fallback():
    """
    Fallback: derive a Fear & Greed proxy from the current VIX level via yfinance.
    Used when alternative.me is unavailable.
    """
    try:
        import yfinance as yf
        vix_hist = yf.Ticker("^VIX").history(period="5d")["Close"].dropna()
        if vix_hist.empty:
            return None
        vix = float(vix_hist.iloc[-1])
        vix_prev = float(vix_hist.iloc[-2]) if len(vix_hist) >= 2 else vix
        if vix >= 40:
            val, label = 5, "Extreme Fear"
        elif vix >= 30:
            val, label = 20, "Fear"
        elif vix >= 22:
            val, label = 35, "Fear"
        elif vix >= 18:
            val, label = 50, "Neutral"
        elif vix >= 14:
            val, label = 65, "Greed"
        else:
            val, label = 82, "Extreme Greed"
        trend = "rising" if vix > vix_prev * 1.05 else ("falling" if vix < vix_prev * 0.95 else "stable")
        return {
            "value": val,
            "label": label,
            "week_ago": val,
            "trend": trend,
            "source": f"VIX-derived ({vix:.1f})",
        }
    except Exception as e:
        logger.debug(f"VIX fear/greed fallback failed: {e}")
        return None


def get_fear_greed():
    """
    Fetch Fear & Greed Index. Tries alternative.me first; falls back to VIX proxy.
    Returns dict with value (0=Extreme Fear, 100=Extreme Greed), label, week trend.
    """
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=7", timeout=8)
        if r.status_code == 200:
            data = r.json().get("data", [])
            if data:
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
        logger.debug(f"Fear & Greed primary fetch failed: {e}")

    logger.info("Fear & Greed: falling back to VIX-derived proxy")
    return _get_fear_greed_vix_fallback()


def get_fred_macro():
    """
    Fetch three FRED macro indicators:
      - T10Y2Y  : 10Y-2Y Treasury spread (yield curve). Negative = inverted.
      - CPIAUCSL: CPI (monthly). Compute annualised 3-month % change.
      - FEDFUNDS: Effective Federal Funds Rate (monthly).

    Falls back to yfinance for yield curve when FRED_API_KEY is not set.
    Returns a dict of sub-dicts, each containing 'value' and 'signal'.
    """
    result = {}

    # ── Yield Curve (T10Y2Y via FRED, or 10Y-3M via yfinance) ────────────────
    t10y2y = _fetch_fred_series("T10Y2Y", limit=10)
    if t10y2y:
        val = t10y2y[0]
        result["yield_curve"] = {
            "value": round(val, 2),
            "inverted": val < 0,
            "signal": "⚠️ Inverted" if val < 0 else ("⚡ Flat" if val < 0.5 else "✅ Normal"),
        }
    else:
        yc_fallback = _get_yield_curve_yfinance()
        if yc_fallback:
            result["yield_curve"] = yc_fallback

    # ── CPI (CPIAUCSL) ────────────────────────────────────
    cpi = _fetch_fred_series("CPIAUCSL", limit=6)
    if cpi and len(cpi) >= 2:
        n = len(cpi) - 1
        annualised = ((cpi[0] / cpi[-1]) ** (12 / n) - 1) * 100
        result["cpi"] = {
            "value": round(annualised, 1),
            "accelerating": annualised > 3.5,
            "signal": "🔴 Hot" if annualised > 3.5 else ("🟡 Elevated" if annualised > 2.5 else "✅ Contained"),
        }

    # ── Fed Funds Rate (FEDFUNDS) ─────────────────────────
    fedfunds = _fetch_fred_series("FEDFUNDS", limit=4)
    if fedfunds:
        val = fedfunds[0]
        prev = fedfunds[1] if len(fedfunds) >= 2 else val
        result["fed_funds"] = {
            "value": round(val, 2),
            "rising": val > prev,
            "signal": "🔴 Tight" if val >= 5.0 else ("🟡 Elevated" if val >= 3.0 else "✅ Accommodative"),
        }

    # ── HY Credit Spread (BAMLH0A0HYM2) ──────────────────
    hy = _fetch_fred_series("BAMLH0A0HYM2", limit=25)
    if hy and len(hy) >= 2:
        val_now  = hy[0]
        val_20d  = hy[min(20, len(hy) - 1)]
        widening = val_now - val_20d  # positive = spread widened (risk-off)
        result["hy_spread"] = {
            "value":    round(val_now, 2),
            "widening_20d": round(widening, 2),
            "widening": widening > 0.50,
            "signal":   "🔴 Widening" if widening > 0.50 else ("🟡 Stable" if abs(widening) <= 0.50 else "✅ Tightening"),
        }

    return result


def get_small_cap_breadth():
    """
    Fetch IWM/QQQ relative strength and IWM realized volatility vs SPY.
    Returns dict with:
      - iwm_qqq_rs_20d: IWM 20-day return minus QQQ 20-day return (positive = small-cap leading)
      - iwm_vol_ratio: IWM 30-day realized vol / SPY 30-day realized vol (>1.5 = small-cap stress)
      - small_cap_leading: bool
      - small_cap_stress: bool
    """
    try:
        import yfinance as yf
        import numpy as np
        data = yf.download(["IWM", "QQQ", "SPY"], period="3mo", progress=False)["Close"]
        if data.empty or len(data) < 22:
            return {}
        iwm = data["IWM"].dropna()
        qqq = data["QQQ"].dropna()
        spy = data["SPY"].dropna()

        iwm_20d = float((iwm.iloc[-1] / iwm.iloc[-21] - 1)) if len(iwm) >= 21 else 0.0
        qqq_20d = float((qqq.iloc[-1] / qqq.iloc[-21] - 1)) if len(qqq) >= 21 else 0.0
        rs_20d  = round(iwm_20d - qqq_20d, 4)

        iwm_vol  = float(iwm.pct_change().dropna().tail(30).std() * (252 ** 0.5)) if len(iwm) >= 31 else 0.0
        spy_vol  = float(spy.pct_change().dropna().tail(30).std() * (252 ** 0.5)) if len(spy) >= 31 else 0.0
        vol_ratio = round(iwm_vol / spy_vol, 2) if spy_vol > 0 else 1.0

        return {
            "iwm_qqq_rs_20d":   rs_20d,
            "iwm_vol_ratio":    vol_ratio,
            "small_cap_leading": rs_20d > 0.03,
            "small_cap_stress":  vol_ratio > 1.5,
        }
    except Exception as e:
        logger.warning(f"get_small_cap_breadth failed: {e}")
        return {}


def build_macro_context():
    """
    Combine FRED macro indicators, Fear & Greed, and small-cap breadth into one dict.
    All fields are optional — callers must use .get() with defaults.
    """
    macro = get_fred_macro()
    fg = get_fear_greed()
    if fg:
        macro["fear_greed"] = fg
    breadth = get_small_cap_breadth()
    if breadth:
        macro["small_cap_breadth"] = breadth
    return macro
