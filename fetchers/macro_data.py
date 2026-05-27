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


def _compute_equity_fear_greed(hy_spread_val=None):
    """
    Multi-factor equity Fear & Greed index using publicly available data.
    Modelled on CNN F&G — four components, each scored 0-100, equally weighted.

    Components:
      1. VIX level          — market volatility (fear = high VIX)
      2. SPY momentum       — SPY vs its 125-day MA (greed = above MA)
      3. Safe-haven demand  — SPY 20-day return vs TLT (greed = stocks outperform bonds)
      4. HY spread          — junk-bond spread (greed = tight spreads, risk-on)
    """
    try:
        import yfinance as yf
        data = yf.download(["SPY", "^VIX", "TLT"], period="9mo", progress=False, auto_adjust=True)["Close"]
        if data.empty or len(data) < 130:
            return None

        spy = data["SPY"].dropna()
        vix = data["^VIX"].dropna()
        tlt = data["TLT"].dropna()

        scores = {}

        # 1. VIX: linear scale 10→100 (Extreme Greed) … 40→0 (Extreme Fear)
        vix_now = float(vix.iloc[-1])
        vix_prev_week = float(vix.iloc[-6]) if len(vix) >= 6 else vix_now
        scores["vix"] = max(0, min(100, (40 - vix_now) / 30 * 100))

        # 2. SPY momentum: % distance from 125-day MA, scaled ±10% → 0-100
        ma125 = float(spy.rolling(125).mean().iloc[-1])
        pct_from_ma = (float(spy.iloc[-1]) - ma125) / ma125 * 100
        scores["momentum"] = max(0, min(100, 50 + pct_from_ma * 5))

        # 3. Safe-haven demand: SPY 20d return minus TLT 20d return, scaled ±5% → 0-100
        if len(spy) >= 21 and len(tlt) >= 21:
            spy_20d = (float(spy.iloc[-1]) / float(spy.iloc[-21]) - 1) * 100
            tlt_20d = (float(tlt.iloc[-1]) / float(tlt.iloc[-21]) - 1) * 100
            rel = spy_20d - tlt_20d
            scores["safe_haven"] = max(0, min(100, 50 + rel * 10))

        # 4. HY spread: tight = greed (score 100 at ≤3%), wide = fear (score 0 at ≥7%)
        if hy_spread_val is not None:
            scores["hy_spread"] = max(0, min(100, (7 - hy_spread_val) / 4 * 100))

        if not scores:
            return None

        composite = round(sum(scores.values()) / len(scores))
        week_ago_vix = float(vix.iloc[-6]) if len(vix) >= 6 else vix_now
        week_ago_score = max(0, min(100, (40 - week_ago_vix) / 30 * 100))

        if composite >= 75:
            label = "Extreme Greed"
        elif composite >= 55:
            label = "Greed"
        elif composite >= 45:
            label = "Neutral"
        elif composite >= 25:
            label = "Fear"
        else:
            label = "Extreme Fear"

        trend = ("rising" if composite > week_ago_score + 5
                 else "falling" if composite < week_ago_score - 5
                 else "stable")

        return {
            "value": composite,
            "label": label,
            "week_ago": int(week_ago_score),
            "trend": trend,
            "source": f"Equity composite (VIX·Momentum·SafeHaven·HY)",
            "_components": scores,
        }
    except Exception as e:
        logger.debug(f"Equity F&G computation failed: {e}")
        return None


def get_fear_greed(hy_spread_val=None):
    """
    Multi-factor equity Fear & Greed index (VIX, SPY momentum, safe-haven, HY spread).
    Entirely stock-market based — no crypto data used.
    Returns dict with value (0=Extreme Fear, 100=Extreme Greed), label, week trend.
    """
    return _compute_equity_fear_greed(hy_spread_val=hy_spread_val)


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

    # ── Fed Funds Rate (DFF = daily, falls back to monthly FEDFUNDS) ─────────
    fedfunds = _fetch_fred_series("DFF", limit=10) or _fetch_fred_series("FEDFUNDS", limit=4)
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
    hy_val = macro.get("hy_spread", {}).get("value")
    fg = get_fear_greed(hy_spread_val=hy_val)
    if fg:
        macro["fear_greed"] = fg
    breadth = get_small_cap_breadth()
    if breadth:
        macro["small_cap_breadth"] = breadth
    return macro
