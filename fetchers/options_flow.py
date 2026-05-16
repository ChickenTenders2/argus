"""Options unusual activity feed — provider-abstracted.

Set OPTIONS_FLOW_PROVIDER env var to:
  barchart_free  (default) — scrape Barchart public page
  market_chameleon — scrape Market Chameleon IV rank page
  unusual_whales — use UW API (requires UW_API_KEY env var)

Returns dicts with keys:
  vol_oi_ratio: float — option volume / open interest ratio (higher = more unusual)
  call_bias: float — call volume / (call + put volume), 0.5 = neutral
  iv_rank: float or None — IV rank 0–100 (higher = historically elevated IV)
  bullish: bool — True if call_bias > 0.6 and vol_oi_ratio > 2
  source: str
"""
import os
import logging
import time

logger = logging.getLogger("Argus.OptionsFlow")

_CACHE: dict = {}
_CACHE_TTL = 3600  # 1 hour


def _cached(key, fn):
    now = time.time()
    if key in _CACHE and now - _CACHE[key]["ts"] < _CACHE_TTL:
        return _CACHE[key]["data"]
    result = fn()
    _CACHE[key] = {"data": result, "ts": now}
    return result


def _fetch_barchart_free(ticker: str) -> dict:
    """Scrape Barchart unusual options page for a specific ticker."""
    try:
        import urllib.request
        url = f"https://www.barchart.com/stocks/quotes/{ticker}/options"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml",
        })
        resp = urllib.request.urlopen(req, timeout=10).read().decode("utf-8", errors="ignore")
        # Look for IV rank in page (Barchart shows "IV Rank: NN%")
        iv_rank = None
        if "IV Rank" in resp:
            try:
                idx = resp.index("IV Rank")
                snippet = resp[idx:idx+60]
                import re
                m = re.search(r"(\d+\.?\d*)%", snippet)
                if m:
                    iv_rank = float(m.group(1))
            except Exception:
                pass
        # Can't reliably get vol/OI from Barchart without auth — return what we have
        time.sleep(1)  # polite rate limit
        return {
            "vol_oi_ratio": None,
            "call_bias": None,
            "iv_rank": iv_rank,
            "bullish": None,
            "source": "barchart_free",
        }
    except Exception as e:
        logger.debug(f"Barchart free fetch failed for {ticker}: {e}")
        return {}


def _fetch_market_chameleon(ticker: str) -> dict:
    """Fetch IV rank from Market Chameleon (free tier)."""
    try:
        import urllib.request, re
        url = f"https://marketchameleon.com/Overview/{ticker}/IV/"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        })
        resp = urllib.request.urlopen(req, timeout=10).read().decode("utf-8", errors="ignore")
        iv_rank = None
        # Market Chameleon shows IV Rank as "IV Rank: NN" or similar
        m = re.search(r"IV Rank[:\s]+(\d+\.?\d*)", resp)
        if m:
            iv_rank = float(m.group(1))
        bullish = iv_rank is not None and iv_rank > 70
        time.sleep(1)
        return {
            "vol_oi_ratio": None,
            "call_bias": None,
            "iv_rank": iv_rank,
            "bullish": bullish,
            "source": "market_chameleon",
        }
    except Exception as e:
        logger.debug(f"Market Chameleon fetch failed for {ticker}: {e}")
        return {}


def _fetch_unusual_whales(ticker: str) -> dict:
    """Fetch from Unusual Whales API (requires UW_API_KEY env var)."""
    uw_key = os.environ.get("UW_API_KEY", "")
    if not uw_key:
        return {}
    try:
        import urllib.request, json
        url = f"https://api.unusualwhales.com/api/stock/{ticker}/option-chains/flow"
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {uw_key}",
            "Accept": "application/json",
        })
        resp = json.loads(urllib.request.urlopen(req, timeout=10).read())
        data = resp.get("data", {})
        call_vol = float(data.get("call_volume", 0) or 0)
        put_vol  = float(data.get("put_volume", 0) or 0)
        oi       = float(data.get("open_interest", 1) or 1)
        total_vol = call_vol + put_vol
        call_bias = call_vol / total_vol if total_vol > 0 else 0.5
        vol_oi    = total_vol / oi if oi > 0 else 0
        bullish   = call_bias > 0.60 and vol_oi > 2
        return {
            "vol_oi_ratio": round(vol_oi, 2),
            "call_bias":    round(call_bias, 3),
            "iv_rank":      data.get("iv_rank"),
            "bullish":      bullish,
            "source":       "unusual_whales",
        }
    except Exception as e:
        logger.debug(f"Unusual Whales fetch failed for {ticker}: {e}")
        return {}


def get_options_flow(ticker: str) -> dict:
    """Fetch options flow using the configured provider. Returns {} on failure."""
    provider = os.environ.get("OPTIONS_FLOW_PROVIDER", "barchart_free").lower()

    def _fetch():
        if provider == "unusual_whales":
            return _fetch_unusual_whales(ticker)
        elif provider == "market_chameleon":
            return _fetch_market_chameleon(ticker)
        else:
            return _fetch_barchart_free(ticker)

    try:
        return _cached(f"{provider}:{ticker}", _fetch)
    except Exception as e:
        logger.debug(f"get_options_flow({ticker}) failed: {e}")
        return {}
