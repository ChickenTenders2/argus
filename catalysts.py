"""Catalyst score aggregator — combines EDGAR cluster buys, 8-K filings, and options flow.

compute_catalyst_score(ticker, fmp_context=None) -> (score: int, reasons: list[str])
  score: 0–15 pts
  reasons: human-readable catalyst tags (e.g. ["Insider cluster (3x)", "8-K: Material Agreement"])
"""
import logging

logger = logging.getLogger("Argus.Catalysts")

_CATALYST_CACHE: dict = {}
import time as _time
_CATALYST_TTL = 7200  # 2 hours


def compute_catalyst_score(ticker: str, fmp_context: dict = None) -> tuple:
    """Aggregate catalyst signals into a 0–15 point score with reasons."""
    cache_key = ticker
    _now = _time.time()
    if cache_key in _CATALYST_CACHE and _now - _CATALYST_CACHE[cache_key]["ts"] < _CATALYST_TTL:
        cached = _CATALYST_CACHE[cache_key]
        return cached["score"], cached["reasons"]

    total, reasons = 0, []

    # 1. EDGAR insider cluster buys
    try:
        from edgar_fetch import get_insider_cluster_score
        iscore, ireasons = get_insider_cluster_score(ticker)
        total += iscore
        reasons.extend(ireasons)
    except Exception as e:
        logger.debug(f"{ticker}: insider cluster score failed — {e}")

    # 2. Form 8-K catalyst
    try:
        from edgar_fetch import get_8k_catalyst_score
        kscore, kreasons = get_8k_catalyst_score(ticker)
        total += kscore
        reasons.extend(kreasons)
    except Exception as e:
        logger.debug(f"{ticker}: 8-K catalyst score failed — {e}")

    # 3. Options unusual activity
    try:
        from options_flow import get_options_flow
        flow = get_options_flow(ticker)
        if flow.get("bullish"):
            iv_rank = flow.get("iv_rank")
            if iv_rank and iv_rank > 70:
                total += 3
                reasons.append(f"Options: bullish flow (IV rank {iv_rank:.0f})")
            else:
                total += 2
                reasons.append("Options: bullish call flow")
        elif flow.get("iv_rank") and flow["iv_rank"] > 80:
            total += 1
            reasons.append(f"Options: high IV rank ({flow['iv_rank']:.0f})")
    except Exception as e:
        logger.debug(f"{ticker}: options flow score failed — {e}")

    total = min(15, total)

    _CATALYST_CACHE[cache_key] = {"score": total, "reasons": reasons, "ts": _now}
    return total, reasons
