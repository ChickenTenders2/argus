import requests
import os
from datetime import datetime, timedelta

FMP_API_KEY = os.environ.get("FMP_API_KEY", "")
BASE = "https://financialmodelingprep.com/api"
HIGH_CONVICTION_THRESHOLD = 80


# ── Safe Fetch ───────────────────────────────────────────
def safe_get(url):
    """Returns None gracefully on paywalled or failed responses."""
    try:
        r = requests.get(url, timeout=10)
        if r.status_code in [401, 402, 403]:
            return None
        if r.status_code != 200:
            return None
        data = r.json()
        return data if data else None
    except Exception:
        return None


# ── Fetch All Endpoints ──────────────────────────────────
def get_fmp_data(ticker):
    return {
        "profile":       safe_get(f"{BASE}/v3/profile/{ticker}?apikey={FMP_API_KEY}"),
        "income":        safe_get(f"{BASE}/v3/income-statement/{ticker}?limit=2&apikey={FMP_API_KEY}"),
        "cashflow":      safe_get(f"{BASE}/v3/cash-flow-statement/{ticker}?limit=1&apikey={FMP_API_KEY}"),
        "key_metrics":   safe_get(f"{BASE}/v3/key-metrics-ttm/{ticker}?apikey={FMP_API_KEY}"),
        "insiders":      safe_get(f"{BASE}/v4/insider-trading?symbol={ticker}&limit=20&apikey={FMP_API_KEY}"),
        "institutions":  safe_get(f"{BASE}/v3/institutional-holder/{ticker}?apikey={FMP_API_KEY}"),
        "earnings_cal":  safe_get(f"{BASE}/v3/historical/earning_calendar/{ticker}?apikey={FMP_API_KEY}"),
        "analyst_grade": safe_get(f"{BASE}/v3/grade/{ticker}?limit=5&apikey={FMP_API_KEY}"),
    }


# ── Parse Into Argus Fields ──────────────────────────────
def parse_fmp_data(ticker, data):
    result = {"ticker": ticker}

    # Profile
    profile = (data["profile"] or [{}])[0]
    market_cap = profile.get("mktCap", 0) or 0
    result["market_cap_m"] = f"${market_cap / 1e6:,.0f}M" if market_cap else "N/A"

    # Income Statement
    income = data["income"] or []
    ebitda_growth_pct = None
    if len(income) >= 2:
        rev_curr = income[0].get("revenue", 0) or 0
        rev_prev = income[1].get("revenue", 1) or 1
        result["rev_growth"] = f"{((rev_curr - rev_prev) / abs(rev_prev)) * 100:+.1f}%"

        ebitda_curr = income[0].get("ebitda", 0) or 0
        ebitda_prev = income[1].get("ebitda", 1) or 1
        ebitda_growth_pct = ((ebitda_curr - ebitda_prev) / abs(ebitda_prev)) * 100
        result["ebitda_growth"] = f"{ebitda_growth_pct:+.1f}%"

        gm = income[0].get("grossProfitRatio", None)
        result["gross_margin"] = f"{gm * 100:.1f}%" if gm else "N/A"
    else:
        result["rev_growth"] = "N/A"
        result["ebitda_growth"] = "N/A"
        result["gross_margin"] = "N/A"

    # Cash Flow — FCF Yield (strongest single multibagger predictor)
    cashflow = (data["cashflow"] or [{}])[0]
    fcf = cashflow.get("freeCashFlow", None)
    if fcf is not None and market_cap > 0:
        fcf_yield = round((fcf / market_cap) * 100, 2)
        result["fcf_yield"] = f"{fcf_yield:.2f}%"
        result["fcf_positive"] = fcf > 0
    else:
        result["fcf_yield"] = "N/A"
        result["fcf_positive"] = None

    # Key Metrics TTM + Asset vs EBITDA disqualifier check
    km = (data["key_metrics"] or [{}])[0]
    asset_growth = km.get("assetGrowth", None)
    if asset_growth is not None and ebitda_growth_pct is not None:
        ag_pct = asset_growth * 100
        flag = "⚠️ DISQUALIFIER" if ag_pct > ebitda_growth_pct else "✅ Healthy"
        result["asset_vs_ebitda"] = f"{flag} — Asset growth {ag_pct:.1f}% vs EBITDA growth {ebitda_growth_pct:.1f}%"
    else:
        result["asset_vs_ebitda"] = "N/A"

    # Insider Buying — open-market purchases only, last 90 days
    cutoff = datetime.now() - timedelta(days=90)
    raw_insiders = data["insiders"]
    if isinstance(raw_insiders, dict):
        transactions = raw_insiders.get("data", [])
    elif isinstance(raw_insiders, list):
        transactions = raw_insiders
    else:
        transactions = []

    buys = []
    for t in transactions:
        if t.get("transactionType") == "P-Purchase":
            try:
                if datetime.strptime(t["transactionDate"], "%Y-%m-%d") >= cutoff:
                    buys.append(
                        f"{t.get('reportingName', 'Unknown')} "
                        f"({t.get('typeOfOwner', '?')}) — "
                        f"{t.get('securitiesTransacted', 0):,} shares on {t['transactionDate']}"
                    )
            except Exception:
                pass

    if buys:
        result["insider_buys"] = f"✅ {len(buys)} open-market buy(s):\n      " + "\n      ".join(buys[:3])
    else:
        result["insider_buys"] = "❌ No open-market buys in last 90 days"

    # Institutional Ownership
    institutions = data["institutions"]
    if institutions:
        total_inst = sum(h.get("shares", 0) for h in institutions)
        shares_out = profile.get("sharesOutstanding", 0) or 0
        if shares_out > 0:
            inst_pct = round((total_inst / shares_out) * 100, 1)
            flag = "✅" if inst_pct < 40 else ("⚠️" if inst_pct < 60 else "🔴")
            result["inst_ownership"] = f"{flag} {inst_pct}% institutional"
        else:
            result["inst_ownership"] = "N/A"
    else:
        result["inst_ownership"] = "⚠️ Not available on free tier"

    # Next Earnings Date
    earnings = data["earnings_cal"] or []
    today_str = datetime.now().strftime("%Y-%m-%d")
    future = [e for e in earnings if e.get("date", "") >= today_str]
    result["next_earnings"] = future[0].get("date", "N/A") if future else "N/A"

    # Analyst Grade Changes
    grades = data["analyst_grade"] or []
    if grades:
        grade_lines = [
            f"{g.get('gradingCompany', '?')} → {g.get('newGrade', '?')} ({g.get('date', '?')})"
            for g in grades[:3]
        ]
        result["analyst_grades"] = "\n      ".join(grade_lines)
    else:
        result["analyst_grades"] = "N/A"

    return result


# ── Format Telegram Block ────────────────────────────────
def format_fmp_block(parsed):
    fcf_flag = (
        "✅" if parsed.get("fcf_positive") is True
        else "❌" if parsed.get("fcf_positive") is False
        else "❓"
    )
    return (
        f"\n📡 *FMP Enrichment — {parsed['ticker']}*\n"
        f"{'─' * 30}\n"
        f"📈 Revenue Growth: {parsed['rev_growth']}\n"
        f"📊 Gross Margin: {parsed['gross_margin']}\n"
        f"💰 FCF Yield: {parsed['fcf_yield']} {fcf_flag}\n"
        f"📉 EBITDA Growth: {parsed['ebitda_growth']}\n"
        f"🔬 Asset vs EBITDA: {parsed['asset_vs_ebitda']}\n"
        f"🏛 Institutions: {parsed['inst_ownership']}\n"
        f"🧠 Insider Buys (90d):\n      {parsed['insider_buys']}\n"
        f"📅 Next Earnings: {parsed['next_earnings']}\n"
        f"🎯 Analyst Grades:\n      {parsed['analyst_grades']}\n"
        f"{'─' * 30}\n"
    )


# ── Main Entry Point ─────────────────────────────────────
def run_fmp_enrichment(results, send_telegram_fn):
    """
    Call from argus.py main() after scoring is complete.
    Only fetches data for HIGH CONVICTION tickers (score >= 80).
    Sends a separate Telegram message per qualifying ticker.
    """
    if not FMP_API_KEY:
        print("⚠️ FMP_API_KEY not set — skipping FMP enrichment")
        return

    high_conviction = [r for r in results if r.get("score", 0) >= HIGH_CONVICTION_THRESHOLD]

    if not high_conviction:
        print("ℹ️ No HIGH CONVICTION picks today — FMP enrichment skipped")
        return

    print(f"📡 FMP enrichment running for {len(high_conviction)} HIGH CONVICTION ticker(s)...")

    for pick in high_conviction:
        ticker = pick["ticker"]
        print(f"  → Fetching {ticker}...")
        raw = get_fmp_data(ticker)
        parsed = parse_fmp_data(ticker, raw)
        send_telegram_fn(format_fmp_block(parsed))
        print(f"  ✅ {ticker} sent")
