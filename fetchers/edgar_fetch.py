import requests
import logging
import json
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

# ── Finnhub rate-limit guard ──────────────────────────────
# Finnhub free tier: 60 calls/minute. A lightweight token bucket shared across
# all threads prevents bursts that would cause 429 errors on full scans.
import threading as _threading
_FINNHUB_LOCK = _threading.Lock()
_FINNHUB_LAST_CALL: float = 0.0
_FINNHUB_MIN_GAP = 1.1  # seconds between calls → ~54 calls/min, safely under 60

logger = logging.getLogger("Argus.EDGAR")

# SEC EDGAR requires User-Agent in the format: "App Name contact@email.com"
# A malformed email (e.g. argus@local) causes 403 on all EDGAR endpoints.
_EDGAR_AGENT = "Argus Investment Workstation contact@argus-scanner.app"
_EDGAR_HEADERS = {"User-Agent": _EDGAR_AGENT}
_SEC_HEADERS   = {"User-Agent": _EDGAR_AGENT}

_cik_map: dict = {}
_CIK_CACHE_FILE = os.path.join("data", "edgar_cik_cache.json")
_CIK_CACHE_MAX_AGE_H = 24   # refresh the file cache once per day

# In-process cache for CIK submissions JSON — shared by insider + 8-K fetchers
# so the same endpoint is only hit once per ticker per scan run (TTL 5 min).
_submissions_cache: dict = {}
_SUBMISSIONS_TTL = 300


def _get_submissions(cik: str) -> dict:
    """Fetch (or return cached) the SEC submissions JSON for a CIK."""
    now = time.time()
    if cik in _submissions_cache and now - _submissions_cache[cik]["ts"] < _SUBMISSIONS_TTL:
        return _submissions_cache[cik]["data"]
    try:
        r = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            headers=_EDGAR_HEADERS,
            timeout=12,
        )
        if r.status_code != 200:
            logger.warning(f"EDGAR submissions CIK{cik}: HTTP {r.status_code}")
            return {}
        data = r.json()
        _submissions_cache[cik] = {"data": data, "ts": now}
        return data
    except Exception as e:
        logger.warning(f"EDGAR submissions fetch for CIK{cik} failed: {e}")
        return {}

HIGH_CONVICTION_THRESHOLD = 80


# ── CIK Resolution ───────────────────────────────────────
def _load_cik_map():
    """Load the SEC company_tickers CIK map with a file cache + retry.

    Priority:
      1. In-process memory dict (fastest — survives within a session)
      2. File cache at data/edgar_cik_cache.json if < 24 h old
      3. Live fetch from sec.gov/files/company_tickers.json (with 2 retries)
    """
    global _cik_map
    if _cik_map:
        return _cik_map

    # ── Try file cache first ──────────────────────────────
    try:
        if os.path.exists(_CIK_CACHE_FILE):
            age_h = (time.time() - os.path.getmtime(_CIK_CACHE_FILE)) / 3600
            if age_h < _CIK_CACHE_MAX_AGE_H:
                with open(_CIK_CACHE_FILE, "r") as f:
                    _cik_map = json.load(f)
                if _cik_map:
                    logger.info(f"EDGAR CIK map loaded from file cache ({len(_cik_map)} tickers, {age_h:.1f}h old)")
                    return _cik_map
    except Exception as _ce:
        logger.debug(f"CIK file cache read failed: {_ce}")

    # ── Live fetch with retry ─────────────────────────────
    url = "https://www.sec.gov/files/company_tickers.json"
    for attempt in range(3):
        try:
            r = requests.get(url, headers=_SEC_HEADERS, timeout=15)
            if r.status_code == 200:
                for entry in r.json().values():
                    ticker = entry.get("ticker", "").upper()
                    cik = str(entry.get("cik_str", "")).zfill(10)
                    if ticker:
                        _cik_map[ticker] = cik
                logger.info(f"EDGAR CIK map fetched from SEC ({len(_cik_map)} tickers)")
                # Save to file cache
                try:
                    os.makedirs("data", exist_ok=True)
                    with open(_CIK_CACHE_FILE, "w") as f:
                        json.dump(_cik_map, f)
                except Exception as _we:
                    logger.debug(f"CIK file cache write failed: {_we}")
                return _cik_map
            elif r.status_code in (403, 429):
                wait = 2 ** attempt
                logger.warning(f"EDGAR CIK fetch {r.status_code} (attempt {attempt+1}/3) — retrying in {wait}s")
                time.sleep(wait)
            else:
                logger.warning(f"EDGAR CIK fetch returned {r.status_code}")
                break
        except Exception as e:
            logger.warning(f"EDGAR CIK fetch error (attempt {attempt+1}/3): {e}")
            time.sleep(2 ** attempt)

    logger.warning("EDGAR CIK map unavailable — catalyst scores will be 0 this run")
    return _cik_map


def get_cik(ticker):
    """Return zero-padded 10-digit CIK string for a ticker, or None if not found."""
    return _load_cik_map().get(ticker.upper())


# ── Form 4 XML Parser ────────────────────────────────────
def _parse_form4(xml_text):
    """
    Parse Form 4 XML for open-market purchase transactions (transactionCode = P).
    Returns list of {name, role, shares, price, date}.
    """
    purchases = []
    try:
        root = ET.fromstring(xml_text)

        name_el = root.find(".//reportingOwnerId/rptOwnerName")
        name = name_el.text.strip() if name_el is not None else "Unknown"

        rel = root.find(".//reportingOwnerRelationship")
        role = "Unknown"
        if rel is not None:
            title_el = rel.find("officerTitle")
            is_officer = rel.find("isOfficer")
            is_director = rel.find("isDirector")
            if title_el is not None and title_el.text:
                role = title_el.text.strip()
            elif is_officer is not None and is_officer.text == "1":
                role = "Officer"
            elif is_director is not None and is_director.text == "1":
                role = "Director"

        for tx in root.findall(".//nonDerivativeTransaction"):
            code_el = tx.find("transactionAmounts/transactionCode")
            if code_el is None or code_el.text != "P":
                continue
            shares_el = tx.find("transactionAmounts/transactionShares/value")
            price_el = tx.find("transactionAmounts/transactionPricePerShare/value")
            date_el = tx.find("transactionDate/value")
            purchases.append({
                "name": name,
                "role": role,
                "shares": int(float(shares_el.text)) if shares_el is not None else 0,
                "price": float(price_el.text) if price_el is not None else 0.0,
                "date": date_el.text if date_el is not None else "N/A",
            })
    except Exception as e:
        logger.debug(f"Form 4 XML parse error: {e}")
    return purchases


# ── Main Fetch ───────────────────────────────────────────
def get_insider_buys(ticker, days=30):
    """
    Fetch recent open-market insider purchases from SEC EDGAR Form 4 filings.
    Returns list of {name, role, shares, price, date}, newest first.
    """
    cik = get_cik(ticker)
    if not cik:
        logger.debug(f"EDGAR: no CIK found for {ticker}")
        return []

    cik_int = int(cik)
    cutoff = datetime.now() - timedelta(days=days)

    try:
        sub = _get_submissions(cik)
        if not sub:
            return []

        recent = sub.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        buys = []
        for form, date_str, acc, pdoc in zip(forms, dates, accessions, primary_docs):
            if form != "4":
                continue
            try:
                filing_date = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                continue
            if filing_date < cutoff:
                break  # filings are newest-first; stop when older than cutoff

            acc_clean = acc.replace("-", "")
            xml_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/{pdoc}"
            try:
                xr = requests.get(xml_url, headers=_SEC_HEADERS, timeout=8)
                if xr.status_code == 200:
                    purchases = _parse_form4(xr.text)
                    buys.extend(purchases)
            except Exception as e:
                logger.debug(f"EDGAR filing fetch failed for {ticker} {acc}: {e}")
                continue

        return buys

    except Exception as e:
        logger.warning(f"EDGAR insider fetch failed for {ticker}: {e}")
        return []


# ── Finnhub Insider Fetch (Streamlit Cloud fallback) ─────
def get_insider_buys_finnhub(ticker: str, days: int = 14) -> list:
    """Fetch open-market insider purchases from Finnhub's API.

    Used as a fallback when EDGAR is blocked (e.g. Streamlit Cloud shared IPs).
    Returns the same schema as get_insider_buys: [{name, role, shares, price, date}].
    Requires FINNHUB_API_KEY env var. Returns [] if key missing or on any error.
    """
    global _FINNHUB_LAST_CALL
    api_key = os.environ.get("FINNHUB_API_KEY", "")
    if not api_key:
        return []
    try:
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        to_date   = datetime.now().strftime("%Y-%m-%d")
        url = (
            f"https://finnhub.io/api/v1/stock/insider-transactions"
            f"?symbol={ticker}&from={from_date}&to={to_date}&token={api_key}"
        )
        # Enforce minimum gap between calls so we stay under 60 req/min free-tier limit
        with _FINNHUB_LOCK:
            now = time.time()
            gap = _FINNHUB_MIN_GAP - (now - _FINNHUB_LAST_CALL)
            if gap > 0:
                time.sleep(gap)
            _FINNHUB_LAST_CALL = time.time()

        r = requests.get(url, timeout=10)
        if r.status_code == 429:
            logger.debug(f"Finnhub rate limit hit for {ticker} — skipping")
            return []
        if r.status_code != 200:
            logger.debug(f"Finnhub insider fetch for {ticker}: HTTP {r.status_code}")
            return []

        transactions = r.json().get("data") or []
        buys = []
        for tx in transactions:
            if tx.get("transactionCode") != "P":
                continue  # only open-market purchases
            shares = int(tx.get("change", 0) or 0)
            price  = float(tx.get("transactionPrice", 0) or 0)
            if shares <= 0 or price <= 0:
                continue
            buys.append({
                "name":   tx.get("name", "Unknown"),
                "role":   "",  # Finnhub omits role in this endpoint
                "shares": shares,
                "price":  price,
                "date":   tx.get("transactionDate") or tx.get("filingDate", "N/A"),
            })
        return buys
    except Exception as e:
        logger.debug(f"Finnhub insider fetch failed for {ticker}: {e}")
        return []


# ── Insider Cluster Score ────────────────────────────────
def get_insider_cluster_score(ticker, days=14):
    """Return (score, reasons) based on insider buy clustering within `days`.

    Tries EDGAR first (works on GitHub Actions). Also checks Finnhub (works on
    Streamlit Cloud where EDGAR is often blocked). Results are combined so
    neither source can double-count the same transaction.
    """
    try:
        edgar_buys = get_insider_buys(ticker, days=days)

        # Finnhub supplement — fills in when EDGAR returns 0 on Streamlit Cloud
        finnhub_buys = get_insider_buys_finnhub(ticker, days=days)
        seen_keys = {(b["name"], b["date"]) for b in edgar_buys}
        for b in finnhub_buys:
            key = (b["name"], b["date"])
            if key not in seen_keys:
                edgar_buys.append(b)
                seen_keys.add(key)

        buys = edgar_buys
        if not buys:
            return 0, []

        score, reasons = 0, []

        distinct_insiders = {b["name"] for b in buys}
        count = len(distinct_insiders)

        if count >= 3:
            score += 7
            reasons.append(f"Insider cluster ({count}x)")
        elif count == 2:
            score += 4
            reasons.append("Dual insider buy")
        else:
            score += 2

        # CEO/CFO bonus — exec buying own company stock is one of the strongest pre-run signals
        exec_roles = {"Chief Executive", "CEO", "Chief Financial", "CFO", "President"}
        for b in buys:
            role = b.get("role", "")
            if any(r in role for r in exec_roles):
                score += 3
                reasons.append(f"Exec buy ({role})")
                break  # one bonus only

        score = min(9, score)
        return score, reasons
    except Exception as e:
        logger.debug(f"get_insider_cluster_score({ticker}) failed: {e}")
        return 0, []


# ── Form 8-K Fetcher ─────────────────────────────────────
_8K_ITEM_LABELS = {
    "1.01": "Material Agreement",
    "1.02": "Termination of Agreement",
    "2.01": "Asset Acquisition",
    "2.02": "Earnings Results",
    "5.02": "Officer/Director Change",
    "7.01": "Reg FD Disclosure",
    "8.01": "Other Events",
}

_8K_ITEM_POINTS = {
    "1.01": 3,
    "7.01": 3,
    "8.01": 3,
    "2.01": 2,
    "2.02": 2,
}


def fetch_recent_8k(ticker, days=10):
    """Return list of recent 8-K filings as [{date, items, description}]."""
    try:
        cik = get_cik(ticker)
        if not cik:
            return []

        cik_int = int(cik)
        cutoff = datetime.now() - timedelta(days=days)

        sub = _get_submissions(cik)
        if not sub:
            return []

        recent = sub.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        items_list = recent.get("items", [])

        filings = []
        for form, date_str, items_raw in zip(forms, dates, items_list):
            if form != "8-K":
                continue
            try:
                filing_date = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                continue
            if filing_date < cutoff:
                break

            items_str = str(items_raw) if items_raw else ""
            item_codes = [i.strip() for i in items_str.replace(",", " ").split() if i.strip()]
            descriptions = [_8K_ITEM_LABELS.get(c, "Other") for c in item_codes]
            filings.append({
                "date": date_str,
                "items": item_codes,
                "description": ", ".join(descriptions) if descriptions else "8-K Filing",
            })

        return filings
    except Exception as e:
        logger.debug(f"fetch_recent_8k({ticker}) failed: {e}")
        return []


def get_8k_catalyst_score(ticker, days=10):
    """Return (score, reasons) from recent 8-K filings."""
    try:
        filings = fetch_recent_8k(ticker, days=days)
        if not filings:
            return 0, []

        score, reasons = 0, []
        items_counted = 0

        for filing in filings:
            for code in filing.get("items", []):
                if items_counted >= 3:
                    break
                pts = _8K_ITEM_POINTS.get(code, 1)
                label = _8K_ITEM_LABELS.get(code, "8-K Event")
                score += pts
                reasons.append(f"8-K: {label}")
                items_counted += 1

        score = min(7, score)
        return score, reasons
    except Exception as e:
        logger.debug(f"get_8k_catalyst_score({ticker}) failed: {e}")
        return 0, []


# ── Telegram Block Formatter ─────────────────────────────
def format_edgar_block(ticker, buys):
    lines = [f"\n\U0001f3db *EDGAR Insider Activity — {ticker}*\n{'─' * 28}"]
    if not buys:
        lines.append("❌ No open-market purchases in last 30 days")
    else:
        lines.append(f"✅ {len(buys)} open-market purchase transaction(s):")
        for b in buys[:4]:
            price_str = f" @ ${b['price']:.2f}" if b.get("price") else ""
            lines.append(
                f"  • {b['name']} ({b['role']}) — "
                f"{b['shares']:,} shares{price_str} on {b['date']}"
            )
    lines.append(f"{'─' * 28}\n")
    return "\n".join(lines)


# ── Post-Scan Enrichment Entry Point ─────────────────────
def run_edgar_enrichment(results, send_telegram_fn):
    """
    Fetch EDGAR Form 4 data for HIGH CONVICTION tickers and send to Telegram.
    Only runs for picks with score >= HIGH_CONVICTION_THRESHOLD.
    """
    high_conviction = [r for r in results if r.get("score", 0) >= HIGH_CONVICTION_THRESHOLD]
    if not high_conviction:
        logger.info("EDGAR enrichment: no HIGH CONVICTION picks — skipping")
        return

    logger.info(f"EDGAR enrichment for {len(high_conviction)} ticker(s)...")
    blocks = []
    for pick in high_conviction:
        ticker = pick["ticker"]
        try:
            buys = get_insider_buys(ticker, days=30)
            blocks.append(format_edgar_block(ticker, buys))
            logger.info(f"EDGAR: {ticker} ready ({len(buys)} buy(s) found)")
        except Exception as e:
            logger.warning(f"EDGAR enrichment failed for {ticker}: {e}")

    if blocks:
        send_telegram_fn("".join(blocks))
        logger.info(f"EDGAR: {len(blocks)} ticker(s) sent in one message")
