import requests
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

logger = logging.getLogger("Argus.EDGAR")

_EDGAR_AGENT = "Argus Investment Workstation/1.0 (argus@local)"
_EDGAR_HEADERS = {"User-Agent": _EDGAR_AGENT}
_SEC_HEADERS = {"User-Agent": _EDGAR_AGENT}

_cik_map: dict = {}

HIGH_CONVICTION_THRESHOLD = 80


# ── CIK Resolution ───────────────────────────────────────
def _load_cik_map():
    """Download the full EDGAR company_tickers.json once and cache in memory."""
    global _cik_map
    if _cik_map:
        return _cik_map
    try:
        r = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=_SEC_HEADERS,
            timeout=15,
        )
        if r.status_code != 200:
            logger.warning(f"CIK map fetch returned {r.status_code}")
            return {}
        for entry in r.json().values():
            ticker = entry.get("ticker", "").upper()
            cik = str(entry.get("cik_str", "")).zfill(10)
            if ticker:
                _cik_map[ticker] = cik
        logger.info(f"EDGAR CIK map loaded: {len(_cik_map)} tickers")
    except Exception as e:
        logger.warning(f"Failed to load EDGAR CIK map: {e}")
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
        r = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            headers=_EDGAR_HEADERS,
            timeout=12,
        )
        if r.status_code != 200:
            logger.warning(f"EDGAR submissions {ticker}: HTTP {r.status_code}")
            return []

        recent = r.json().get("filings", {}).get("recent", {})
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


# ── Telegram Block Formatter ─────────────────────────────
def format_edgar_block(ticker, buys):
    lines = [f"\n\U0001f3db *EDGAR Insider Activity \u2014 {ticker}*\n{'─' * 28}"]
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
    for pick in high_conviction:
        ticker = pick["ticker"]
        try:
            buys = get_insider_buys(ticker, days=30)
            block = format_edgar_block(ticker, buys)
            send_telegram_fn(block)
            logger.info(f"EDGAR: {ticker} sent ({len(buys)} buy(s) found)")
        except Exception as e:
            logger.warning(f"EDGAR enrichment failed for {ticker}: {e}")
