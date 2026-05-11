import os
import yfinance as yf
import pandas as pd
import requests
import logging
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fmp_fetch import run_fmp_enrichment, get_fmp_data, parse_fmp_data, format_fmp_block
from edgar_fetch import run_edgar_enrichment, get_insider_buys, format_edgar_block
from engine import Config, run_scan, save_results, load_memory, monitor_portfolio

try:
    from llm import generate_ai_thesis
except ImportError:
    generate_ai_thesis = None

# ── Setup Logging ───────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Argus")

config = Config()
TELEGRAM_MESSAGE_LIMIT = 4000


def _split_telegram_message(message, limit=TELEGRAM_MESSAGE_LIMIT):
    if len(message) <= limit:
        return [message]

    chunks = []
    remaining = message
    while len(remaining) > limit:
        split_at = remaining.rfind("\n\n", 0, limit)
        if split_at == -1:
            split_at = remaining.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        chunk = remaining[:split_at].strip()
        if not chunk:
            chunk = remaining[:limit]
            split_at = limit
        chunks.append(chunk)
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks

# ── Telegram ─────────────────────────────────────────────
def send_telegram(message):
    try:
        with open("argus_alerts_log.txt", "a", encoding="utf-8") as f:
            f.write(f"--- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n{message}\n\n")
    except Exception as e:
        logger.error(f"Failed to write to alerts log: {e}")

    if not config.TELEGRAM_TOKEN or not config.TELEGRAM_CHAT_ID:
        logger.error("Telegram credentials missing. Cannot send message.")
        return False
    url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
    sent_chunks = 0
    for chunk in _split_telegram_message(message):
        payload = {
            "chat_id": config.TELEGRAM_CHAT_ID,
            "text": chunk,
            "parse_mode": "Markdown"
        }
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.ok:
                sent_chunks += 1
                continue

            logger.error(f"Telegram API rejected message chunk: {response.status_code} {response.text}")
            fallback_payload = {
                "chat_id": config.TELEGRAM_CHAT_ID,
                "text": chunk,
            }
            fallback_response = requests.post(url, json=fallback_payload, timeout=10)
            if fallback_response.ok:
                sent_chunks += 1
            else:
                logger.error(f"Telegram fallback send failed: {fallback_response.status_code} {fallback_response.text}")
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")

    if sent_chunks == 0:
        logger.error("Telegram delivery failed for all message chunks.")
        return False
    else:
        logger.info(f"Telegram delivery succeeded for {sent_chunks} message chunk(s).")
        return True

# ── Format Telegram Message ──────────────────────────────
def format_pick(pick, memory_df, ai_thesis=None):
    ticker     = pick["ticker"]
    prev       = memory_df[memory_df["ticker"] == ticker]
    memory_note = ""
    if not prev.empty:
        times = int(prev["times_flagged"].values[0])
        first = prev["first_seen"].values[0]
        memory_note = f"\n⚡ _Previously flagged {times}x since {first} — signals strengthening_"

    reasons_str = " | ".join(pick["reasons"])
    thesis_str = f"\n💡 *AI Note:* _{ai_thesis}_" if ai_thesis else ""
    
    return (
        f"{pick['tier']}\n"
        f"*{ticker}* — Score: *{pick['score']}/100*{memory_note}\n"
        f"💰 Price: ${pick['price']} | Cap: {pick['mkt_cap']}\n"
        f"📊 {reasons_str}{thesis_str}\n"
    )

# ── Watchlist Monitor ────────────────────────────────────
def run_watchlist_monitor():
    try:
        wl = pd.read_csv(config.WATCHLIST_FILE)
        tickers = wl["ticker"].dropna().tolist()
    except:
        return

    if not tickers:
        return

    lines = [f"👁 *Argus Watchlist Update — {datetime.now().strftime('%d %b %Y')}*\n{'─'*30}"]
    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker).history(period="2d")
            if len(hist) >= 2:
                price   = round(hist["Close"].iloc[-1], 2)
                day_chg = ((hist["Close"].iloc[-1] - hist["Close"].iloc[-2]) / hist["Close"].iloc[-2]) * 100
                arrow   = "📈" if day_chg > 0 else "📉"
                chg_str = f"{'+' if day_chg > 0 else ''}{day_chg:.1f}%"
                lines.append(f"{arrow} *{ticker}* — ${price} ({chg_str} today)")
            else:
                lines.append(f"📌 *{ticker}* — data unavailable")
        except:
            lines.append(f"📌 *{ticker}* — data unavailable")

    send_telegram("\n".join(lines))

# ── Main ─────────────────────────────────────────────────
def main():
    if not config.TELEGRAM_TOKEN or not config.TELEGRAM_CHAT_ID:
        logger.error("Telegram credentials missing. Exiting.")
        return

    try:
        _run()
    except Exception as e:
        logger.error(f"Fatal scan error: {e}")
        send_telegram(
            f"\U0001f6a8 *Argus Scan FAILED \u2014 {datetime.now().strftime('%d %b %Y')}*\n"
            f"Error: `{str(e)[:300]}`\nCheck GitHub Actions for details."
        )
        raise


def _send_combined_enrichment(results):
    """Build FMP + EDGAR enrichment blocks for all HIGH CONVICTION picks and send as one message."""
    high_conviction = [r for r in results if r.get("score", 0) >= 80]
    if not high_conviction:
        logger.info("Enrichment: no HIGH CONVICTION picks — skipping")
        return

    fmp_key = os.environ.get("FMP_API_KEY", "")
    blocks = []
    for pick in high_conviction:
        ticker = pick["ticker"]
        if fmp_key:
            try:
                raw = get_fmp_data(ticker)
                parsed = parse_fmp_data(ticker, raw)
                blocks.append(format_fmp_block(parsed))
            except Exception as e:
                logger.warning(f"FMP enrichment failed for {ticker}: {e}")
        try:
            buys = get_insider_buys(ticker, days=30)
            blocks.append(format_edgar_block(ticker, buys))
        except Exception as e:
            logger.warning(f"EDGAR enrichment failed for {ticker}: {e}")

    if blocks:
        send_telegram("".join(blocks))
        logger.info(f"Enrichment: combined message sent for {len(high_conviction)} HC ticker(s)")


def _run():
    logger.info("Argus scan starting...")
    scan_payload = run_scan(config=config, scan_limit=None, update_memory=True, run_type="scheduled")
    results = scan_payload["results"]
    scan_date = scan_payload["scan_date"]
    scan_timestamp = scan_payload["scan_timestamp"]
    scanned_count = scan_payload["scanned_count"]

    if not results:
        send_ok = send_telegram(
            f"👁 *Argus Daily Scan — {datetime.now().strftime('%d %b %Y')}*\n"
            f"No high-conviction picks found today. Market may be choppy."
        )
        if not send_ok:
            raise RuntimeError("Failed to deliver daily Telegram message.")
        save_results(
            results=[],
            scan_date=scan_date,
            scan_timestamp=scan_timestamp,
            run_type="scheduled",
            latest_file=config.RESULTS_FILE,
            history_file=config.RESULTS_HISTORY_FILE,
            write_latest=True,
            feature_file=config.FEATURES_FILE,
        )
        run_watchlist_monitor()
        return

    save_results(
        results=results,
        scan_date=scan_date,
        scan_timestamp=scan_timestamp,
        run_type="scheduled",
        latest_file=config.RESULTS_FILE,
        history_file=config.RESULTS_HISTORY_FILE,
        write_latest=True,
        feature_file=config.FEATURES_FILE,
    )

    # ── Build & send Telegram message ──
    today_str = datetime.now().strftime("%d %b %Y")
    
    try:
        memory_df = load_memory(config.MEMORY_FILE)
    except Exception:
        memory_df = pd.DataFrame(columns=["ticker"])
        
    highest = [p for p in results if p["score"] >= 80]
    high = [p for p in results if p["score"] < 80]
    
    # Optional LLM qualitative analysis for highest conviction picks
    formatted_highest = []
    for p in highest:
        ai_note = None
        if config.GROQ_API_KEY and generate_ai_thesis:
            ai_note = generate_ai_thesis(p["ticker"], p["score"], p["reasons"], config.GROQ_API_KEY)
            
            # Prevent markdown formatting issues in telegram if AI uses asterisks
            if ai_note:
                ai_note = ai_note.replace('*', '').replace('_', '')
                
        formatted_highest.append(format_pick(p, memory_df, ai_note))

    header = f"👁 *Argus Daily Scan — {today_str}*\n{'─'*30}\n"
    
    alerts = monitor_portfolio()
    alerts_block = ""
    if alerts:
        alerts_block = "*🚨 PORTFOLIO ALERTS*\n" + "\n".join(alerts) + f"\n\n{'─'*30}\n"
        
    highest_block = "*🚀 Highest scoring picks*\n" + ("\n".join(formatted_highest) if formatted_highest else "_None today_")
    high_block = ("\n*📌 High scoring picks*\n" + "\n".join([format_pick(p, memory_df) for p in high])) if high else ""
    footer = f"\n{'─'*30}\n_Scanned {scanned_count} tickers • Top {len(results)} picks shown_"
    body = highest_block + high_block
    send_ok = send_telegram(header + alerts_block + body + footer)
    if not send_ok:
        raise RuntimeError("Failed to deliver daily Telegram message.")
    
    _send_combined_enrichment(results)

    # ── Watchlist monitor ──
    run_watchlist_monitor()
    logger.info("Argus scan complete.")

if __name__ == "__main__":
    main()
