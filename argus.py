import yfinance as yf
import pandas as pd
import requests
import logging
from datetime import datetime
from fmp_fetch import run_fmp_enrichment
from engine import Config, run_scan, save_results

# ── Setup Logging ───────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Argus")

config = Config()

# ── Telegram ─────────────────────────────────────────────
def send_telegram(message):
    if not config.TELEGRAM_TOKEN or not config.TELEGRAM_CHAT_ID:
        logger.error("Telegram credentials missing. Cannot send message.")
        return
    url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": config.TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")

# ── Format Telegram Message ──────────────────────────────
def format_pick(pick, memory_df):
    ticker     = pick["ticker"]
    prev       = memory_df[memory_df["ticker"] == ticker]
    memory_note = ""
    if not prev.empty:
        times = int(prev["times_flagged"].values[0])
        first = prev["first_seen"].values[0]
        memory_note = f"\n⚡ _Previously flagged {times}x since {first} — signals strengthening_"

    reasons_str = " | ".join(pick["reasons"])
    return (
        f"{pick['tier']}\n"
        f"*{ticker}* — Score: *{pick['score']}/100*{memory_note}\n"
        f"💰 Price: ${pick['price']} | Cap: {pick['mkt_cap']}\n"
        f"📊 {reasons_str}\n"
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

    logger.info("Argus scan starting...")
    scan_payload = run_scan(config=config, scan_limit=400, update_memory=True)
    results = scan_payload["results"]
    scan_date = scan_payload["scan_date"]
    scan_timestamp = scan_payload["scan_timestamp"]
    scanned_count = scan_payload["scanned_count"]

    if not results:
        send_telegram(
            f"👁 *Argus Daily Scan — {datetime.now().strftime('%d %b %Y')}*\n"
            f"No high-conviction picks found today. Market may be choppy."
        )
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
    memory_df = pd.read_csv(config.MEMORY_FILE)
    highest = [p for p in results if p["score"] >= 80]
    high = [p for p in results if p["score"] < 80]
    header = f"👁 *Argus Daily Scan — {today_str}*\n{'─'*30}\n"
    highest_block = "*🚀 Highest scoring picks*\n" + ("\n".join([format_pick(p, memory_df) for p in highest]) if highest else "_None today_")
    high_block = "\n*📌 High scoring picks*\n" + ("\n".join([format_pick(p, memory_df) for p in high]) if high else "_None today_")
    footer = f"\n{'─'*30}\n_Scanned {scanned_count} tickers • Top {len(results)} picks shown_"
    body = highest_block + high_block
    send_telegram(header + body + footer)
    
    # FMP enrichment is intentionally disabled — enrichment runs separately
    # via fmp_fetch.py to avoid blocking the nightly scan Action.
    # run_fmp_enrichment(results, send_telegram)

    # ── Watchlist monitor ──
    run_watchlist_monitor()
    logger.info("Argus scan complete.")

if __name__ == "__main__":
    main()
