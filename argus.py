import yfinance as yf
import pandas as pd
import requests
import logging
import os
from datetime import datetime
from fmp_fetch import run_fmp_enrichment
from engine import Config, score_stock, load_memory, save_memory, get_universe

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
    tickers   = get_universe()
    memory_df = load_memory(config.MEMORY_FILE)
    results   = []

    logger.info("Pre-filtering universe...")
    batch_hist = yf.download(tickers, period="1mo", group_by="ticker", progress=False, threads=True)
    valid_tickers = []
    for t in tickers:
        try:
            if isinstance(batch_hist.columns, pd.MultiIndex) and t in batch_hist.columns.get_level_values(0):
                data = batch_hist[t]
            elif t in batch_hist.columns:
                data = batch_hist[t]
            else:
                continue
            if (not data['Close'].dropna().empty and
                data['Close'].iloc[-1] > config.PRICE_FLOOR and
                data['Volume'].mean() > config.VOL_FLOOR):
                valid_tickers.append(t)
        except:
            continue

    tickers = valid_tickers[:400]

    for ticker in tickers:
        pick = score_stock(ticker, memory_df, config)
        if pick:
            results.append(pick)

    from collections import Counter
    sector_picks = {}
    for p in results:
        sector = p.get("sector", "Unknown")
        if sector not in sector_picks or len(sector_picks[sector]) < 3:
            sector_picks.setdefault(sector, []).append(p)
    results = [p for picks in sector_picks.values() for p in picks][:config.TOP_N]

    if not results:
        send_telegram(
            f"👁 *Argus Daily Scan — {datetime.now().strftime('%d %b %Y')}*\n"
            f"No high-conviction picks found today. Market may be choppy."
        )
        run_watchlist_monitor()
        return

    # ── Update memory ──
    today = datetime.now().strftime("%d %b %Y")
    for pick in results:
        t = pick["ticker"]
        if t in memory_df["ticker"].values:
            times = int(memory_df.loc[memory_df["ticker"] == t, "times_flagged"].values[0])
            memory_df.loc[memory_df["ticker"] == t, "times_flagged"] += 1
            memory_df.loc[memory_df["ticker"] == t, "last_score"] = pick["score"]
        else:
            new_row = pd.DataFrame([{
                "ticker":        t,
                "first_seen":    today,
                "times_flagged": 1,
                "last_score":    pick["score"]
            }])
            memory_df = pd.concat([memory_df, new_row], ignore_index=True)
            
    # Save using the engine func
    save_memory(memory_df, config.MEMORY_FILE)

    # ── Build & send Telegram message ──
    today_str = datetime.now().strftime("%d %b %Y")
    header    = f"👁 *Argus Daily Scan — {today_str}*\n{'─'*30}\n"
    body      = "\n".join([format_pick(p, memory_df) for p in results])
    footer    = f"\n{'─'*30}\n_Scanned {len(tickers)} tickers • Top {len(results)} picks shown_"
    send_telegram(header + body + footer)
    
    # run_fmp_enrichment(results, send_telegram)

    # ── Watchlist monitor ──
    run_watchlist_monitor()
    logger.info("Argus scan complete.")

if __name__ == "__main__":
    main()
