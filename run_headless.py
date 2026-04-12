import sys
import os
from datetime import datetime
import pandas as pd
from engine import Config, run_scan, save_results

def send_telegram_message(token, chat_id, message):
    import requests
    # Always log locally first (this is what Streamlit reads)
    try:
        with open("argus_alerts_log.txt", "a", encoding="utf-8") as f:
            f.write(f"--- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n{message}\n\n")
    except Exception as e:
        print(f"Failed to write to local log: {e}")

    if not token or not chat_id:
        print("Missing Telegram credentials. Skipping API request.")
        return False
        
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        req = requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=10)
        req.raise_for_status()
        print("Telegram pushed successfully!")
        return True
    except Exception as e:
        print(f"Telegram error: {e}")
        return False

def main():
    config = Config()
    
    print("Starting automated headless scan...")
    # Scan a set amount for the automated daily run
    payload = run_scan(config=config, scan_limit=200, update_memory=True)
    
    results = payload["results"]
    scanned_count = payload["scanned_count"]
    
    save_results(
        results=results,
        scan_date=payload["scan_date"],
        scan_timestamp=payload["scan_timestamp"],
        run_type="automated",
        latest_file=config.RESULTS_FILE,
        history_file=config.RESULTS_HISTORY_FILE,
        write_latest=True,
        feature_file=config.FEATURES_FILE,
    )
    
    print(f"Scan complete. {len(results)} picks found.")
    
    # Construct Message
    top_lines = [f"{r['ticker']} ({r['score']})" for r in sorted(results, key=lambda x: x["score"], reverse=True)]
    top_text = "\n".join(top_lines) if top_lines else "No qualifying picks."
    message = (
        f"🤖 *Argus Auto-Scan — {datetime.now().strftime('%d %b %Y')}*\n"
        f"Scanned {scanned_count} tickers\n\n{top_text}"
    )
    
    # Fetch tokens from environment (GitHub Actions) or fallback to config
    token = os.environ.get("TELEGRAM_TOKEN") or getattr(config, "TELEGRAM_TOKEN", None)
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or getattr(config, "TELEGRAM_CHAT_ID", None)
    
    # Push Telegram and explicitly log to argus_alerts_log.txt
    send_telegram_message(
        token=token,
        chat_id=chat_id,
        message=message
    )

if __name__ == "__main__":
    main()