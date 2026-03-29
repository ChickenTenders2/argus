import streamlit as st
import pandas as pd
import yfinance as yf
from engine import Config, run_scan, save_results
import os
import requests
from datetime import datetime

st.set_page_config(page_title="Argus Dashboard", layout="wide")
st.title("👁 Argus Investment Workstation")

def send_telegram_message(token, chat_id, message):
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
        return True
    except Exception:
        return False

with st.sidebar:
    st.header("Parameters")
    min_score   = st.slider("Minimum Score", 50, 95, 65)
    price_floor = st.number_input("Price Floor ($)", 0.0, 100.0, 2.0)
    vol_floor   = st.number_input("Volume Floor (avg daily)", 0, 2_000_000, 500_000, step=100_000)
    scan_limit  = st.slider("Universe size (tickers to scan)", 50, 400, 200, step=50)
    send_to_telegram = st.checkbox("Send manual run to Telegram", value=False)

    config = Config(MIN_SCORE=min_score, PRICE_FLOOR=price_floor, VOL_FLOOR=vol_floor)

latest_tab, history_tab, manual_tab = st.tabs(["Latest Scheduled", "History", "Manual Run"])

with latest_tab:
    st.subheader("📡 Latest Scheduled Scan")
    if os.path.exists(config.RESULTS_FILE):
        nightly = pd.read_csv(config.RESULTS_FILE)
        if nightly.empty:
            st.info("Latest scheduled scan has no picks.")
        else:
            scan_date = nightly["scan_date"].iloc[0] if "scan_date" in nightly.columns else "Unknown"
            st.caption(f"Scan date: {scan_date}  ·  {len(nightly)} tickers")
            st.dataframe(nightly.drop(columns=["scan_timestamp"], errors="ignore"), use_container_width=True)
    else:
        st.info("No nightly scan data yet — scheduled results will appear here automatically.")

with history_tab:
    st.subheader("🗂 Scan History")
    if os.path.exists(config.RESULTS_HISTORY_FILE):
        history = pd.read_csv(config.RESULTS_HISTORY_FILE)
        if history.empty:
            st.info("No history yet.")
        else:
            daily = (
                history.groupby(["scan_date", "run_type"], as_index=False)
                .agg(picks=("ticker", "count"), top_score=("score", "max"))
                .sort_values("scan_date", ascending=False)
            )
            st.caption("Daily run summaries")
            st.dataframe(daily, use_container_width=True)

            day_options = daily["scan_date"].drop_duplicates().tolist()
            selected_day = st.selectbox("View day details", day_options)
            detail = history[history["scan_date"] == selected_day].sort_values("score", ascending=False)
            st.dataframe(detail, use_container_width=True)
    else:
        st.info("No history file yet. It will be created after first run.")

with manual_tab:
    st.subheader("⚙️ Manual Scan")
    st.caption("Use this for weekend/on-demand runs. Results are written to history.")
    if st.button("Run Global Scan"):
        with st.spinner("Scanning tickers..."):
            payload = run_scan(config=config, scan_limit=scan_limit, update_memory=True)
            results = payload["results"]
            scan_date = payload["scan_date"]
            scan_timestamp = payload["scan_timestamp"]
            scanned_count = payload["scanned_count"]

            save_results(
                results=results,
                scan_date=scan_date,
                scan_timestamp=scan_timestamp,
                run_type="manual",
                latest_file=config.RESULTS_FILE,
                history_file=config.RESULTS_HISTORY_FILE,
                write_latest=False,
            )

        st.caption(f"Scanned {scanned_count} tickers")
        if results:
            df_res = pd.DataFrame(results).sort_values("score", ascending=False)
            st.dataframe(df_res, use_container_width=True)

            st.subheader("High Conviction Charts (Score >= 80)")
            high_conviction = df_res[df_res["tier"] == "🟢 HIGH CONVICTION"]
            if not high_conviction.empty:
                cols = st.columns(3)
                for idx, row in high_conviction.reset_index(drop=True).iterrows():
                    with cols[idx % 3]:
                        st.write(f"**{row['ticker']}** - Score: {row['score']}")
                        hist = yf.Ticker(row["ticker"]).history(period="6mo")
                        if not hist.empty:
                            st.line_chart(hist["Close"])
                        else:
                            st.write("No history available")
            else:
                st.info("No high conviction picks found in this run.")
        else:
            st.warning("No tickers met the current criteria.")

        if send_to_telegram:
            top_lines = [f"{r['ticker']} ({r['score']})" for r in sorted(results, key=lambda x: x["score"], reverse=True)]
            top_text = "\n".join(top_lines) if top_lines else "No qualifying picks."
            message = (
                f"👁 *Argus Manual Scan — {datetime.now().strftime('%d %b %Y %H:%M')}*\n"
                f"Scanned {scanned_count} tickers\n\n{top_text}"
            )
            delivered = send_telegram_message(config.TELEGRAM_TOKEN, config.TELEGRAM_CHAT_ID, message)
            if delivered:
                st.success("Manual run sent to Telegram.")
            else:
                st.error("Could not send to Telegram. Check TELEGRAM_TOKEN and TELEGRAM_CHAT_ID.")
