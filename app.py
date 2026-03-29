import streamlit as st
import pandas as pd
import yfinance as yf
from engine import (
    Config,
    run_scan,
    save_results,
    build_prediction_model,
    add_predictions,
    add_risk_guidance,
    save_journal_entry,
    load_journal,
)
import os
import requests
from datetime import datetime

st.set_page_config(page_title="Argus Dashboard", layout="wide")
st.title("👁 Argus Investment Workstation")

def safe_line_chart(data, y_label="value"):
    """
    Streamlit line_chart may fail on some Cloud runtimes due to Altair/Python
    compatibility. Fall back to a table to keep the app usable.
    """
    try:
        st.line_chart(data)
    except Exception:
        st.warning("Chart unavailable in current runtime; showing data table instead.")
        if isinstance(data, pd.Series):
            fallback = data.reset_index()
            fallback.columns = ["x", y_label]
            st.dataframe(fallback, use_container_width=True)
        else:
            st.dataframe(data, use_container_width=True)

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
    min_score = st.slider("Minimum Score", 50, 95, 65)
    price_floor = st.number_input("Price Floor ($)", 0.0, 100.0, 2.0)
    vol_floor = st.number_input("Volume Floor (avg daily)", 0, 2_000_000, 500_000, step=100_000)
    scan_limit = st.slider("Universe size (tickers to scan)", 50, 400, 200, step=50)
    st.divider()
    st.subheader("Prediction Model")
    horizon_days = st.selectbox("Horizon days", [42, 63, 84], index=1)
    target_return = st.slider("Target return (%)", 5, 40, 10) / 100.0
    st.subheader("Risk Rules")
    risk_per_trade_pct = st.slider("Risk per trade (% of portfolio)", 0.25, 2.0, 0.75, 0.25)
    max_position_pct = st.slider("Max position size (%)", 2.0, 20.0, 8.0, 0.5)
    st.divider()
    send_to_telegram = st.checkbox("Send manual run to Telegram", value=False)

    config = Config(MIN_SCORE=min_score, PRICE_FLOOR=price_floor, VOL_FLOOR=vol_floor)

model = build_prediction_model(
    features_file=config.FEATURES_FILE,
    horizon_days=horizon_days,
    target_return=target_return,
)

if os.path.exists(config.RESULTS_FILE):
    latest_df = pd.read_csv(config.RESULTS_FILE)
else:
    latest_df = pd.DataFrame()

if os.path.exists(config.RESULTS_HISTORY_FILE):
    history_df = pd.read_csv(config.RESULTS_HISTORY_FILE)
else:
    history_df = pd.DataFrame()

tabs = st.tabs(["Overview", "History", "Ticker Detail", "Journal", "Manual Run", "Prediction Model"])

with tabs[0]:
    st.subheader("📡 Latest Scheduled Scan")
    if latest_df.empty:
        st.info("No latest scheduled results yet.")
    else:
        latest_view = latest_df.copy()
        latest_view = add_predictions(latest_view, model)
        latest_view = add_risk_guidance(
            latest_view,
            model,
            risk_per_trade_pct=risk_per_trade_pct,
            max_position_pct=max_position_pct,
        )
        scan_date = latest_view["scan_date"].iloc[0] if "scan_date" in latest_view.columns else "Unknown"
        st.caption(f"Scan date: {scan_date} · {len(latest_view)} picks")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Picks", len(latest_view))
        c2.metric("High Conviction", int((latest_view["tier"] == "🟢 HIGH CONVICTION").sum()))
        c3.metric("Avg Score", f"{latest_view['score'].mean():.1f}")
        if "prob_upside" in latest_view.columns and latest_view["prob_upside"].notna().any():
            c4.metric("Avg Upside Prob", f"{latest_view['prob_upside'].mean() * 100:.1f}%")
        else:
            c4.metric("Avg Upside Prob", "N/A")

        display_cols = [
            "ticker", "sector", "score", "tier", "price", "mkt_cap",
            "prob_upside", "scenario_bear", "scenario_base", "scenario_bull",
            "confidence", "suggested_position_pct", "stop_loss_pct", "take_profit_pct", "entry_style",
        ]
        for col in ["prob_upside", "scenario_bear", "scenario_base", "scenario_bull"]:
            if col in latest_view.columns:
                latest_view[col] = (latest_view[col] * 100).round(1)
        st.dataframe(latest_view[[c for c in display_cols if c in latest_view.columns]], use_container_width=True)

with tabs[1]:
    st.subheader("🗂 History")
    if history_df.empty:
        st.info("No history file yet.")
    else:
        history_df["scan_date"] = pd.to_datetime(history_df["scan_date"], errors="coerce")
        run_type_opts = ["all"] + sorted(history_df["run_type"].dropna().unique().tolist())
        selected_run_type = st.selectbox("Run type", run_type_opts)
        if selected_run_type != "all":
            filtered = history_df[history_df["run_type"] == selected_run_type].copy()
        else:
            filtered = history_df.copy()

        filtered = filtered.dropna(subset=["scan_date"])
        if filtered.empty:
            st.warning("No rows after filters.")
        else:
            filtered["scan_day"] = filtered["scan_date"].dt.date
            daily = (
                filtered.groupby("scan_day", as_index=False)
                .agg(picks=("ticker", "count"), top_score=("score", "max"), avg_score=("score", "mean"))
                .sort_values("scan_day", ascending=False)
            )
            st.dataframe(daily, use_container_width=True)

            st.caption("Score trend")
            trend = (
                filtered.groupby("scan_day", as_index=False)
                .agg(avg_score=("score", "mean"))
                .sort_values("scan_day")
            )
            safe_line_chart(trend.set_index("scan_day")["avg_score"], y_label="avg_score")

            date_options = [d.strftime("%Y-%m-%d") for d in sorted(filtered["scan_day"].unique(), reverse=True)]
            selected_day = st.selectbox("Day details", date_options)
            day_df = filtered[filtered["scan_date"].dt.strftime("%Y-%m-%d") == selected_day].sort_values("score", ascending=False)
            st.dataframe(day_df, use_container_width=True)

with tabs[2]:
    st.subheader("🔎 Ticker Detail")
    if history_df.empty:
        st.info("No history available yet.")
    else:
        ticker_options = sorted(history_df["ticker"].dropna().unique().tolist())
        ticker = st.selectbox("Select ticker", ticker_options)
        tdf = history_df[history_df["ticker"] == ticker].copy()
        tdf["scan_date"] = pd.to_datetime(tdf["scan_date"], errors="coerce")
        tdf = tdf.sort_values("scan_date")
        st.dataframe(tdf, use_container_width=True)

        if not tdf.empty:
            st.caption("Argus score history")
            safe_line_chart(tdf.set_index("scan_date")["score"], y_label="score")

            hist = yf.Ticker(ticker).history(period="1y")
            if not hist.empty:
                st.caption("Price (1 year)")
                safe_line_chart(hist["Close"], y_label="close")
            else:
                st.info("No market price data available for chart.")

            latest_ticker_row = tdf.sort_values("scan_date", ascending=False).head(1).copy()
            latest_ticker_row = add_predictions(latest_ticker_row, model)
            latest_ticker_row = add_risk_guidance(
                latest_ticker_row,
                model,
                risk_per_trade_pct=risk_per_trade_pct,
                max_position_pct=max_position_pct,
            )
            st.markdown("**Execution Guidance (latest signal)**")
            st.dataframe(
                latest_ticker_row[
                    [
                        "ticker",
                        "score",
                        "prob_upside",
                        "confidence",
                        "suggested_position_pct",
                        "stop_loss_pct",
                        "take_profit_pct",
                        "entry_style",
                    ]
                ],
                use_container_width=True,
            )

with tabs[3]:
    st.subheader("📝 Decision Journal")
    with st.form("journal_form"):
        col1, col2, col3 = st.columns(3)
        journal_ticker = col1.text_input("Ticker").upper().strip()
        journal_action = col2.selectbox("Action", ["BUY", "PASS", "SELL", "SCALE_IN", "TRIM"])
        journal_scan_date = col3.text_input("Related scan date (YYYY-MM-DD)", value=datetime.now().strftime("%Y-%m-%d"))
        col4, col5, col6 = st.columns(3)
        entry_price = col4.number_input("Entry price", min_value=0.0, value=0.0)
        position_size_pct = col5.number_input("Position size %", min_value=0.0, max_value=100.0, value=2.0)
        stop_loss_pct = col6.number_input("Stop loss %", min_value=0.0, max_value=100.0, value=8.0)
        take_profit_pct = st.number_input("Take profit %", min_value=0.0, max_value=300.0, value=25.0)
        notes = st.text_area("Notes")
        submitted = st.form_submit_button("Save Journal Entry")
        if submitted:
            if not journal_ticker:
                st.error("Ticker is required.")
            else:
                save_journal_entry(
                    config.JOURNAL_FILE,
                    {
                        "ticker": journal_ticker,
                        "action": journal_action,
                        "scan_date": journal_scan_date,
                        "entry_price": entry_price,
                        "position_size_pct": position_size_pct,
                        "stop_loss_pct": stop_loss_pct,
                        "take_profit_pct": take_profit_pct,
                        "notes": notes,
                    },
                )
                st.success("Journal entry saved.")

    journal_df = load_journal(config.JOURNAL_FILE)
    if journal_df.empty:
        st.info("No journal entries yet.")
    else:
        st.dataframe(journal_df.sort_values("timestamp", ascending=False), use_container_width=True)

with tabs[4]:
    st.subheader("⚙️ Manual Scan")
    st.caption("Use this for weekend/on-demand runs. Results are written to history + feature store.")
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
                feature_file=config.FEATURES_FILE,
            )

        st.caption(f"Scanned {scanned_count} tickers")
        if results:
            df_res = pd.DataFrame(results).sort_values("score", ascending=False)
            df_res = add_predictions(df_res, model)
            df_res = add_risk_guidance(
                df_res,
                model,
                risk_per_trade_pct=risk_per_trade_pct,
                max_position_pct=max_position_pct,
            )
            for col in ["prob_upside", "scenario_bear", "scenario_base", "scenario_bull"]:
                if col in df_res.columns:
                    df_res[col] = (df_res[col] * 100).round(1)
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
                            safe_line_chart(hist["Close"], y_label="close")
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

with tabs[5]:
    st.subheader("📈 Prediction Model Quality")
    if not model.get("ready"):
        st.info(model.get("reason", "Model is not ready yet."))
        st.caption("Keep running scans to accumulate matured samples.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Matured Samples", model["samples"])
        col2.metric("Global Hit Rate", f"{model['global_prob'] * 100:.1f}%")
        col3.metric("Brier Score", f"{model['brier_score']:.3f}")
        st.caption(f"Target: +{target_return * 100:.0f}% in {horizon_days} days")

        bucket_stats = model["bucket_stats"].copy()
        for c in ["prob", "bear", "base", "bull"]:
            bucket_stats[c] = (bucket_stats[c] * 100).round(1)
        st.markdown("**Score bucket outcomes**")
        st.dataframe(bucket_stats, use_container_width=True)

        calibration = model["calibration"].copy()
        calibration["actual_hit_rate"] = (calibration["actual_hit_rate"] * 100).round(1)
        st.markdown("**Calibration table**")
        st.dataframe(calibration, use_container_width=True)
