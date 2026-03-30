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

def format_pct_columns(df, cols):
    """Coerce columns to numeric before % formatting to avoid dtype errors."""
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = (out[col] * 100).round(1)
    return out

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

tabs = st.tabs(["Overview", "History", "Ticker Detail", "Journal", "Manual Run", "Prediction Model", "Help", "Prompts"])

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
        latest_view = format_pct_columns(
            latest_view,
            ["prob_upside", "scenario_bear", "scenario_base", "scenario_bull"],
        )
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
        tdf = tdf.sort_values("scan_date", ascending=False)
        st.dataframe(tdf, use_container_width=True)

        if not tdf.empty:
            st.caption("Argus score history")
            safe_line_chart(tdf.set_index("scan_date")["score"], y_label="score")

            hist = yf.Ticker(ticker).history(period="1y")
            if not hist.empty:
                st.caption("Price (1 year)")
                safe_line_chart(hist["Close"], y_label="close")
                st.dataframe(hist.sort_index(ascending=False), use_container_width=True)
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

            st.markdown("### Buy & Sell Strategy")
            score_val = latest_ticker_row["score"].iloc[0]
            upside_prob = latest_ticker_row.get("prob_upside", pd.Series([0])).iloc[0]
            entry = latest_ticker_row.get("entry_style", pd.Series([""])).iloc[0]
            sl_pct = latest_ticker_row.get("stop_loss_pct", pd.Series([0])).iloc[0]
            tp_pct = latest_ticker_row.get("take_profit_pct", pd.Series([0])).iloc[0]
            
            # Formulate the strategy
            st.markdown(f"**Buy Strategy**: With a score of **{score_val}** and an upside probability of **{upside_prob*100:.1f}%**, the recommended entry style is **{entry}**. "
                        "Scale in gently if it is a breakout, or buy at support if it's a pullback.")
            st.markdown(f"**Sell Strategy**: Set a strict stop loss at **-{sl_pct*100:.1f}%** from your entry price to protect capital. "
                        f"Consider taking profits around **+{tp_pct*100:.1f}%** or trailing your stop loss as the price rises to lock in gains. "
                        "Re-evaluate your position if the company's fundamentals change or if broader market conditions deteriorate.")

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
            df_res = format_pct_columns(
                df_res,
                ["prob_upside", "scenario_bear", "scenario_base", "scenario_bull"],
            )
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
            bucket_stats[c] = pd.to_numeric(bucket_stats[c], errors="coerce")
            bucket_stats[c] = (bucket_stats[c] * 100).round(1)
        st.markdown("**Score bucket outcomes**")
        st.dataframe(bucket_stats, use_container_width=True)

        calibration = model["calibration"].copy()
        calibration["actual_hit_rate"] = pd.to_numeric(calibration["actual_hit_rate"], errors="coerce")
        calibration["actual_hit_rate"] = (calibration["actual_hit_rate"] * 100).round(1)
        st.markdown("**Calibration table**")
        st.dataframe(calibration, use_container_width=True)

with tabs[6]:
    st.subheader("❓ Help & Documentation")
    st.markdown('''
    ### Pages Overview
    * **Overview:** Displays the results of the latest scheduled scan, highlighting top picks, their financial metrics, and execution guidance.
    * **History:** Browse past scans to track how Argus scores and market conditions have trended over time.
    * **Ticker Detail:** Deep dive into a specific stock. Shows historic price charts, Argus score trends, and actionable buy/sell strategies.
    * **Journal:** A customized logbook to track your trading decisions, entry points, and rationales for future review.
    * **Manual Run:** Perform an immediate, on-demand scan of the market using your current sidebar parameters.
    * **Prediction Model:** Reviews the performance, hit rate, and calibration of the Argus AI prediction model to gauge its accuracy.

    ---
    ### Sidebar Settings Explained

    **Parameters**
    * **Minimum Score:** The minimum Argus score required for a stock to pass the screen. 
        * *Default:* 65 (Filters out weaker stocks while providing enough choices).
        * *Alternative:* 80 for stricter quality, 50 for a broader, exploratory search.
    * **Price Floor ($):** The lowest acceptable stock price. 
        * *Default:* $2.00 (Avoids highly illiquid or volatile penny stocks).
        * *Alternative:* $10.00 to focus purely on established companies.
    * **Volume Floor:** Minimum average daily trading volume.
        * *Default:* 500,000 (Ensures you can easily enter and exit positions without price slippage).
        * *Alternative:* 1,000,000+ for large caps, 100,000 for discovering niche small caps.
    * **Universe Size:** How many top market cap tickers to scan.
        * *Default:* 200 (Balances comprehensive coverage with fast scan times).
    
    **Prediction Model Focus**
    * **Horizon Days:** The timeframe over which the model predicts price movement. 
        * *Default:* 63 days (Roughly 1 quarter, a sweet spot for swing trading trends).
    * **Target Return (%):** The profit percentage the model evaluates against.
        * *Default:* 10% (A realistic expectation for a strong quarter).
    
    **Risk Rules**
    * **Risk per Trade:** The maximum percentage of your total portfolio you are willing to lose if the stock drops to your stop loss.
        * *Default:* 0.75% (A standard, safe capital preservation rule).
        * *Alternative:* 0.25% for conservative trading, up to 2.0% for aggressive scaling.
    * **Max Position Size:** The maximum allocation of your portfolio that can be tied up in a single stock.
        * *Default:* 8.0% (Forces diversification across at least 12-15 stocks).
    ''')

with tabs[7]:
    st.subheader("🤖 AI Prompts for Research")
    st.caption("Copy these templates and paste them into AI tools (like Perplexity, ChatGPT, or Claude) to deeper analyze tickers.")

    st.markdown("#### 1. Fundamental & Deep Research")
    st.code("I am researching the stock [TICKER]. Please provide a comprehensive overview of its business model, primary revenue streams, recent financial performance (last two earnings reports), and any significant regulatory or macroeconomic headwinds it is currently facing.", language="text")

    st.markdown("#### 2. Buy & Sell Strategy Validation")
    st.code("I am considering taking a position in [TICKER]. Given the current macroeconomic climate and the stock's recent price action, what are the most critical support and resistance levels to watch? Suggest a logical stop-loss percentage and a realistic price target for a 3-month hold.", language="text")

    st.markdown("#### 3. Upcoming Catalysts & Speculations")
    st.code("What are the major upcoming catalysts, product launches, clinical trials, or earnings reports for [TICKER] in the next 3 to 6 months? How might these events impact the stock price, and what are the current market speculations or analyst sentiments surrounding them?", language="text")

    st.markdown("#### 4. Competitive Moat Analysis")
    st.code("Analyze the competitive landscape for [TICKER]. Who are its top 3 direct competitors? Compare [TICKER]'s market share, unique competitive advantages (moat), and profit margins against these competitors. Is [TICKER] gaining or losing ground?", language="text")
