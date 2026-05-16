# 👁 Argus Investment Workstation

Argus is an advanced quantitative stock-screening and machine-learning-driven portfolio management system. It automatically scores, filters, and ranks U.S. equities based on fundamental, technical, and macroeconomic rules, applying an AI layer (XGBoost + LLM Qualitative Analysis) to assign upside probabilities and suggest actionable position sizes.

The project is designed to run automatically via local cron or GitHub Actions, saving history over time to build out a robust, personalized prediction model and market regime analyzer.

---

## 🌟 Core Phases & Capabilities

1. **Phase 1: ML Prediction Model (XGBoost)**
   - Learns from accumulating scheduled scans (feature snapshots).
   - Generates forward predictions (e.g., probability of hitting +10% in 63 days) based on Brier scoring and historical hit rates.
2. **Phase 2: Live AI News Sentiment Scoring (Groq Llama 3.1)**
   - Generates qualitative, actionable AI investment theses.
   - Cross-references the quantitative Argus score and fundamentals with live `yfinance` news headlines to dynamically boost or penalize final scores (up to +/- 15 points) based on current sentiment.
3. **Phase 3: Macro Market Regime Filter**
   - Dynamically categorizes the broader market into Bull, Bear, Neutral, or Extreme Fear regimes based on the S&P 500 (SPY) moving averages and the ^VIX.
   - Multiplies and adjusts individual stock scores to protect capital in bear markets and press advantages in bull markets.
4. **Phase 4: Portfolio Optimizer & Auto-Pilot Monitor**
   - Uses 1-year historical correlations for highest conviction tickers to compute Mean-Variance optimizations.
   - **Auto-Pilot Monitor:** Tracks logged `Journal` entries against real-time market prices, raising alerts and generating Telegram notifications instantly if strict Take-Profit or Stop-Loss limits are hit.

---

## 🚀 Features

*   **Modern Fintech Dashboard:** Built entirely in Streamlit featuring a **Visual Card Grid** UI with equal-height card rows, sticky regime strip, hero Top 3 panel, sector conviction heatmap, score-velocity sparklines, and catalyst pill badges — all on a deep-slate dark theme.
*   **Quantitative Scoring:** Ranks tickers (0–100) using a multi-factor `score_stock()` algorithm across six components: Fundamentals (18 pts), Valuation (6 pts), Momentum (34 pts), Smart Money (22 pts), Catalyst (15 pts), Persistence (5 pts, time-decayed). Component caps load from `config/weights_*.json` so a "runner" weight profile (momentum 40 + catalyst 20) can be selected via `ARGUS_WEIGHTS_FILE=runner`. The MIN_SCORE gate is checked against the **pre-regime, pre-sentiment quality score** so it stays reachable in bear markets; the regime multiplier (×0.65–1.15) and the bounded sentiment delta (±10, scaled by regime) are display adjustments on top. Signal labels: 🟢 Strong Buy ≥85 · 🔵 Buy ≥75 · 🟡 Moderate Buy ≥65 · ⚪ Hold ≥55 · 🔴 Avoid <55.
*   **Market Regime Panel:** Overview tab displays the current regime (Bull/Bear/Neutral/Extreme Fear), live VIX level with trend arrow, SPY vs 200-day MA gap, last scan date, and automatic bear-transition warnings. A hoverable **ⓘ** icon in the regime strip defines all metrics (×mult, VIX, 50/200-day MA, macro adj) on demand.
*   **Two-Panel Price & Score Chart:** Ticker Detail tab shows a stacked Plotly chart — price on top, Argus score on bottom with colour-coded bands (🟢 ≥75, 🟡 50–75, 🔴 <50) so you immediately understand where a score sits in context.
*   **Flexible Scan Universe:** Three universe modes — **Core** (IWM holdings, ~2000 R2000 tickers), **Rockets** (Nasdaq sub-$1B + recent IPOs, ~800–1200 names), or **Combined** (both merged, ~2500+). Scheduled scans always run a full pass. Live scan progress bar shows current ticker and count during any manual or auto-scan.
*   **Deep Dive Integrations:** Seamlessly navigate to a specific ticker via table clicks and interactive callbacks to review technical setups, AI thesis, and execution guidance.
*   **Portfolio Analytics Journal:** Built-in quantitative logbook tracking Total Invested, Net Returns, SPY Benchmark Comparisons, and Sector Exposure Pie Charts. Journal entries are stored locally and are **never overwritten** by GitHub Actions.
*   **Telegram Alerts + Alerts Log:** Opt-in automated push notifications sending top 'High Conviction' tickers to your phone. The Alerts Log tab displays all historical messages, including those sent by the daily GitHub Action.
*   **AI Research Prompts:** Curated prompt templates for macro market overview, bear signal checklists, sector rotation analysis, and individual ticker deep-dives — ready to paste into ChatGPT, Claude, Perplexity, or Grok.

---

## 📂 Project Structure

*   **`engine.py`**: The core quantitative brain. Handles `yfinance` data fetching, XGBoost ML routing, the Macro Market Regime filter, the Portfolio Optimizer, and mathematical rules. Exposes `run_scan(progress_fn=...)` for live UI progress callbacks.
*   **`app.py`**: The Streamlit user interface featuring 6 tabs: Overview, Scans, Ticker Detail, Journal, Prediction Model, and Docs. Includes live auto-scan progress bar, equal-height card rows, hero Top 3 panel, and sector heatmap.
*   **`argus.py`**: A lightweight, headless script designed for executing the scan programmatically via GitHub Actions cron scheduling. Accepts `--profile {premarket,full,postclose,catalyst}` and bridges results to Telegram.
*   **`theme.py`**: Design tokens and `ENHANCED_CSS` string — deep-slate dark theme, fintech typography, equal-height card flexbox rules.
*   **`ui_components.py`**: Reusable Streamlit components — sparklines, catalyst pill badges, score waterfall, hero card, sector heatmap renderer.
*   **`catalysts.py`**: Unified catalyst score aggregator (0–15 pts) combining EDGAR cluster insider buys, Form 8-K events, LLM multi-label tags, and options flow signals.
*   **`options_flow.py`**: Options flow provider abstraction — Barchart free scraper, Market Chameleon IV rank, and Unusual Whales (opt-in via `UW_API_KEY`).
*   **`scan_profiles.py`**: `ScanProfile` dataclasses for premarket / full / postclose / catalyst daily slots.
*   **`llm.py`**: Groq JSON-mode multi-label catalyst classifier — returns `{sentiment, catalysts: [...], urgency}` instead of a bare integer.
*   **`fmp_fetch.py`**: FMP fundamentals, float, short interest, IPO calendar, and S-3/424B filing detection.
*   **`macro_data.py`**: FRED macro indicators (yield curve, CPI, Fed Funds, HY spreads), IWM/QQQ breadth, Fear & Greed, small-cap realized volatility.
*   **`edgar_fetch.py`**: SEC EDGAR Form 4 cluster insider detection and Form 8-K fetcher. No API key required.
*   **`backtest.py`**: Replay harness — tests any weight vector against `argus_feature_history.csv` for 21/63/126-day forward returns.
*   **`pattern_match.py`**: Mahalanobis distance similarity vs 8 curated pre-run runner profiles (ONDS, SOUN, RKLB, ASTS, IONQ, HIMS, DAVE, CRDO).

---

## 🔧 Prerequisites & Setup

### 1. Requirements

Install the dependencies:
```bash
pip install -r requirements.txt
```

*Required Packages include:* `streamlit`, `streamlit-shadcn-ui`, `pandas`, `yfinance`, `xgboost`, `scikit-learn`, `groq`, `plotly`, `requests`.

### 2. Environment Variables

To activate the AI summaries and Telegram alerts, you must export the following keys in your terminal or store them safely in a `.env` file (or GitHub Action Secrets):

```bash
export GROQ_API_KEY="gsk_your_key_here"              # Required for LLM catalyst classifier (Groq Llama 3)
export TELEGRAM_TOKEN="bot_token_from_botfather"     # Required for Telegram push alerts
export TELEGRAM_CHAT_ID="your_telegram_id"           # Required for Telegram push alerts
export FMP_API_KEY="fmp_api_key_here"                # Optional — deeper fundamentals, float, IPO calendar
export FRED_API_KEY="your_fred_key_here"             # Optional — free at fred.stlouisfed.org
export DATABASE_URL="postgresql://..."               # Optional — Supabase PostgreSQL (SQLite used locally if absent)
export OPTIONS_FLOW_PROVIDER="barchart_free"         # Optional — barchart_free (default) | market_chameleon | unusual_whales
export UW_API_KEY="your_uw_key_here"                 # Optional — Unusual Whales API ($48/mo), enables dark-pool flow data
```

> **FRED API Key:** Free registration at [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html). When set, Argus adds yield curve, CPI trend, and Fed Funds Rate signals to the macro regime filter, and displays them as metric cards in the Overview dashboard. Without the key, all existing features continue to work unchanged.
>
> **SEC EDGAR** (insider trading): No API key required. Argus automatically fetches Form 4 insider purchase data from `data.sec.gov` for HIGH CONVICTION picks after each scan.

### 3. Running Argus Locally

Boot the dashboard on your local machine:
```bash
streamlit run app.py
```

The workstation will launch in your browser at `http://localhost:8502`.

---

## 📚 Dashboard Navigation

1. **Overview:** Daily snapshot of the latest scan results — sticky macro regime strip (IWM/QQQ ratio, HY spread chip, small-cap stress flag), hero Top 3 panel with score-velocity sparklines and catalyst pill badges, sector conviction heatmap (11 sectors × 3 score buckets), and full card grid. Auto-scan runs once per calendar day and shows a live progress bar.
2. **Scans:** Full scan history with run-type filter, interactive score trend chart, day-picker, and detailed card grids with unit-sizing and entry/SL/TP guidance.
3. **Ticker Detail:** Deep dive into a specific ticker — two-panel stacked Plotly chart (price + Argus score with 🟢/🟡/🔴 bands), quantitative execution guidance, financial snapshot, earnings date, analyst targets, insider activity, and auto-generated Groq AI investment thesis.
4. **Journal:** Full portfolio management hub. Log BUY/SELL/SCALE_IN/TRIM transactions, import via CSV, track open holdings with live P&L and colour-coded unrealized returns, view realized win-rate, benchmark net return against SPY, and visualize sector exposure via interactive pie chart. **Portfolio Optimizer** (Markowitz Max Sharpe / Min Volatility) accessible via expander.
5. **Prediction Model:** Step-by-step activation guide (auto-expands when model isn't ready) plus live status indicator. Once 30+ samples mature: XGBoost hit rates, Brier score, confusion matrix, feature importance, SHAP plots, and score-bucket calibration table.
6. **Docs:** Quick-access tools at the top (copy HC tickers, Perplexity research prompt, AI deep-dive templates). Full reference documentation (scoring system, sidebar settings, ML timeline) in a collapsed expander below.

---

## 🗄️ Data Persistence

| File | Tracked in Git | Updated by | Purpose |
|---|---|---|---|
| `argus_results.csv` | ✅ Yes | GitHub Action | Latest scan results |
| `argus_results_history.csv` | ✅ Yes | **Both** | Full scan history — written by every scan (app + CI) |
| `argus_feature_history.csv` | ✅ Yes | Both | ML feature store |
| `argus_memory.csv` | ✅ Yes | Both | Ticker persistence memory |
| `argus_alerts_log.txt` | ✅ Yes | Both | Telegram alert log |
| `argus.db` | ❌ No (gitignored) | Local app only | Journal + local DB |

> `argus.db` is intentionally gitignored so that journal entries on your local machine are never lost when pulling GitHub Action updates.
>
> `argus_results_history.csv` is now written by **every** scan (manual, auto, and GitHub Actions) so the auto-scan dedup check works correctly across Streamlit Cloud restarts and code pushes.

---

## ⚠️ Disclaimer

**Argus** is an automated research workstation intended for educational and analytical purposes only. Stock scores, regime multipliers, portfolio optimizations, and AI-generated text are not financial advice. Algorithms (including XGBoost and LLMS) can make incorrect assumptions. Always conduct your own research before deploying capital in the stock market.