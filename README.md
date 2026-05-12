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

*   **Sleek Multi-Tab Dashboard:** Built entirely in Streamlit featuring a **Visual Card Grid** UI, interactive AgGrid history tables, annotated metric badges, and styled metric cards for professional-grade navigation and data consumption.
*   **Quantitative Scoring:** Ranks tickers (0–100) using a multi-factor `score_stock()` algorithm across five components: Fundamentals (35 pts), Valuation (10 pts), Momentum (30 pts), Smart Money (20 pts), Persistence (5 pts). Score is regime-adjusted (Bull ×1.05 → Bear ×0.7) before the minimum-score gate is applied. Signal labels: 🟢 Strong Buy ≥85 · 🔵 Buy ≥75 · 🟡 Moderate Buy ≥65 · ⚪ Hold ≥55 · 🔴 Avoid <55.
*   **Market Regime Panel:** Overview tab displays the current regime (Bull/Bear/Neutral/Extreme Fear), live VIX level with trend arrow, SPY vs 200-day MA gap, last scan date, and automatic bear-transition warnings. A hoverable **ⓘ** icon in the regime strip defines all metrics (×mult, VIX, 50/200-day MA, macro adj) on demand.
*   **Two-Panel Price & Score Chart:** Ticker Detail tab shows a stacked Plotly chart — price on top, Argus score on bottom with colour-coded bands (🟢 ≥75, 🟡 50–75, 🔴 <50) so you immediately understand where a score sits in context.
*   **Flexible Scan Universe:** Three universe modes — **Fixed Top** (same top-weighted R2000 names each run), **Random** (different subset each run for broader coverage over time), or **Full Universe** (all ~2000 R2000 tickers, ~5–10 min). Scheduled nightly scans always run the full universe. The scan engine shuffles tickers before prefiltering so no names are systematically skipped.
*   **Deep Dive Integrations:** Seamlessly navigate to a specific ticker via table clicks and interactive callbacks to review technical setups, AI thesis, and execution guidance.
*   **Portfolio Analytics Journal:** Built-in quantitative logbook tracking Total Invested, Net Returns, SPY Benchmark Comparisons, and Sector Exposure Pie Charts. Journal entries are stored locally and are **never overwritten** by GitHub Actions.
*   **Telegram Alerts + Alerts Log:** Opt-in automated push notifications sending top 'High Conviction' tickers to your phone. The Alerts Log tab displays all historical messages, including those sent by the daily GitHub Action.
*   **AI Research Prompts:** Curated prompt templates for macro market overview, bear signal checklists, sector rotation analysis, and individual ticker deep-dives — ready to paste into ChatGPT, Claude, Perplexity, or Grok.

---

## 📂 Project Structure

*   **`engine.py`**: The core quantitative brain. Handles `yfinance` data fetching, XGBoost ML routing, the Macro Market Regime filter, the Portfolio Optimizer, and mathematical rules.
*   **`app.py`**: The Streamlit user interface featuring 6 tabs: Overview, Ticker Detail, Scans (Manual Scan + History), Journal (P&L analytics + Portfolio Optimizer), Prediction Model, and Help.
*   **`argus.py`**: A lightweight, headless script designed for executing the scan programmatically via GitHub Actions cron scheduling. It bridges results to Telegram using `llm.py`.
*   **`llm.py`**: Integrated Groq API client to process generative AI summaries based on fresh news URLs and technical data.
*   **`fmp_fetch.py`**: (Optional) Fetches deeper fundamental company data using Financial Modeling Prep if available.
*   **`macro_data.py`**: Fetches FRED macro indicators (yield curve, CPI, Fed Funds Rate) and the Alternative.me Fear & Greed Index to power the enhanced regime filter.
*   **`edgar_fetch.py`**: Fetches SEC EDGAR Form 4 insider trading data (open-market purchases) for HIGH CONVICTION picks after each scan. No API key required.

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
export GROQ_API_KEY="gsk_your_key_here"              # Required for Phase 2 Llama 3 thesis generation
export TELEGRAM_TOKEN="bot_token_from_botfather"     # Required for Phase 1/4 push alerts 
export TELEGRAM_CHAT_ID="your_telegram_id"           # Required for Phase 1/4 push alerts
export FMP_API_KEY="fmp_api_key_here"                # Optional for deeper fundamental reads
export FRED_API_KEY="your_fred_key_here"             # Optional — free at fred.stlouisfed.org (instant signup)
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

1. **Overview:** Daily snapshot of the latest scan results as interactive visual metric cards. Displays the current Macro Market Regime, FRED macro signals (yield curve, CPI, Fed Funds), live VIX level, SPY vs 200-day MA gap, last scan date, and automatic bear-transition warnings. Alerts Log (Telegram history) accessible via expander.
2. **Ticker Detail:** Deep dive into a specific ticker — two-panel stacked Plotly chart (price + Argus score with 🟢/🟡/🔴 bands), quantitative execution guidance, financial snapshot, earnings date, analyst targets, insider activity, and auto-generated Groq AI investment thesis.
3. **Scans:** On-demand **Manual Scan** at the top — choose Universe Mode (Fixed / Random / Full), set minimum score and universe size, then fire a scan with live progress. Below: full **Scan History** with run-type filter, interactive score trend chart, day-picker, and detailed card grids.
4. **Journal:** Full portfolio management hub. Log BUY/SELL/SCALE_IN/TRIM transactions, import via CSV, track open holdings with live P&L and colour-coded unrealized returns, view realized win-rate, benchmark net return against SPY, and visualize sector exposure via interactive pie chart. **Portfolio Optimizer** (Markowitz Max Sharpe / Min Volatility) accessible via expander.
5. **Prediction Model:** ML diagnostics — XGBoost hit rates, Brier score, confusion matrix, feature importance table, SHAP global impact plot, score-bucket outcome stats, and calibration table. Activates automatically once 30+ matured samples exist.
6. **Help:** Full documentation, sidebar settings reference (presets, universe modes, risk rules), ML activation timeline, and curated AI research prompt templates for macro overview, bearish checklist, sector rotation, and individual ticker deep-dives.

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