# 👁 Argus Investment Workstation

Argus is an advanced quantitative stock-screening and machine-learning-driven portfolio management system. It automatically scores, filters, and ranks U.S. equities based on fundamental, technical, and macroeconomic rules, applying an AI layer (XGBoost + LLM Qualitative Analysis) to assign upside probabilities and suggest actionable position sizes.

The project is designed to run automatically via local chron or GitHub Actions, saving history over time to build out a robust, personalized prediction model and market regime analyzer.

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
*   **Quantitative Scoring:** Ranks tickers (0-100) using a multi-factor `get_score()` algorithm focusing on revenue growth, margins, technical momentum (50MA, 200MA), and relative strength vs IWM/SPY.
*   **Market Regime Panel:** Overview tab displays the current regime (Bull/Bear/Neutral/Extreme Fear), live VIX level with trend arrow, SPY vs 200-day MA gap, last scan date, and automatic bear-transition warnings.
*   **Two-Panel Price & Score Chart:** Ticker Detail tab shows a stacked Plotly chart — price on top, Argus score on bottom with colour-coded bands (🟢 ≥75, 🟡 50–75, 🔴 <50) so you immediately understand where a score sits in context.
*   **Deep Dive Integrations:** Seamlessly navigate to a specific ticker via table clicks and interactive callbacks to review technical setups, AI thesis, and execution guidance.
*   **Portfolio Analytics Journal:** Built-in quantitative logbook tracking Total Invested, Net Returns, SPY Benchmark Comparisons, and Sector Exposure Pie Charts. Journal entries are stored locally and are **never overwritten** by GitHub Actions.
*   **Telegram Alerts + Alerts Log:** Opt-in automated push notifications sending top 'High Conviction' tickers to your phone. The Alerts Log tab displays all historical messages, including those sent by the daily GitHub Action.
*   **AI Research Prompts:** Curated prompt templates for macro market overview, bear signal checklists, sector rotation analysis, and individual ticker deep-dives — ready to paste into ChatGPT, Claude, Perplexity, or Grok.

---

## 📂 Project Structure

*   **`engine.py`**: The core quantitative brain. Handles `yfinance` data fetching, XGBoost ML routing, the Macro Market Regime filter, the Portfolio Optimizer, and mathematical rules.
*   **`app.py`**: The Streamlit user interface featuring 10 custom panels (Overview, Ticker Detail, Portfolio Optimizer, Manual Run, etc.).
*   **`argus.py`**: A lightweight, headless script designed for executing the scan programmatically via GitHub Actions cron scheduling. It bridges results to Telegram using `llm.py`.
*   **`llm.py`**: Integrated Groq API client to process generative AI summaries based on fresh news URLs and technical data.
*   **`fmp_fetch.py`**: (Optional) Fetches deeper fundamental company data using Financial Modeling Prep if available.

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
```

### 3. Running Argus Locally

Boot the dashboard on your local machine:
```bash
streamlit run app.py
```

The workstation will launch in your browser at `http://localhost:8502`.

---

## 📚 Dashboard Navigation

1. **Overview:** Daily snapshot of your most recent scan outputs, current Macro Market Regime (Bull/Bear/Neutral/Extreme Fear), live VIX level, SPY vs 200-day MA gap, last scan date, and automatic bear-transition risk alerts.
2. **Ticker Detail:** Deep dive into a two-panel price+score chart, quantitative execution guidance, and generate qualitative AI investment thesis reports via `nav_to_ticker`.
3. **Portfolio Optimizer:** Select multiple tickers from the recent scan to calculate Max Sharpe allocations using Markowitz efficient frontier logic.
4. **Manual Run:** Instantly trigger a new global algorithmic scan across the market using your sidebar settings, and evaluate **Auto-Pilot** alerts.
5. **History:** Review interactive sortable scan tables. Select any day to view a visual card grid with full scoring reasons.
6. **Journal:** Personal logbook for swing trade tracking. Entries are stored locally in `argus.db` and are never overwritten by GitHub Actions.
7. **Prediction Model:** ML diagnostics reviewing XGBoost hit rates. Activates automatically once 30+ matured samples exist (scans older than your configured Horizon Days). Typically active after 1–2 months of daily GitHub Action runs.
8. **Alerts Log:** Historical record of all Telegram push notifications from both manual runs and the GitHub Actions daily scan. Populated by `argus_alerts_log.txt` which the Action commits back to the repository.
9. **Prompts:** 5 market condition prompt templates (macro overview, bearish checklist, watchlist impact, sector rotation) plus 4 individual ticker research templates — paste into any AI tool.

---

## 🗄️ Data Persistence

| File | Tracked in Git | Updated by | Purpose |
|---|---|---|---|
| `argus_results.csv` | ✅ Yes | GitHub Action | Latest scan results |
| `argus_results_history.csv` | ✅ Yes | GitHub Action | Full scan history |
| `argus_feature_history.csv` | ✅ Yes | GitHub Action | ML feature store |
| `argus_alerts_log.txt` | ✅ Yes | Both | Telegram alert log |
| `argus.db` | ❌ No (gitignored) | Local app only | Journal + local DB |

> `argus.db` is intentionally gitignored so that journal entries on your local machine are never lost when pulling GitHub Action updates.

---

## ⚠️ Disclaimer

**Argus** is an automated research workstation intended for educational and analytical purposes only. Stock scores, regime multipliers, portfolio optimizations, and AI-generated text are not financial advice. Algorithms (including XGBoost and LLMS) can make incorrect assumptions. Always conduct your own research before deploying capital in the stock market.