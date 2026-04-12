# 👁 Argus Investment Workstation

Argus is an advanced quantitative stock-screening and machine-learning-driven portfolio management system. It automatically scores, filters, and ranks U.S. equities based on fundamental, technical, and macroeconomic rules, applying an AI layer (XGBoost + LLM Qualitative Analysis) to assign upside probabilities and suggest actionable position sizes.

The project is designed to run automatically via local chron or GitHub Actions, saving history over time to build out a robust, personalized prediction model and market regime analyzer.

---

## 🌟 Core Phases & Capabilities

1. **Phase 1: ML Prediction Model (XGBoost)**
   - Learns from accumulating scheduled scans (feature snapshots).
   - Generates forward predictions (e.g., probability of hitting +10% in 63 days) based on Brier scoring and historical hit rates.
2. **Phase 2: LLM Qualitative Analysis (Groq Llama 3.1)**
   - Generates qualitative, actionable AI investment theses.
   - Cross-references the quantitative Argus score and fundamental reasons with the latest `yfinance` company news.
3. **Phase 3: Macro Market Regime Filter**
   - Dynamically categorizes the broader market into Bull, Bear, Neutral, or Extreme Fear regimes based on the S&P 500 (SPY) moving averages and the ^VIX.
   - Multiplies and adjusts individual stock scores to protect capital in bear markets and press advantages in bull markets.
4. **Phase 4: Portfolio Optimizer & Sizing**
   - Uses 1-year historical correlations for the highest conviction tickers to compute Mean-Variance optimizations.
   - Outputs dynamic position sizing percentages for Max Sharpe Ratio and Minimum Volatility portfolios.
   - Auto-calculates strict Take-Profit and Stop-Loss limits per trade.

---

## 🚀 Features

*   **Sleek Multi-Tab Dashboard:** Built entirely in Streamlit with `streamlit-shadcn-ui` components and `plotly` interactive charts for professional-grade navigation, metric cards, and scalable data tables.
*   **Quantitative Scoring:** Ranks tickers (0-100) using a multi-factor `get_score()` algorithm focusing on revenue growth, margins, technical momentum (50MA, 200MA), and relative strength vs IWM/SPY.
*   **Execution Strategy:** Clear "Buy/Sell" text generation with precise stop loss/take profit recommendations based on recent price volatility.
*   **Journal & History:** Built-in logging to write down entry prices and review past performances.
*   **Telegram Alerts:** Opt-in automated push notifications sending the top tier 'High Conviction' tickers straight to your phone.

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

1. **Overview:** Daily snapshot of your most recent scan outputs and current Macro Market Regime.
2. **Ticker Detail:** Deep dive into Plotly charts, quantitative execution guidance, and generate qualitative AI investment thesis reports.
3. **Portfolio Optimizer:** Select multiple tickers from the recent scan to calculate Max Sharpe allocations using Markowitz efficient frontier logic.
4. **Manual Run:** Instantly trigger a new global algorithmic scan across the market using your sidebar settings.
5. **History & Journal:** Review historic databases to see how the engine's ratings mature over time, and log your personal swing trades.
6. **Prediction Model:** ML diagnostics reviewing XGBoost hit rates across thousands of accumulated past predictions.

---

## ⚠️ Disclaimer

**Argus** is an automated research workstation intended for educational and analytical purposes only. Stock scores, regime multipliers, portfolio optimizations, and AI-generated text are not financial advice. Algorithms (including XGBoost and LLMS) can make incorrect assumptions. Always conduct your own research before deploying capital in the stock market.