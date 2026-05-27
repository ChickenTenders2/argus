# 👁 Argus Investment Workstation

Argus is a quantitative stock screener targeting Russell 2000 small-caps and Nasdaq micro-caps. It identifies explosive runner candidates (3–5× plays inspired by ONDS, RKLB, ASTS) by scoring every ticker across fundamentals, momentum, smart money, and catalyst signals — then posts the best picks to Telegram three times a day.

---

## 🌟 Core Capabilities

1. **Quantitative Scoring Engine** — 0–100 composite score across six components. Regime multiplier (×0.65–1.15) adjusts for market conditions. MIN_SCORE gate checked against the pre-regime quality score so it stays reachable in bear markets.
2. **Market Regime Filter** — Bull / Bear / Neutral / Extreme Fear / Stagflation classification from SPY moving averages, VIX, HY credit spreads, and IWM/QQQ breadth.
3. **Macro Dashboard** — Yield curve (FRED T10Y2Y / yfinance fallback), CPI trend (FRED CPIAUCSL, monthly), Fed Funds Rate (FRED DFF, daily), and a multi-factor equity Fear & Greed index (VIX + SPY momentum + safe-haven demand + HY spread — no crypto data).
4. **Catalyst Detection** — EDGAR Form 4 cluster insider buys, Form 8-K event detection, Groq LLM multi-label catalyst classifier, options flow IV rank.
5. **Runner Pattern Matching** — Mahalanobis similarity vs 8 curated pre-run profiles (ONDS, SOUN, RKLB, ASTS, IONQ, HIMS, DAVE, CRDO).
6. **ML Prediction Model** — XGBoost classifier trained on accumulated scan history. Activates automatically after ~30 matured samples (~1–2 months of daily scans).
7. **Portfolio Journal** — BUY/SELL/TRIM/SCALE_IN logging with live P&L, SPY benchmark, sector pie chart, and Markowitz portfolio optimizer.
8. **Telegram Alerts** — 3× daily GitHub Actions scans (07:00, 10:00, 17:30 ET) push HC picks to Telegram automatically.

---

## 🚀 UI Layout (5-tab design)

| Tab | What it shows |
|---|---|
| **Alerts** | Latest scan picks with scan timestamp, market regime strip, macro signal cards (yield curve, CPI, Fed Funds, Fear & Greed), hero Top 3 panel, portfolio alerts |
| **Deep Dive** | Per-ticker analysis — price + score chart, last scan findings (score waterfall + catalyst pills + reasons), financial snapshot, buy/sell strategy, AI thesis (Groq) |
| **History** | Full scan history with run-type filter, day picker, score trend chart, unit sizing |
| **Journal** | Portfolio logbook — live P&L, SPY benchmark, sector exposure, portfolio optimizer |
| **Tools** | Manual Scan · ML Model diagnostics · Docs & AI research prompts |

### Sidebar (always visible)
- **Global Preset** — one-click strategy configuration
- **💷 Capital & Sizing** (collapsible) — Unit Value, Portfolio Size, Daily Unit Cap
- **Send to Telegram** — opt-in for manual scan alerts
- **Auto-refresh** — 15-min page reload

---

## 📊 Scoring System (0–100)

| Component | Max pts | Key signals |
|---|---|---|
| Fundamentals | 18 | Revenue growth, ROCE, gross margin, FCF. Pre-revenue alt path if TTM rev <$5M |
| Valuation | 6 | PEG, P/S |
| Momentum | 34 | Tiered RS vs IWM, price performance, volume spike, 1.5× vol bonus |
| Smart Money | 22 | Institutional ownership (undiscovered), float <5M, short squeeze setup |
| Catalyst | 15 | EDGAR insider cluster, 8-K events, LLM tags, options flow |
| Persistence | 5 | Time-decayed: `min(5, times_flagged × 1.5 × 0.95^days_since_last_seen)` |

**Score velocity** — difference between current final score and last scan's final score. Displayed as ↑/↓ badge on cards. Bug fix (May 2026): velocity is now computed after all components and the regime multiplier are applied, so the badge reflects a true final-vs-final comparison.

**Regime multiplier** (×0.65–1.15) applied to display score only. MIN_SCORE gate uses the pre-regime quality score.

**Signal labels:**

| Score | Label |
|---|---|
| ≥ 85 | 🟢 Strong Buy |
| ≥ 75 | 🔵 Buy |
| ≥ 65 | 🟡 Moderate Buy |
| ≥ 55 | ⚪ Hold / Watch |
| < 55 | 🔴 Avoid |

---

## 📡 Macro Indicators

| Indicator | Source | Update frequency |
|---|---|---|
| Market Regime | yfinance SPY + VIX | Every 30 min (app cache) |
| Yield Curve | FRED T10Y2Y / yfinance ^TNX-^IRX fallback | Daily (1-day FRED lag) |
| CPI Trend | FRED CPIAUCSL (annualised 3-month) | Monthly (BLS release) |
| Fed Funds Rate | FRED DFF (daily rate) | Daily |
| Fear & Greed (Equity) | VIX + SPY momentum + SPY vs TLT + HY spread | Daily |

**Fear & Greed is equity-specific** — four equally-weighted components derived from stock market data only. The crypto alternative.me index is not used.

---

## 🔧 Setup

### Requirements
```bash
pip install -r requirements.txt
```

### Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `TELEGRAM_TOKEN` | ✅ For alerts | Telegram bot token |
| `TELEGRAM_CHAT_ID` | ✅ For alerts | Telegram chat ID |
| `GROQ_API_KEY` | ✅ For AI thesis & catalyst LLM | Groq API key |
| `DATABASE_URL` | Optional | Supabase PostgreSQL (SQLite used locally if absent) |
| `FMP_API_KEY` | Optional | Deeper fundamentals, financial snapshot fallback |
| `FRED_API_KEY` | Optional | Yield curve, CPI, Fed Funds Rate. Free at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html). Without it, yield curve falls back to yfinance. |
| `FINNHUB_API_KEY` | Optional | EDGAR fallback for Streamlit Cloud (EDGAR blocked on shared IPs) |
| `ARGUS_WEIGHTS_FILE` | Optional | `default` (balanced) or `runner` (momentum/catalyst-heavy) |
| `OPTIONS_FLOW_PROVIDER` | Optional | `barchart_free` (default) \| `market_chameleon` \| `unusual_whales` |
| `UW_API_KEY` | Optional | Unusual Whales dark-pool data ($48/mo) |

### Run locally
```bash
streamlit run app.py
```

---

## 📅 Daily Scan Schedule (GitHub Actions)

| Cron (UTC) | ET time | Profile | Description |
|---|---|---|---|
| `0 11 * * 1-5` | ~07:00 ET | `premarket` | HC picks only (score ≥75), Telegram alert |
| `0 14 * * 1-5` | ~10:00 ET | `full` | Full universe scan, all picks, Telegram alert |
| `30 21 * * 1-5` | ~17:30 ET | `postclose` | Full scan + portfolio monitor, memory update |

The Alerts tab shows the scan timestamp and run type above the picks so you always know when the data was last refreshed.

---

## 🗄️ Data Persistence

| File | Git tracked | Updated by | Purpose |
|---|---|---|---|
| `data/argus_results.csv` | ✅ | GitHub Actions | Latest scan (most recent run only) |
| `data/argus_results_history.csv` | ✅ | Both | Full scan history across all runs |
| `data/argus_feature_history.csv` | ✅ | Both | ML training feature store |
| `data/argus_memory.csv` | ✅ | Both | Ticker persistence memory (times_flagged, last_score) |
| `argus_alerts_log.txt` | ✅ | Both | Telegram alert history |
| `argus.db` | ❌ gitignored | Local only | Journal entries + local DB (never overwritten by CI) |

---

## 🧩 Project Structure

```
core/
  engine.py           — Scoring engine, universe fetch, ML model, regime filter
fetchers/
  macro_data.py       — FRED macro, equity Fear & Greed, IWM/QQQ breadth
  edgar_fetch.py      — Form 4 cluster insider detection, Form 8-K fetcher
  fmp_fetch.py        — FMP fundamentals, float, IPO calendar
  catalysts.py        — Unified catalyst score aggregator (0–15 pts)
  options_flow.py     — Options flow provider abstraction
  llm.py              — Groq multi-label catalyst classifier + AI thesis
analysis/
  pattern_match.py    — Mahalanobis runner similarity (8 curated profiles)
  backtest.py         — Backtest harness vs yfinance forward returns
ui/
  ui_components.py    — Score waterfall, catalyst pills, sparkline, heatmap
  theme.py            — Dark fintech theme, ENHANCED_CSS
config/
  weights_default.json / weights_runner.json — Scoring weight profiles
app.py                — Streamlit frontend (5-tab UI)
argus.py              — CLI entry point, Telegram dispatch, --profile flag
scan_profiles.py      — ScanProfile dataclasses for each daily slot
```

---

## ⚠️ Disclaimer

Argus is an automated research workstation for educational and analytical purposes only. Scores, regime signals, and AI-generated text are not financial advice. Always conduct your own research before deploying capital.
