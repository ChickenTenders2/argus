# where is the readme?

Here's the full README — copy everything between the triple backticks below and paste it directly into your `README.md` file in GitHub:

```markdown
# Argus

Argus is a personal stock screening and analysis system built to identify small and mid-cap stocks with multi-bagger potential over a 3-year horizon.

It runs a daily quantitative scan across a universe of stocks, scores each one against a multi-factor model (fundamentals, valuation, momentum, and "smart money" signals), and sends the highest-scoring tickers to Telegram each morning.

The core philosophy is that repeated flags carry increasing weight: a stock that surfaces multiple days in a row is signaling strengthening conviction across multiple dimensions, and can justify increased research priority and (optionally) scaled position sizing.

## What's in this repo

- `argus.py`
	- Builds a ticker universe (Russell 2000 via IWM holdings when available)
	- Batch pre-filters tickers for liquidity and price before deep scoring
	- Scores tickers using `yfinance` metadata and price/volume history
	- Sends a daily Telegram message with the top picks, diversified by sector
	- Updates `argus_memory.csv` to track repeated flags (with persistence bonus)
    - Writes `argus_results.csv` after each scan for the Streamlit dashboard to consume
	- Sends a watchlist "daily change" update from `argus_watchlist.csv`
- `fmp_fetch.py`
	- Optional "enrichment" step using Financial Modeling Prep (FMP)
	- Currently disabled — `run_fmp_enrichment` is commented out in `main()`
	- Runs only for **HIGH CONVICTION** picks and only if `FMP_API_KEY` is set
- `argus_memory.csv`
	- Persistent memory of tickers previously flagged by Argus
- `argus_watchlist.csv`
	- Your manual watchlist (tickers you always want a daily update on)
- `app.py`
    - Streamlit dashboard for manual scans and viewing nightly results
    - Auto-displays the last nightly scan results from `argus_results.csv` without any button press
    - Sidebar controls for minimum score, price floor, volume floor, and universe size
    - "Run Global Scan" button for on-demand scanning
- `argus_results.csv`
    - Written by `argus.py` after each nightly run
    - Read automatically by `app.py` to display last night's picks on the dashboard
- `.github/workflows/argus.yml`
	- GitHub Action to run Argus on a weekday schedule and commit memory updates

## Requirements

- Python 3.11+ (GitHub Action uses 3.11)
- Telegram bot token + chat ID
- Optional: Financial Modeling Prep API key for enrichment

Install dependencies:

```bash
pip install -r requirements.txt
```


## Configuration (environment variables)

Argus reads configuration from environment variables:

- `TELEGRAM_TOKEN` (required)
    - Token for your Telegram bot, e.g. `123456:ABCDEF...`
- `TELEGRAM_CHAT_ID` (required)
    - Chat ID to send messages to (your user chat, group, or channel)
- `FMP_API_KEY` (optional)
    - Financial Modeling Prep API key for enrichment messages (currently disabled)

Local example:

```bash
export TELEGRAM_TOKEN="..."
export TELEGRAM_CHAT_ID="..."
export FMP_API_KEY="..."   # optional

python argus.py
```

If `TELEGRAM_TOKEN` / `TELEGRAM_CHAT_ID` are missing, `argus.py` will error on startup.

## Daily run behavior

Each run does the following:

1. **Build the universe**
    - Primary source: iShares IWM holdings CSV (Russell 2000 ETF)
    - Fallback: Wikipedia Russell 2000 page
    - Fallback: `yfinance` fund holdings
    - Final fallback: a hardcoded list of ~70 small/mid-cap growth tickers
    - Universe is up to ~1,940 Russell 2000 components depending on source
2. **Batch pre-filter** *(new)*
    - Downloads 1 month of price/volume data for the entire universe in a single batched `yf.download()` call (threaded)
    - Drops tickers with: price < \$2, avg daily volume < 200k, or missing data
    - Caps the surviving list at **400 tickers** for deep scoring
3. **Score each ticker** using `yfinance` metadata and 6-month price/volume history
4. **Select the top picks**
    - Tiering:
        - **🟢 HIGH CONVICTION**: score ≥ 80
        - **🟡 WATCHLIST**: score ≥ 65
    - Applies **sector diversity cap**: max 3 picks per sector *(new)*
    - Sends only the **top 10** picks (by score) by default
5. **Update memory \& apply persistence bonus** *(updated)*
    - For tickers in the day's top list, increment `times_flagged` in `argus_memory.csv`
    - Tickers flagged **3+ times** receive a **+5 persistence bonus** to their score
    - The Telegram message includes a note when a ticker has been previously flagged
6. **Optional: FMP enrichment** *(disabled)*
    - Commented out. Re-enable by uncommenting `run_fmp_enrichment(results, send_telegram)` in `main()`
7. **Watchlist update**
    - Sends a second Telegram message summarizing daily % change for tickers in `argus_watchlist.csv`

## The scoring model (as implemented)

The model in `argus.py` is a points-based heuristic, intended to surface candidates for *research* (not to be used as investment advice).

### Factors

- **Fundamentals (max 50 pts)**
    - Revenue growth — calculated from quarterly financials (TTM vs prior year TTM); falls back to `yfinance` info field if data unavailable *(improved)*
    - Gross margin
    - Positive free cash flow
- **Valuation (max 25 pts)** *(expanded)*
    - PEG ratio (max 15 pts) — scores < 1 as undervalued, < 2 as fair
    - Price/Sales ratio (max 10 pts) — P/S < 4x scores positively; better suited for pre-profit small caps *(new)*
- **Momentum (max 40 pts)** *(expanded)*
    - 6-month price momentum
    - Price vs 50/200-day moving averages
    - Unusual volume spike (today vs 30-day avg)
    - Relative Strength vs IWM: 3-month return vs Russell 2000 benchmark — +10 pts if outperforming by 20%+ *(new)*
- **Smart money (max 20 pts)**
    - Lower institutional ownership scores higher (early-stage "not crowded" signal)
- **Market cap (max 10 pts)**
    - Prefer small/mid caps (roughly \$50M–\$10B)
- **Quality (max 8 pts)** *(new)*
    - ROCE (Return on Capital Employed) > 20% — strongest empirical predictor of multi-bagger outcomes
- **Persistence bonus (+5 pts)** *(new)*
    - Applied at memory-update time to tickers flagged 3+ consecutive days


### Red-flag vetoes

If any of the following trigger, the ticker is excluded (returns `None`):

- Extreme debt (`debtToEquity > 500`)
- Extreme short interest (`shortPercentOfFloat > 45%`)
- Dilution risk (`sharesPercentSharesOut > 15%`)
- Earnings miss: last reported EPS was >10% below estimate *(new)*


## State files

### Memory: `argus_memory.csv`

Tracks repeated flags:

- `ticker`
- `first_seen` (e.g. `15 Mar 2026`)
- `times_flagged` (increments when the ticker appears in the day's top list)
- `last_score`

This file is designed to be committed back to the repo by the GitHub Action.

### Watchlist: `argus_watchlist.csv`

One ticker per line:

```csv
ticker
ASTS
RKLB
...
```

Argus will send a daily watchlist update message (price + day change) for any tickers listed.

## Automation (GitHub Actions)

The workflow in `.github/workflows/argus.yml` runs:

- On a weekday schedule (`0 7 * * 1-5`, i.e. 7am UTC)
- Or manually via `workflow_dispatch`

It requires these GitHub repository secrets:

- `TELEGRAM_TOKEN`
- `TELEGRAM_CHAT_ID`
- `FMP_API_KEY` (optional)

After a successful run, it commits and pushes updates to `argus_memory.csv`.

After a successful run, it also commits `argus_results.csv` so the Streamlit dashboard automatically reflects the latest scan results without requiring a manual button press.

## Argus Analysis Template (research workflow)

When a ticker is flagged, Argus is designed to hand off to a structured research write-up. The repo doesn't automate this portion yet, but this template is intended to keep the process consistent.

### 1) Snapshot

- Ticker / date flagged / score / times flagged
- Market cap / business model / "why now" in one sentence


### 2) Moat \& business quality

- What makes this business hard to copy?
- Evidence: pricing power, switching costs, network effects, distribution, brand, proprietary tech, regulatory advantage


### 3) Bull case (3-year)

- Key drivers that could compound fundamentals
- What would have to be true for a multi-bagger outcome?


### 4) Bear case / failure modes

- The 2–3 most plausible ways this breaks
- What signals would prove the thesis wrong?


### 5) Catalysts \& timeline

- Upcoming earnings dates
- Product launches, regulatory events, contract awards, macro tailwinds
- What can change sentiment in the next 1–3 quarters?


### 6) Empirical cross-check (multibagger predictors)

- FCF yield and/or path to sustainable FCF
- Improving operating leverage / margins
- "Not crowded" ownership
- Insider buying
- Clean balance sheet and dilution profile
- ROCE > 20% *(new)*
- Relative Strength outperforming IWM *(new)*


### 7) Decision \& sizing framework

- What would justify an initial starter position?
- What would justify scaling?
- What triggers a trim or exit?


### 8) Outcome tracking

- Add the ticker to your Google Sheets tracker (entry date, thesis, catalysts, expected timeline)
- Update it when Argus re-flags the ticker or catalysts resolve


## Customization

The main knobs are at the top of `argus.py`:

- `TOP_N` (default 10)
- `MIN_SCORE` (default 65)
- `MEMORY_FILE`, `WATCHLIST_FILE`

Pre-filter tuning (in `main()`):

- Price floor: `data['Close'].iloc[-1] > 2` — raise to `> 4` to exclude more penny stocks
- Volume floor: `data['Volume'].mean() > 200000` — raise to `> 300000` for stricter liquidity
- Universe cap: `valid_tickers[:400]` — raise to `[:600]` for more picks, lower for speed


## Notes \& limitations

- `yfinance` fields can be missing or inconsistent (especially for small caps); the scorer is best-effort and uses safe fallbacks throughout.
- Revenue growth is now calculated from quarterly financials where available, falling back to the `info` field.
- ROCE field availability varies; the scorer tries `returnOnCapitalEmployed` then `returnOnCapital`.
- The IWM holdings source caps at ~300 tickers; the pre-filter then selects the best 400 from the full available universe.
- Telegram messages are sent using Markdown parse mode; if you add formatting, ensure it stays valid Markdown.
- `fmp_fetch.py` is still in the repo but its call is commented out in `main()`. It can be re-enabled at any time.

```

Copy → open `README.md` in GitHub → click edit → paste → commit. Done!```

