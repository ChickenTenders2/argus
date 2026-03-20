# Argus

Argus is a personal stock screening and analysis system built to identify small and mid-cap stocks with multi-bagger potential over a 3-year horizon.

It runs a daily quantitative scan across a universe of stocks, scores each one against a multi-factor model (fundamentals, valuation, momentum, and “smart money” signals), and sends the highest-scoring tickers to Telegram each morning.

The core philosophy is that repeated flags carry increasing weight: a stock that surfaces multiple days in a row is signaling strengthening conviction across multiple dimensions, and can justify increased research priority and (optionally) scaled position sizing.

## What’s in this repo

- `argus.py`
	- Builds a ticker universe (Russell 2000 via IWM holdings when available)
	- Scores tickers using `yfinance`
	- Sends a daily Telegram message with the top picks
	- Updates `argus_memory.csv` to track repeated flags
	- Sends a watchlist “daily change” update from `argus_watchlist.csv`
- `fmp_fetch.py`
	- Optional “enrichment” step using Financial Modeling Prep (FMP)
	- Runs only for **HIGH CONVICTION** picks and only if `FMP_API_KEY` is set
- `argus_memory.csv`
	- Persistent memory of tickers previously flagged by Argus
- `argus_watchlist.csv`
	- Your manual watchlist (tickers you always want a daily update on)
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
	- Financial Modeling Prep API key for enrichment messages

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
	 - Final fallback: a hardcoded list
	 - Note: the script currently limits the universe to the first ~300 parsed tickers for speed.

2. **Score each ticker** using `yfinance` metadata and 6-month price/volume history.

3. **Select the top picks**
	 - Tiering:
		 - **🟢 HIGH CONVICTION**: score $\ge 80$
		 - **🟡 WATCHLIST**: score $\ge 65$
	 - Sends only the **top 10** picks (by score) by default.

4. **Update memory**
	 - For tickers in the day’s top list, increment `times_flagged` in `argus_memory.csv`.
	 - The Telegram message includes a note when a ticker has been previously flagged.

5. **Optional: FMP enrichment**
	 - If `FMP_API_KEY` is set, Argus sends a *separate* Telegram message per HIGH CONVICTION ticker, including:
		 - revenue growth, gross margin, FCF yield, EBITDA growth
		 - “asset vs EBITDA” disqualifier check
		 - insider buying (last 90 days)
		 - institutional ownership (best effort, depends on FMP tier)
		 - next earnings date + recent analyst grade changes

6. **Watchlist update**
	 - Sends a second Telegram message summarizing daily % change for tickers in `argus_watchlist.csv`.

## The scoring model (as implemented)

The model in `argus.py` is a points-based heuristic, intended to surface candidates for *research* (not to be used as investment advice).

### Factors

- **Fundamentals (max 50 pts)**
	- Revenue growth
	- Gross margin
	- Positive free cash flow
- **Valuation (max 15 pts)**
	- PEG ratio
- **Momentum (max 30 pts)**
	- 6-month momentum
	- price vs 50/200 moving averages
	- unusual volume spike
- **Smart money (max 20 pts)**
	- Lower institutional ownership scores higher (early-stage “not crowded” signal)
- **Market cap (max 10 pts)**
	- Prefer small/mid caps (roughly $50M–$10B)

### Red-flag vetoes

If any of the following trigger, the ticker is excluded (returns `None`):

- extreme debt (`debtToEquity > 500`)
- extreme short interest (`shortPercentOfFloat > 45%`)
- dilution risk (`sharesPercentSharesOut > 15%`)

## State files

### Memory: `argus_memory.csv`

Tracks repeated flags:

- `ticker`
- `first_seen` (e.g. `15 Mar 2026`)
- `times_flagged` (increments when the ticker appears in the day’s top list)
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

## Argus Analysis Template (research workflow)

When a ticker is flagged, Argus is designed to hand off to a structured research write-up. The repo doesn’t automate this portion yet, but this template is intended to keep the process consistent.

### 1) Snapshot

- Ticker / date flagged / score / times flagged
- Market cap / business model / “why now” in one sentence

### 2) Moat & business quality

- What makes this business hard to copy?
- Evidence: pricing power, switching costs, network effects, distribution, brand, proprietary tech, regulatory advantage

### 3) Bull case (3-year)

- Key drivers that could compound fundamentals
- What would have to be true for a multi-bagger outcome?

### 4) Bear case / failure modes

- The 2–3 most plausible ways this breaks
- What signals would prove the thesis wrong?

### 5) Catalysts & timeline

- Upcoming earnings dates
- Product launches, regulatory events, contract awards, macro tailwinds
- What can change sentiment in the next 1–3 quarters?

### 6) Empirical cross-check (multibagger predictors)

Cross-check against the predictors you’re using from your multibagger study (2009–2024):

- FCF yield and/or path to sustainable FCF
- improving operating leverage / margins
- “not crowded” ownership
- insider buying
- clean balance sheet and dilution profile

### 7) Decision & sizing framework

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

## Notes & limitations

- `yfinance` fields can be missing or inconsistent (especially for small caps); the scorer is best-effort.
- The universe builder uses multiple sources and may fall back depending on availability.
- Telegram messages are sent using Markdown parse mode; if you add formatting, ensure it stays valid Markdown.
