import yfinance as yf
import pandas as pd
import logging
import os
import json
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger("Argus.Engine")

CACHE_FILE = "metadata_cache.json"

def get_stock_info(ticker):
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                cache = json.load(f)
        else:
            cache = {}

        date_str = datetime.now().strftime("%Y-%m-%d")

        if ticker in cache and cache[ticker].get("date") == date_str:
            return cache[ticker]["data"]

        info = yf.Ticker(ticker).info
        cache[ticker] = {"date": date_str, "data": info}
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)
        return info
    except Exception as e:
        logger.warning(f"Failed to fetch or cache info for {ticker}: {e}")
        try:
            return yf.Ticker(ticker).info
        except:
            return {}

@dataclass(frozen=True)
class Config:
    TELEGRAM_TOKEN: str = os.environ.get("TELEGRAM_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.environ.get("TELEGRAM_CHAT_ID", "")
    MEMORY_FILE: str = "argus_memory.csv"
    WATCHLIST_FILE: str = "argus_watchlist.csv"
    MIN_SCORE: int = 65
    TOP_N: int = 10
    PRICE_FLOOR: float = 2.0
    VOL_FLOOR: int = 200000

def load_memory(filepath):
    try:
        return pd.read_csv(filepath)
    except:
        return pd.DataFrame(columns=["ticker", "first_seen", "times_flagged", "last_score"])

def save_memory(df, filepath):
    df.to_csv(filepath, index=False)

def _check_red_flags(info, stock):
    flags = []
    if (info.get("debtToEquity", 0) or 0) > 500:
        flags.append("Extreme debt")
    if (info.get("shortPercentOfFloat", 0) or 0) > 0.45:
        flags.append("High short interest")
    if (info.get("sharesPercentSharesOut", 0) or 0) > 0.15:
        flags.append("Dilution risk")
    try:
        earnings = stock.calendar
        if not earnings.empty and 'EPS Estimate' in earnings.columns and 'Reported EPS' in earnings.columns:
            last_est = earnings['EPS Estimate'].dropna().iloc[-1]
            last_rep = earnings['Reported EPS'].dropna().iloc[-1] if len(earnings['Reported EPS'].dropna()) > 0 else 0
            if last_est > 0 and last_rep / last_est < 0.90:
                flags.append("Earnings miss")
    except:
        pass
    return flags

def _score_fundamentals(info, stock):
    """Score: revenue growth (max 25), ROCE (max 8), gross margin (max 10), FCF (max 8) = 51 pts max."""
    score, reasons = 0, []

    # ── Revenue Growth ──────────────────────────────────────
    try:
        financials = stock.quarterly_financials
        if "Total Revenue" in financials.index and financials.shape[1] >= 5:
            rev_now  = financials.loc["Total Revenue"].iloc[:4].sum()
            rev_prev = financials.loc["Total Revenue"].iloc[4:8].sum()
            growth   = (rev_now - rev_prev) / abs(rev_prev) if rev_prev != 0 else 0
        else:
            growth = info.get("revenueGrowth", 0) or 0
    except:
        growth = info.get("revenueGrowth", 0) or 0

    if growth >= 0.30:   score += 25; reasons.append(f"Rev growth {growth*100:.0f}%")
    elif growth >= 0.20: score += 15; reasons.append(f"Rev growth {growth*100:.0f}%")

    # ── ROCE ────────────────────────────────────────────────
    roce = info.get("returnOnCapitalEmployed", info.get("returnOnCapital", 0)) or 0
    if roce > 0.20: score += 8; reasons.append(f"ROCE {roce*100:.0f}%")

    # ── Gross Margin ────────────────────────────────────────
    # Strongest single predictor of durable competitive advantage.
    # >60% = pricing power / platform business (max pts)
    # >40% = solid margin profile
    gross_margin = info.get("grossMargins", 0) or 0
    if gross_margin > 0.60:   score += 10; reasons.append(f"Gross margin {gross_margin*100:.0f}%")
    elif gross_margin > 0.40: score += 5;  reasons.append(f"Gross margin {gross_margin*100:.0f}%")

    # ── Free Cash Flow ──────────────────────────────────────
    # Positive FCF separates real businesses from burn-rate stories.
    try:
        cf = stock.quarterly_cashflow
        if "Free Cash Flow" in cf.index:
            ttm_fcf = cf.loc["Free Cash Flow"].iloc[:4].sum()
        elif "Operating Cash Flow" in cf.index and "Capital Expenditure" in cf.index:
            ttm_fcf = (cf.loc["Operating Cash Flow"].iloc[:4].sum()
                       - abs(cf.loc["Capital Expenditure"].iloc[:4].sum()))
        else:
            ttm_fcf = info.get("freeCashflow", None)
    except:
        ttm_fcf = info.get("freeCashflow", None)

    if ttm_fcf is not None and ttm_fcf > 0:
        score += 8; reasons.append("FCF positive")

    return score, reasons

def _score_momentum(hist, ticker):
    """Score: 6mo momentum (10), 50MA (5), 200MA (5), volume spike (10), IWM RS (10) = 40 pts max."""
    score, reasons = 0, []
    if hist.empty or len(hist) < 50:
        return score, reasons

    price_now  = hist["Close"].iloc[-1]
    price_6mo  = hist["Close"].iloc[0]
    ma50       = hist["Close"].rolling(50).mean().iloc[-1]
    vol_avg    = hist["Volume"].rolling(30).mean().iloc[-1]
    vol_today  = hist["Volume"].iloc[-1]

    # 6-month price momentum
    if price_now > price_6mo * 1.15:
        score += 10; reasons.append("6mo Momentum")

    # Price above 50-day MA
    if price_now > ma50:
        score += 5; reasons.append("Above 50MA")

    # Price above 200-day MA (trend health check)
    if len(hist) >= 200:
        ma200 = hist["Close"].rolling(200).mean().iloc[-1]
        if price_now > ma200:
            score += 5; reasons.append("Above 200MA")
    else:
        logger.debug(f"{ticker}: fewer than 200 days of history — 200MA check skipped")

    # Unusual volume spike
    if vol_today > vol_avg * 2:
        score += 10; reasons.append("⚡ Volume spike")

    # Relative Strength vs IWM (Russell 2000 benchmark)
    # +10 pts if the stock has outperformed IWM by 20%+ over the last 3 months.
    try:
        # Use the last ~63 trading days as a proxy for 3 months
        if len(hist) >= 63:
            stock_3mo_return = (price_now / hist["Close"].iloc[-63]) - 1

            iwm_hist = yf.Ticker("IWM").history(period="3mo")
            if not iwm_hist.empty:
                iwm_return = (iwm_hist["Close"].iloc[-1] / iwm_hist["Close"].iloc[0]) - 1
                rs_delta   = stock_3mo_return - iwm_return
                if rs_delta >= 0.20:
                    score += 10; reasons.append(f"RS vs IWM +{rs_delta*100:.0f}%")
    except Exception as e:
        logger.debug(f"{ticker}: IWM relative strength check failed — {e}")

    return score, reasons

def score_stock(ticker, memory_df, config):
    """The central scoring logic used by both Scanner and Streamlit."""
    try:
        stock = yf.Ticker(ticker)
        info  = get_stock_info(ticker)
        if not info or 'symbol' not in info:
            return None

        red_flags = _check_red_flags(info, stock)
        if red_flags:
            return None

        score, reasons = 0, []

        # 1. Fundamentals (rev growth + ROCE + gross margin + FCF)
        f_score, f_reasons = _score_fundamentals(info, stock)
        score += f_score; reasons.extend(f_reasons)

        # 2. Valuation
        peg = info.get("pegRatio", 2)
        if 0 < peg < 1: score += 15; reasons.append(f"PEG {peg:.1f}")
        ps = info.get("priceToSalesTrailing12Months", 10)
        if 0 < ps < 4: score += 10; reasons.append(f"P/S {ps:.1f}x")

        # 3. Momentum (6mo, 50MA, 200MA, volume spike, IWM RS)
        hist = stock.history(period="1y")  # 1y gives enough data for 200MA
        m_score, m_reasons = _score_momentum(hist, ticker)
        score += m_score; reasons.extend(m_reasons)

        # 4. Smart Money & Cap
        inst_own = info.get("heldPercentInstitutions", 1) or 1
        if inst_own < 0.40: score += 20; reasons.append(f"Inst. {inst_own*100:.0f}%")
        mkt_cap = info.get("marketCap", 0) or 0
        if 50e6 < mkt_cap < 10e9: score += 10; reasons.append(f"Cap ${mkt_cap/1e6:.0f}M")

        # 5. Persistence Bonus
        prev = memory_df[memory_df["ticker"] == ticker]
        if not prev.empty and int(prev["times_flagged"].values[0]) >= 3:
            score += 5
            reasons.append("\U0001f501 Persistence bonus")

        if score < config.MIN_SCORE:
            return None

        return {
            "ticker":  ticker,
            "sector":  info.get("sector", "Unknown"),
            "score":   score,
            "tier":    "\U0001f7e2 HIGH CONVICTION" if score >= 80 else "\U0001f7e1 WATCHLIST",
            "reasons": reasons,
            "price":   round(hist["Close"].iloc[-1], 2) if not hist.empty else 0,
            "mkt_cap": f"${mkt_cap/1e6:.0f}M"
        }
    except Exception:
        return None

def get_universe():
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    # Method 1: Try iShares IWM holdings CSV
    try:
        url = "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund"
        df = pd.read_csv(url, skiprows=9, on_bad_lines='skip')
        tickers = df.iloc[:, 0].dropna().tolist()
        tickers = [str(t).strip() for t in tickers if isinstance(t, str) and 1 < len(str(t).strip()) < 6 and str(t).strip().isalpha()]
        if len(tickers) > 50:
            logger.info(f"IWM download success: {len(tickers)} tickers")
            return tickers[:300]
        raise ValueError("Too few tickers parsed")
    except Exception as e:
        logger.warning(f"IWM download failed: {e}")

    # Method 2: Try Wikipedia Russell 2000 component list
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Russell_2000_Index")
        for table in tables:
            for col in table.columns:
                if "ticker" in str(col).lower() or "symbol" in str(col).lower():
                    tickers = table[col].dropna().tolist()
                    tickers = [str(t).strip() for t in tickers if 1 < len(str(t).strip()) < 6]
                    if len(tickers) > 50:
                        logger.info(f"Wikipedia success: {len(tickers)} tickers")
                        return tickers[:300]
        raise ValueError("No ticker column found")
    except Exception as e:
        logger.warning(f"Wikipedia failed: {e}")

    # Method 3: Try downloading IWM holdings via yfinance
    try:
        iwm = yf.Ticker("IWM")
        holdings = iwm.funds_data.top_holdings
        if holdings is not None and len(holdings) > 10:
            tickers = holdings.index.tolist()
            tickers = [str(t).strip() for t in tickers if 1 < len(str(t).strip()) < 6]
            logger.info(f"yfinance IWM holdings: {len(tickers)} tickers")
            return tickers
        raise ValueError("No holdings data")
    except Exception as e:
        logger.warning(f"yfinance IWM failed: {e}")

    # Method 4: Hardcoded fallback
    logger.error("All sources failed — using hardcoded fallback list")
    return [
        "ASTS", "RKLB", "LUNR", "ACHR", "JOBY", "IRDM", "SPIR", "PL",
        "BBAI", "SOUN", "IONQ", "ARQT", "DAVE", "RELY", "URGN", "CIFR",
        "IDAI", "IREN", "BTDR", "CLBT", "AMPX", "NRDS", "SEAT", "MAPS",
        "HOOD", "AFRM", "UPST", "OPEN", "LPRO", "PAYO",
        "S", "QLYS", "RDWR", "EVTC", "OSPN",
        "HIMS", "CLOV", "GDRX", "TALK", "GRAL", "MDXG", "NUVL",
        "JANX", "KROS", "RXRX", "BEAM", "EDIT", "NTLA",
        "STEM", "NKLA", "BLNK", "CHPT", "SHLS", "ARRY", "FLNC",
        "CTOS", "LQDT", "RCUS", "ORIC", "PRAX",
        "SERV", "LIDR", "OUST", "AEVA", "MVIS", "KOPN", "XPOF",
        "CELH", "VNCE", "VSCO", "LOVE", "CURV", "BURL"
    ]
