import sqlite3
import yfinance as yf

import pandas as pd
import logging
import os
import json
import numpy as np
from sqlalchemy import create_engine, text

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
from dataclasses import dataclass
from datetime import datetime, date

from llm import get_sentiment_score

logger = logging.getLogger("Argus.Engine")

CACHE_FILE = "metadata_cache.json"
DB_FILE = "argus.db"

_DB_ENGINE = None
_DB_INITIALIZED = False
_SQLITE_CONN = None
import threading
_DB_INIT_LOCK = threading.Lock()

def get_db_connection():
    global _DB_ENGINE, _DB_INITIALIZED, _SQLITE_CONN, _DB_INIT_LOCK
    db_url = os.environ.get("DATABASE_URL")
    
    if db_url:
        import sqlalchemy
        
        if _DB_ENGINE is None:
            # Neon/Supabase often require this exact schema protocol for psycopg2:
            if db_url.startswith("postgres://"):
                db_url = db_url.replace("postgres://", "postgresql://", 1)
            # Add pool sizing for Streamlit multithreading optimization
            _DB_ENGINE = sqlalchemy.create_engine(db_url, pool_size=5, max_overflow=10, pool_pre_ping=True)
            
        conn = _DB_ENGINE.connect()
        
        if not _DB_INITIALIZED:
            with _DB_INIT_LOCK:
                if not _DB_INITIALIZED:
                    # Postgres-compatible schema creation
                    conn.execute(sqlalchemy.text("""CREATE TABLE IF NOT EXISTS memory (
                                        ticker TEXT PRIMARY KEY,
                                        first_seen TEXT,
                                        times_flagged INTEGER,
                                        last_score REAL
                                    )"""))
                    conn.execute(sqlalchemy.text("""CREATE TABLE IF NOT EXISTS journal (
                                        timestamp TEXT,
                                        ticker TEXT,
                                        action TEXT,
                                        scan_date TEXT,
                                        entry_price REAL,
                                        position_size_pct REAL,
                                        shares REAL,
                                        stop_loss_pct REAL,
                                        take_profit_pct REAL,
                                        notes TEXT
                                    )"""))
                    conn.commit()
                    
                    try:
                        conn.execute(sqlalchemy.text("ALTER TABLE journal ADD COLUMN shares REAL"))
                        conn.commit()
                    except:
                        conn.rollback()
                        
                    conn.execute(sqlalchemy.text("""CREATE TABLE IF NOT EXISTS features (
                                        id SERIAL PRIMARY KEY,
                                        ticker TEXT, sector TEXT, score REAL, f_score REAL,
                                        v_score REAL, m_score REAL, s_score REAL, p_score REAL,
                                        tier TEXT, price REAL, mkt_cap_m REAL, reason_count INTEGER,
                                        scan_date TEXT, scan_timestamp TEXT, run_type TEXT
                                    )"""))
                    conn.execute(sqlalchemy.text("""CREATE TABLE IF NOT EXISTS results (
                                        id SERIAL PRIMARY KEY,
                                        ticker TEXT, sector TEXT, score REAL, f_score REAL,
                                        v_score REAL, m_score REAL, s_score REAL, p_score REAL,
                                        tier TEXT, reasons TEXT, price REAL, mkt_cap TEXT,
                                        scan_date TEXT, scan_timestamp TEXT, run_type TEXT
                                    )"""))
                    conn.commit()
                    _DB_INITIALIZED = True
            
        return conn
        
    else:
        # Fallback to local SQLite if cloud not configured
        if _SQLITE_CONN is None:
            _SQLITE_CONN = sqlite3.connect(DB_FILE, check_same_thread=False)
            
        if not _DB_INITIALIZED:
            with _DB_INIT_LOCK:
                if not _DB_INITIALIZED:
                    # Ensure tables exist
                    _SQLITE_CONN.execute("""CREATE TABLE IF NOT EXISTS memory (
                                        ticker TEXT PRIMARY KEY,
                                        first_seen TEXT,
                                        times_flagged INTEGER,
                                        last_score REAL
                                    )""")
                    _SQLITE_CONN.execute("""CREATE TABLE IF NOT EXISTS journal (
                                        timestamp TEXT,
                                        ticker TEXT,
                                        action TEXT,
                                        scan_date TEXT,
                                        entry_price REAL,
                                        position_size_pct REAL,
                                        shares REAL,
                                        stop_loss_pct REAL,
                                        take_profit_pct REAL,
                                        notes TEXT
                                    )""")
                    _SQLITE_CONN.commit()
                    
                    try:
                        _SQLITE_CONN.execute("ALTER TABLE journal ADD COLUMN shares REAL")
                        _SQLITE_CONN.commit()
                    except:
                        pass
                        
                    _SQLITE_CONN.execute("""CREATE TABLE IF NOT EXISTS features (
                                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                                        ticker TEXT, sector TEXT, score REAL, f_score REAL,
                                        v_score REAL, m_score REAL, s_score REAL, p_score REAL,
                                        tier TEXT, price REAL, mkt_cap_m REAL, reason_count INTEGER,
                                        scan_date TEXT, scan_timestamp TEXT, run_type TEXT
                                    )""")
                    _SQLITE_CONN.execute("""CREATE TABLE IF NOT EXISTS results (
                                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                                        ticker TEXT, sector TEXT, score REAL, f_score REAL,
                                        v_score REAL, m_score REAL, s_score REAL, p_score REAL,
                                        tier TEXT, reasons TEXT, price REAL, mkt_cap TEXT,
                                        scan_date TEXT, scan_timestamp TEXT, run_type TEXT
                                    )""")
                    _SQLITE_CONN.commit()
                    _DB_INITIALIZED = True
            
        return _SQLITE_CONN

def migrate_csv_to_sqlite():
    """One-time migration script. Safe to call multiple times."""
    conn = get_db_connection()
    c = Config()
    
    try:
        # Check if we've already migrated memory to avoid duplicate inserts
        db_mem_count = pd.read_sql("SELECT COUNT(*) FROM memory", conn).iloc[0,0]
        if db_mem_count == 0 and os.path.exists(c.MEMORY_FILE):
            df = pd.read_csv(c.MEMORY_FILE)
            df.to_sql("memory", conn, if_exists="append", index=False)

        # Append to journal if journal table is completely empty
        db_jrnl_count = pd.read_sql("SELECT COUNT(*) FROM journal", conn).iloc[0,0]
        if db_jrnl_count == 0 and os.path.exists(c.JOURNAL_FILE):
            df = pd.read_csv(c.JOURNAL_FILE)
            df.to_sql("journal", conn, if_exists="append", index=False)

        db_feat_count = pd.read_sql("SELECT COUNT(*) FROM features", conn).iloc[0,0]
        if db_feat_count == 0 and os.path.exists(c.FEATURES_FILE):
            df = pd.read_csv(c.FEATURES_FILE)
            df.to_sql("features", conn, if_exists="append", index=False)

        db_res_count = pd.read_sql("SELECT COUNT(*) FROM results", conn).iloc[0,0]
        if db_res_count == 0 and os.path.exists(c.RESULTS_HISTORY_FILE):
            df = pd.read_csv(c.RESULTS_HISTORY_FILE)
            df.to_sql("results", conn, if_exists="append", index=False)
            
        conn.commit()
    except Exception as e:
        logger.error(f"Migration error: {e}")
    finally:
        conn.close()

import threading
_cache_lock = threading.Lock()

def get_stock_info(ticker):
    try:
        today = date.today().isoformat()
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        with _cache_lock:
            if os.path.exists(CACHE_FILE):
                try:
                    with open(CACHE_FILE, "r") as f:
                        raw = json.load(f)
                except Exception:
                    raw = {} # Recover from corruption
                cache = {k: v for k, v in raw.items() if v.get("date") == today}
            else:
                cache = {}

            if ticker in cache and cache[ticker].get("date") == date_str:
                return cache[ticker]["data"]

        # Fetch outside the lock so we don't serialize slow network requests unnecessarily!
        try:
            info = yf.Ticker(ticker).info
        except:
            return {}
            
        with _cache_lock:
            # Re-read cache to ensure we don't overwrite other threads that just finished
            if os.path.exists(CACHE_FILE):
                try:
                    with open(CACHE_FILE, "r") as f:
                        raw = json.load(f)
                except Exception:
                    raw = {}
            else:
                raw = {}
                
            raw[ticker] = {"date": date_str, "data": info}
            pruned = {k: v for k, v in raw.items() if v.get("date") == today}
            
            with open(CACHE_FILE, "w") as f:
                json.dump(pruned, f)
                
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
    GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
    MEMORY_FILE: str = "argus_memory.csv"
    WATCHLIST_FILE: str = "argus_watchlist.csv"
    MIN_SCORE: int = 65
    TOP_N: int = 10
    PRICE_FLOOR: float = 2.0
    PRICE_CEILING: float = None
    VOL_FLOOR: int = 200000
    RESULTS_FILE: str = "argus_results.csv"
    RESULTS_HISTORY_FILE: str = "argus_results_history.csv"
    FEATURES_FILE: str = "argus_feature_history.csv"
    JOURNAL_FILE: str = "argus_journal.csv"

# Always ensure migration happens when engine starts (now safely below Config definition)
migrate_csv_to_sqlite()

def load_memory(filepath=None):
    try:
        conn = get_db_connection()
        df = pd.read_sql("SELECT * FROM memory", conn)
        conn.close()
        if not df.empty:
            return df
    except:
        pass
    return pd.DataFrame(columns=["ticker", "first_seen", "times_flagged", "last_score"])

def save_memory(df, filepath=None):
    conn = get_db_connection()
    df.to_sql("memory", conn, if_exists="replace", index=False)
    conn.close()

def _append_feature_rows(results, scan_date, scan_timestamp, run_type, feature_file=None):
    cols = [
        "ticker", "sector", "score", "f_score", "v_score", "m_score", "s_score", "p_score",
        "tier", "price", "mkt_cap_m",
        "reason_count", "scan_date", "scan_timestamp", "run_type",
    ]
    if not results:
        return

    rows = []
    for pick in results:
        mkt_cap_raw = str(pick.get("mkt_cap", "0")).replace("$", "").replace("M", "")
        try:
            mkt_cap_m = float(mkt_cap_raw)
        except Exception:
            mkt_cap_m = 0.0
        rows.append({
            "ticker": pick.get("ticker", ""),
            "sector": pick.get("sector", "Unknown"),
            "score": float(pick.get("score", 0)),
            "f_score": float(pick.get("f_score", 0)),
            "v_score": float(pick.get("v_score", 0)),
            "m_score": float(pick.get("m_score", 0)),
            "s_score": float(pick.get("s_score", 0)),
            "p_score": float(pick.get("p_score", 0)),
            "tier": pick.get("tier", ""),
            "price": float(pick.get("price", 0)),
            "mkt_cap_m": mkt_cap_m,
            "reason_count": len(pick.get("reasons", [])),
            "scan_date": scan_date,
            "scan_timestamp": scan_timestamp,
            "run_type": run_type,
        })
    df = pd.DataFrame(rows)
    conn = get_db_connection()
    df.to_sql("features", conn, if_exists="append", index=False)
    conn.close()

def _prefilter_tickers(tickers, config, scan_limit=None):
    """Pre-filter tickers by price and volume before deep scoring."""
    if not tickers:
        return []

    batch_size = 100
    valid_tickers = []
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            # Let yfinance elegantly handle internal threads safely per batch
            batch_hist = yf.download(batch, period="1mo", group_by="ticker", progress=False, threads=False)
            
            for t in batch:
                try:
                    # Parse depending on single vs multi-index return shapes from yfinance
                    if isinstance(batch_hist.columns, pd.MultiIndex) and t in batch_hist.columns.get_level_values(0):
                        data = batch_hist[t]
                    elif t in batch_hist.columns:
                        data = batch_hist[t]
                    else:
                        continue
                    
                    
                    if (
                        not data["Close"].dropna().empty
                        and data["Close"].iloc[-1] > config.PRICE_FLOOR
                        and (config.PRICE_CEILING is None or data["Close"].iloc[-1] <= config.PRICE_CEILING)
                        and data["Volume"].mean() > config.VOL_FLOOR
                    ):
                        valid_tickers.append(t)
                        if scan_limit and len(valid_tickers) >= scan_limit:
                            return valid_tickers
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Batch fail during prefilter: {e}")
            continue

    return valid_tickers

def _apply_sector_diversity(results, top_n, max_per_sector=3):
    sector_picks = {}
    for pick in results:
        sector = pick.get("sector", "Unknown")
        bucket = sector_picks.setdefault(sector, [])
        if len(bucket) < max_per_sector:
            bucket.append(pick)
            
    final_picks = [pick for picks in sector_picks.values() for pick in picks]
    final_picks.sort(key=lambda x: x["score"], reverse=True)
    return final_picks[:top_n]

def get_market_regime():
    """Phase 3: Macroeconomic & Market Regime Filter"""
    try:
        spy = yf.Ticker("SPY").history(period="1y")
        vix = yf.Ticker("^VIX").history(period="1mo")
        
        if spy.empty or vix.empty or len(spy) < 200:
            return {"regime": "Neutral", "multiplier": 1.0, "reason": "Insufficient data"}
            
        spy_price = spy["Close"].iloc[-1]
        spy_ma50 = spy["Close"].rolling(50).mean().iloc[-1]
        spy_ma200 = spy["Close"].rolling(200).mean().iloc[-1]
        vix_price = vix["Close"].iloc[-1]
        
        if vix_price > 30:
            return {"regime": "Extreme Fear", "multiplier": 0.7, "reason": f"VIX elevated at {vix_price:.1f}"}
        elif spy_price < spy_ma200:
            return {"regime": "Bear", "multiplier": 0.8, "reason": "SPY below 200-day MA"}
        elif spy_price > spy_ma50 and spy_price > spy_ma200:
            return {"regime": "Bull", "multiplier": 1.1, "reason": "SPY above 50-day and 200-day MA"}
        else:
            return {"regime": "Neutral", "multiplier": 1.0, "reason": "SPY consolidating between moving averages"}
    except Exception as e:
        logger.warning(f"Market regime check failed: {e}")
        return {"regime": "Neutral", "multiplier": 1.0, "reason": "Data fetch error"}

def format_pick(pick, memory_df):
    ticker = pick["ticker"]
    prev = memory_df[memory_df["ticker"] == ticker]
    memory_note = ""
    if not prev.empty:
        times = int(prev["times_flagged"].values[0])
        first = prev["first_seen"].values[0]
        memory_note = f"\n⚡ _Previously flagged {times}x since {first} — signals strengthening_"

    reasons_str = " | ".join(pick["reasons"])
    
    return (
        f"{pick['tier']}\n"
        f"*{ticker}* — Score: *{pick['score']}/100*{memory_note}\n"
        f"💰 Price: ${pick['price']} | Cap: {pick['mkt_cap']}\n"
        f"📊 {reasons_str}\n"
    )

def generate_telegram_message(results, scanned_count, title="Argus Daily Scan", date_str=None, alerts=None):
    if not date_str:
        date_str = datetime.now().strftime("%d %b %Y")
        
    memory_df = load_memory()
    
    highest = [p for p in results if p["score"] >= 80]
    high = [p for p in results if p["score"] < 80]
    
    formatted_highest = [format_pick(p, memory_df) for p in highest]
    
    header = f"👁 *{title} — {date_str}*\n{'─'*30}\n"
    
    alerts_block = ""
    if alerts:
        alerts_block = "*🚨 PORTFOLIO ALERTS*\n" + "\n".join(alerts) + f"\n\n{'─'*30}\n"
        
    highest_block = "*🚀 Highest scoring picks*\n" + ("\n".join(formatted_highest) if formatted_highest else "_None today_")
    high_block = "\n*📌 High scoring picks*\n" + ("\n".join([format_pick(p, memory_df) for p in high]) if high else "_None today_")
    footer = f"\n{'─'*30}\n_Scanned {scanned_count} tickers • Top {len(results)} picks shown_"
    
    return header + alerts_block + highest_block + high_block + footer

def run_scan(config, scan_limit=400, update_memory=True, progress_callback=None):
    """
    Execute Argus scan and return standardized payload.
    This is used by both scheduled runs and Streamlit manual runs.
    """
    import concurrent.futures

    tickers = get_universe()
    memory_df = load_memory(config.MEMORY_FILE)
    scan_date = datetime.now().strftime("%Y-%m-%d")
    scan_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    regime_info = get_market_regime()
    logger.info(f"Market Regime: {regime_info['regime']} ({regime_info['reason']})")

    valid_tickers = _prefilter_tickers(tickers, config, scan_limit=scan_limit)
    results = []
    total = len(valid_tickers)

    def scan_worker(ticker):
        try:
            import time
            time.sleep(0.7)
            return score_stock(ticker, memory_df, config, regime_info)
        except Exception:
            return None

    # Limit workers to 10 to avoid too many simultaneous requests to Yahoo Finance
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # submit all tasks
        future_to_ticker = {executor.submit(scan_worker, t): t for t in valid_tickers}
        
        for idx, future in enumerate(concurrent.futures.as_completed(future_to_ticker)):
            ticker = future_to_ticker[future]
            if progress_callback:
                progress_callback(ticker, idx, total)
            pick = future.result()
            if pick:
                results.append(pick)

    results.sort(key=lambda x: x["score"], reverse=True)
    results = _apply_sector_diversity(results, top_n=config.TOP_N, max_per_sector=3)

    if update_memory and results:
        today = datetime.now().strftime("%d %b %Y")
        for pick in results:
            ticker = pick["ticker"]
            if ticker in memory_df["ticker"].values:
                memory_df.loc[memory_df["ticker"] == ticker, "times_flagged"] += 1
                memory_df.loc[memory_df["ticker"] == ticker, "last_score"] = pick["score"]
            else:
                new_row = pd.DataFrame([{
                    "ticker": ticker,
                    "first_seen": today,
                    "times_flagged": 1,
                    "last_score": pick["score"],
                }])
                memory_df = pd.concat([memory_df, new_row], ignore_index=True)
        save_memory(memory_df, config.MEMORY_FILE)

    return {
        "results": results,
        "scan_date": scan_date,
        "scan_timestamp": scan_timestamp,
        "scanned_count": len(valid_tickers),
    }

def save_results(results, scan_date, scan_timestamp, run_type, latest_file, history_file, write_latest, feature_file=None):
    """
    Persist scan outputs
    """
    conn = get_db_connection()
    if results:
        df = pd.DataFrame(results).assign(
            scan_date=scan_date,
            scan_timestamp=scan_timestamp,
            run_type=run_type)
        # Handle "reasons" list conversion to string for sqlite
        if "reasons" in df.columns:
            df["reasons"] = df["reasons"].apply(lambda x: str(x) if isinstance(x, list) else x)
        
        df.to_sql("results", conn, if_exists="append", index=False)
        
        if write_latest:
            # To emulate latest_file logic in app.py, we can leave this part writing to CSV 
            # OR we just rely on latest rows inside sqlite. We will write to latest_file 
            # to not break app.py immediately, although we will update app.py too.
            df.to_csv(latest_file, index=False)
    conn.close()

    _append_feature_rows(results, scan_date, scan_timestamp, run_type)

def save_journal_entry(journal_file, entry):
    row = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": entry.get("ticker", ""),
        "action": entry.get("action", ""),
        "scan_date": entry.get("scan_date", ""),
        "entry_price": entry.get("entry_price", ""),
        "position_size_pct": entry.get("position_size_pct", ""),
        "shares": entry.get("shares", ""),
        "stop_loss_pct": entry.get("stop_loss_pct", ""),
        "take_profit_pct": entry.get("take_profit_pct", ""),
        "notes": entry.get("notes", ""),
    }])
    conn = get_db_connection()
    row.to_sql("journal", conn, if_exists="append", index=False)
    conn.close()

def load_journal(journal_file=None):
    try:
        conn = get_db_connection()
        df = pd.read_sql("SELECT * FROM journal", conn)
        conn.close()
        return df
    except:
        return pd.DataFrame(columns=[
            "timestamp", "ticker", "action", "scan_date", "entry_price",
            "position_size_pct", "shares", "stop_loss_pct", "take_profit_pct", "notes",
        ])

def monitor_portfolio():
    """
    Auto-Pilot Monitor:
    Reads the journal table (active positions), fetches current prices,
    and checks against Stop-Loss and Take-Profit levels.
    """
    alerts = []
    
    try:
        conn = get_db_connection()
        journal_df = pd.read_sql("SELECT * FROM journal", conn)
        conn.close()
    except Exception as e:
        logger.error(f"Failed to load journal: {e}")
        return alerts
    
    if journal_df.empty:
        return alerts
        
    journal_df["timestamp"] = pd.to_datetime(journal_df["timestamp"], errors="coerce")
    
    open_positions = {}
    for ticker, t_df in journal_df.groupby("ticker"):
        t_df = t_df.sort_values("timestamp")
        buys = t_df[t_df["action"].isin(["BUY", "SCALE_IN"])]
        sells = t_df[t_df["action"].isin(["SELL", "TRIM"])]
        
        # If position is completely closed, skip
        if len(buys) <= len(sells):
            continue
            
        avg_buy = buys["entry_price"].mean()
        # Take SL/TP from the latest buy logic
        latest_buy = buys.iloc[-1]
        sl_pct = float(latest_buy.get("stop_loss_pct", 0) or 5.0)  # Default 5% SL
        tp_pct = float(latest_buy.get("take_profit_pct", 0) or 20.0) # Default 20% TP
        
        open_positions[ticker] = {
            "avg_buy": avg_buy,
            "step_loss_pct": sl_pct,
            "take_profit_pct": tp_pct
        }

    tickers = list(open_positions.keys())
    if not tickers:
        return alerts
        
    # Get current prices safely
    try:
        if len(tickers) == 1:
            price_data = yf.download(tickers[0], period="1d", progress=False)
            current_prices = {tickers[0]: float(price_data["Close"].iloc[-1])}
        else:
            price_data = yf.download(tickers, period="1d", group_by="ticker", progress=False)
            current_prices = {}
            for t in tickers:
                if t in price_data.columns.get_level_values(0):
                    current_prices[t] = float(price_data[t]["Close"].iloc[-1])
                elif t in price_data.columns:
                    current_prices[t] = float(price_data["Close"].iloc[-1])
    except Exception as e:
        logger.warning(f"Failed to fetch current prices for monitor: {e}")
        return alerts

    for ticker, pos in open_positions.items():
        if ticker not in current_prices:
            continue
            
        curr_price = float(current_prices[ticker])
        entry = float(pos["avg_buy"])
        
        if entry <= 0:
            continue
        
        try:
            step_loss = float(pos.get("step_loss_pct", 0) or 0)
        except (ValueError, TypeError):
            step_loss = 0.0
            
        try:
            take_profit = float(pos.get("take_profit_pct", 0) or 0)
        except (ValueError, TypeError):
            take_profit = 0.0
            
        sl_price = entry * (1 - (step_loss / 100))
        tp_price = entry * (1 + (take_profit / 100))
        
        pct_move = ((curr_price - entry) / entry) * 100
        
        if curr_price <= sl_price:
            alerts.append(f"🔴 *STOP LOSS BROKEN:* {ticker} hit ${curr_price:.2f} (Entry: ${entry:.2f}, {pct_move:.1f}%)")
        elif curr_price >= tp_price:
            alerts.append(f"✅ *TAKE PROFIT REACHED:* {ticker} hit ${curr_price:.2f} (Entry: ${entry:.2f}, +{pct_move:.1f}%)")

    return alerts

def build_prediction_model(features_file=None, horizon_days=63, target_return=0.10):
    """
    ML model:
    - Uses historical feature snapshots
    - Computes forward return from first future snapshot >= horizon_days
    - Trains XGBoost classifier if available to predict hit probability
    """
    try:
        conn = get_db_connection()
        df = pd.read_sql("SELECT * FROM features", conn)
        conn.close()
    except Exception as e:
        return {"ready": False, "reason": "No feature history yet or failed to connect."}
        
    if df.empty:
        return {"ready": False, "reason": "Feature history is empty."}

    df["scan_date"] = pd.to_datetime(df["scan_date"], errors="coerce")
    df = df.dropna(subset=["scan_date", "ticker", "score", "price"]).copy()
    if df.empty:
        return {"ready": False, "reason": "Feature history lacks valid rows."}

    df = df.sort_values(["ticker", "scan_date"]).reset_index(drop=True)
    fwd_returns = []
    
    # Target creation
    for ticker, group in df.groupby("ticker"):
        g = group.sort_values("scan_date").copy()
        future_dates = g["scan_date"].tolist()
        future_prices = g["price"].tolist()
        for idx in range(len(g)):
            start_date = future_dates[idx]
            start_price = future_prices[idx]
            target_date = start_date + pd.Timedelta(days=horizon_days)
            ret = None
            for j in range(idx + 1, len(g)):
                if future_dates[j] >= target_date and start_price and start_price > 0:
                    ret = (future_prices[j] / start_price) - 1
                    break
            fwd_returns.append(ret)

    df["fwd_return"] = fwd_returns
    train = df.dropna(subset=["fwd_return"]).copy()
    if len(train) < 30:
        return {"ready": False, "reason": "Not enough matured samples yet (need ~30+)."}

    train["target_hit"] = (train["fwd_return"] >= target_return).astype(int)
    
    # Feature selection
    possible_features = ["score", "f_score", "v_score", "m_score", "s_score", "p_score", "reason_count", "mkt_cap_m"]
    features = [f for f in possible_features if f in train.columns]
    
    X = train[features].fillna(0)
    y = train["target_hit"]
    global_prob = float(y.mean())
    
    clf = None
    feature_importance = {}
    
    if HAS_XGB and len(train) > 50:
        try:
            clf = xgb.XGBClassifier(
                n_estimators=50, 
                max_depth=3, 
                learning_rate=0.1, 
                eval_metric="logloss", 
                use_label_encoder=False
            )
            clf.fit(X, y)
            train["pred_prob"] = clf.predict_proba(X)[:, 1]
            imp = clf.feature_importances_
            feature_importance = {features[i]: float(imp[i]) for i in range(len(features))}
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            clf = None
    
    if clf is None:
        # Fallback to empirical deciles
        logger.info("Falling back to empirical decile model.")
        try:
            train["score_bucket"] = pd.qcut(train["score"], q=10, duplicates="drop")
        except Exception:
            train["score_bucket"] = pd.cut(train["score"], bins=5)
            
        bucket_stats_fallback = train.groupby("score_bucket", observed=False)["target_hit"].mean()
        train["pred_prob"] = train["score_bucket"].map(bucket_stats_fallback).fillna(global_prob)
        
    brier = float(((train["pred_prob"] - train["target_hit"]) ** 2).mean())

    calibration = (
        train.assign(prob_bin=pd.cut(train["pred_prob"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True))
        .groupby("prob_bin", observed=False)
        .agg(samples=("target_hit", "count"), actual_hit_rate=("target_hit", "mean"))
        .reset_index()
    )
    
    # Compute scenario metrics across standard bins for UI backward compatibility
    train_for_scenarios = train.copy()
    try:
        train_for_scenarios["mock_score_bucket"] = pd.qcut(train_for_scenarios["score"], q=10, duplicates="drop")
    except:
        train_for_scenarios["mock_score_bucket"] = pd.cut(train_for_scenarios["score"], bins=5)
    
    bucket_stats = (
        train_for_scenarios.groupby("mock_score_bucket", observed=False)
        .agg(
            samples=("target_hit", "count"),
            prob=("pred_prob", "mean"),
            bear=("fwd_return", lambda s: s.quantile(0.2)),
            base=("fwd_return", lambda s: s.quantile(0.5)),
            bull=("fwd_return", lambda s: s.quantile(0.8)))
        .reset_index()
    )
    bucket_stats = bucket_stats.rename(columns={"mock_score_bucket": "score_bucket"})

    cm = None
    if clf is not None:
        import numpy as np
        from sklearn.metrics import confusion_matrix
        y_pred = (train["pred_prob"] >= 0.5).astype(int)
        cm = confusion_matrix(train["target_hit"], y_pred).tolist()

    return {
        "ready": True,
        "horizon_days": horizon_days,
        "target_return": target_return,
        "global_prob": global_prob,
        "samples": int(len(train)),
        "brier_score": brier,
        "bucket_stats": bucket_stats,
        "calibration": calibration,
        "clf": clf,
        "features": features,
        "feature_importance": feature_importance,
        "confusion_matrix": cm,
        "X_sample": train[features]
    }

def add_predictions(df_results, model):
    if df_results.empty:
        return df_results
    out = df_results.copy()
    if not model.get("ready"):
        out["prob_upside"] = None
        out["scenario_bear"] = None
        out["scenario_base"] = None
        out["scenario_bull"] = None
        return out

    global_prob = model["global_prob"]
    
    if model.get("clf") is not None:
        clf = model["clf"]
        features = model["features"]
        X_pred = out.reindex(columns=features, fill_value=0).fillna(0)
        try:
            out["prob_upside"] = clf.predict_proba(X_pred)[:, 1]
        except Exception:
            out["prob_upside"] = global_prob
    else:
        bucket_stats = model["bucket_stats"].copy().sort_values("score_bucket")
        out["score"] = pd.to_numeric(out["score"], errors="coerce")
        def _lookup_prob(score):
            for _, r in bucket_stats.iterrows():
                if pd.notna(score) and score in r["score_bucket"]:
                    return r["prob"]
            return global_prob
        out["prob_upside"] = out["score"].apply(_lookup_prob)

    # Retain the empirical scenarios for risk guidance sizing
    bucket_stats = model["bucket_stats"].copy().sort_values("score_bucket")
    def _lookup_scenario(score, col):
        for _, r in bucket_stats.iterrows():
            if pd.notna(score) and score in r["score_bucket"]:
                return r[col]
        return 0.0

    out["scenario_bear"] = out["score"].apply(lambda s: _lookup_scenario(s, "bear"))
    out["scenario_base"] = out["score"].apply(lambda s: _lookup_scenario(s, "base"))
    out["scenario_bull"] = out["score"].apply(lambda s: _lookup_scenario(s, "bull"))
    return out

def add_risk_guidance(df_results, model, risk_per_trade_pct=0.75, max_position_pct=8.0):
    """
    Add practical execution guidance:
    - confidence label
    - stop loss / take profit (%)
    - suggested position size (% of portfolio)
    """
    if df_results.empty:
        return df_results
    out = df_results.copy()

    if "prob_upside" not in out.columns:
        out = add_predictions(out, model)

    def _volatility_proxy(ticker):
        try:
            hist = yf.Ticker(ticker).history(period="6mo")
            if hist.empty or "Close" not in hist.columns:
                return 8.0
            rets = hist["Close"].pct_change().dropna()
            if rets.empty:
                return 8.0
            # Approximate 1-month move using 20 trading days.
            monthly_vol_pct = float(rets.std() * (20 ** 0.5) * 100)
            return max(4.0, min(16.0, monthly_vol_pct))
        except Exception:
            return 8.0

    global_prob = float(model["global_prob"]) if model.get("ready") else 0.5

    rows = []
    for _, row in out.iterrows():
        ticker = row.get("ticker", "")
        score = float(row.get("score", 0) or 0)
        prob = row.get("prob_upside", global_prob)
        try:
            prob = float(prob)
        except Exception:
            prob = global_prob

        vol_pct = _volatility_proxy(ticker)
        stop_loss_pct = max(6.0, min(18.0, vol_pct * 1.25))

        scenario_base = row.get("scenario_base", 0)
        scenario_bull = row.get("scenario_bull", 0)
        try:
            scenario_base_pct = float(scenario_base) * 100
        except Exception:
            scenario_base_pct = 0.0
        try:
            scenario_bull_pct = float(scenario_bull) * 100
        except Exception:
            scenario_bull_pct = 0.0
        take_profit_pct = max(stop_loss_pct * 2.0, scenario_base_pct, scenario_bull_pct * 0.8, 12.0)

        # Risk-based sizing + conviction adjustment.
        base_position_pct = (risk_per_trade_pct / stop_loss_pct) * 100.0
        conviction_boost = 1.0
        if prob >= global_prob + 0.10 and score >= 80:
            conviction_boost = 1.25
            confidence = "High"
        elif prob >= global_prob and score >= 70:
            conviction_boost = 1.0
            confidence = "Medium"
        else:
            conviction_boost = 0.75
            confidence = "Low"

        suggested_position_pct = max(0.5, min(max_position_pct, base_position_pct * conviction_boost))

        if confidence == "High":
            entry_style = "Scale-in on breakout or first pullback to 20EMA"
        elif confidence == "Medium":
            entry_style = "Starter size only, add if trend confirms"
        else:
            entry_style = "Watchlist-first; wait for stronger confirmation"

        rows.append({
            "confidence": confidence,
            "stop_loss_pct": round(stop_loss_pct, 1),
            "take_profit_pct": round(take_profit_pct, 1),
            "suggested_position_pct": round(suggested_position_pct, 1),
            "entry_style": entry_style,
        })

    guidance_df = pd.DataFrame(rows)
    out = pd.concat([out.reset_index(drop=True), guidance_df], axis=1)
    return out

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

def score_stock(ticker, memory_df, config, regime_info=None):
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
        v_score = 0
        peg = info.get("pegRatio", 2)
        if 0 < peg < 1: v_score += 15; reasons.append(f"PEG {peg:.1f}")
        ps = info.get("priceToSalesTrailing12Months", 10)
        if 0 < ps < 4: v_score += 10; reasons.append(f"P/S {ps:.1f}x")
        score += v_score

        # 3. Momentum (6mo, 50MA, 200MA, volume spike, IWM RS)
        hist = stock.history(period="1y")  # 1y gives enough data for 200MA
        m_score, m_reasons = _score_momentum(hist, ticker)
        score += m_score; reasons.extend(m_reasons)

        # 4. Smart Money & Cap
        s_score = 0
        inst_own = info.get("heldPercentInstitutions", 1) or 1
        if inst_own < 0.40: s_score += 20; reasons.append(f"Inst. {inst_own*100:.0f}%")
        mkt_cap = info.get("marketCap", 0) or 0
        if 50e6 < mkt_cap < 10e9: s_score += 10; reasons.append(f"Cap ${mkt_cap/1e6:.0f}M")
        score += s_score

        # 5. Persistence Bonus
        p_score = 0
        prev = memory_df[memory_df["ticker"] == ticker]
        if not prev.empty and int(prev["times_flagged"].values[0]) >= 3:
            p_score += 5
            reasons.append("\U0001f501 Persistence bonus")
        score += p_score

        # 6. Market Regime Modification (Phase 3)
        if regime_info and regime_info.get("multiplier", 1.0) != 1.0:
            score *= regime_info["multiplier"]
            reasons.append(f"{regime_info['regime']} Regime Adj")
            
        score = int(round(score))

        # 7. AI News Sentiment Score (Only if preliminary math score is decent to save API)
        if score >= 60 and config.GROQ_API_KEY:
            sentiment = get_sentiment_score(ticker, config.GROQ_API_KEY)
            if sentiment != 0:
                score += sentiment
                reasons.append(f"AI Sentiment {sentiment:+d}")

        if score < config.MIN_SCORE:
            return None

        return {
            "ticker":  ticker,
            "sector":  info.get("sector", "Unknown"),
            "score":   score,
            "f_score": f_score,
            "v_score": v_score,
            "m_score": m_score,
            "s_score": s_score,
            "p_score": p_score,
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
            return tickers
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
                        return tickers
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

def optimize_portfolio(tickers: list, risk_free_rate: float = 0.04) -> dict:
    """Phase 4: Run mean-variance portfolio optimization to find Max Sharpe allocation."""
    import logging
    logger = logging.getLogger(__name__)
    if not tickers:
        return {"error": "No tickers provided."}
    
    logger.info(f"Optimizing portfolio for: {tickers}")
    try:
        # Download 1y history
        data = yf.download(tickers, period="1y", interval="1d")["Adj Close"]
        
        # If single ticker was passed accidentally
        if isinstance(data, pd.Series):
            data = data.to_frame()
            
        returns = data.pct_change().dropna()
        if len(returns) < 50:
            return {"error": "Not enough trading days to optimize reliably."}
            
        # Compute mean and covariance
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        num_assets = len(tickers)
        num_portfolios = 5000
        
        results = np.zeros((3, num_portfolios))
        weights_record = []
        
        for i in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            
            weights_record.append(weights)
            
            p_ret = np.sum(weights * mean_returns)
            p_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            p_sharpe = (p_ret - risk_free_rate) / p_std
            
            results[0, i] = p_ret
            results[1, i] = p_std
            results[2, i] = p_sharpe
            
        # Max Sharpe
        max_sharpe_idx = np.argmax(results[2])
        max_sharpe_ret = results[0, max_sharpe_idx]
        max_sharpe_std = results[1, max_sharpe_idx]
        max_sharpe_ratio = results[2, max_sharpe_idx]
        max_sharpe_weights = weights_record[max_sharpe_idx]
        
        # Min Volatility
        min_vol_idx = np.argmin(results[1])
        min_vol_ret = results[0, min_vol_idx]
        min_vol_std = results[1, min_vol_idx]
        min_vol_sharpe = results[2, min_vol_idx]
        min_vol_weights = weights_record[min_vol_idx]
        
        metrics = {
            "max_sharpe": {
                "return": max_sharpe_ret,
                "volatility": max_sharpe_std,
                "sharpe": max_sharpe_ratio,
                "weights": {ticker: float(weight) for ticker, weight in zip(tickers, max_sharpe_weights)}
            },
            "min_volatility": {
                "return": min_vol_ret,
                "volatility": min_vol_std,
                "sharpe": min_vol_sharpe,
                "weights": {ticker: float(weight) for ticker, weight in zip(tickers, min_vol_weights)}
            }
        }
        
        return metrics
    except Exception as e:
        logger.error(f"Portfolio optimization failed: {e}")
        return {"error": str(e)}
