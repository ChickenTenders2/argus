import sqlite3
import json
import logging
import os
import random
import numpy as np
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from scipy.optimize import minimize as _scipy_minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from dataclasses import dataclass
from datetime import datetime, date

from llm import get_sentiment_score

logger = logging.getLogger("Argus.Engine")

os.makedirs("data", exist_ok=True)

CACHE_FILE = "data/metadata_cache.json"
DB_FILE = "data/argus.db"

_DB_ENGINE = None
_DB_INITIALIZED = False
_SQLITE_CONN = None
import threading
_DB_INIT_LOCK = threading.Lock()

def get_db_connection():
    global _DB_ENGINE, _DB_INITIALIZED, _SQLITE_CONN, _DB_INIT_LOCK
    db_url = os.environ.get("DATABASE_URL")

    # ── PostgreSQL path ───────────────────────────────────────────────────────
    if db_url:
        import sqlalchemy

        if _DB_ENGINE is None:
            if db_url.startswith("postgres://"):
                db_url = db_url.replace("postgres://", "postgresql://", 1)
            try:
                _DB_ENGINE = sqlalchemy.create_engine(
                    db_url,
                    pool_size=3,
                    max_overflow=5,
                    pool_pre_ping=True,
                    connect_args={"sslmode": "require", "connect_timeout": 10},
                )
                with _DB_ENGINE.connect() as _test:
                    _test.execute(sqlalchemy.text("SELECT 1"))
            except Exception as _pg_err:
                logger.error(
                    f"PostgreSQL connection failed — falling back to SQLite. "
                    f"Reason: {type(_pg_err).__name__}: {str(_pg_err)[:300]}"
                )
                _DB_ENGINE = None
                db_url = None

    if db_url and _DB_ENGINE is not None:
        conn = _DB_ENGINE.connect()
        if not _DB_INITIALIZED:
            with _DB_INIT_LOCK:
                if not _DB_INITIALIZED:
                    try:
                        conn.execute(sqlalchemy.text("""CREATE TABLE IF NOT EXISTS memory (
                            ticker TEXT PRIMARY KEY, first_seen TEXT,
                            times_flagged INTEGER, last_score REAL)"""))
                        conn.execute(sqlalchemy.text("""CREATE TABLE IF NOT EXISTS journal (
                            timestamp TEXT, ticker TEXT, action TEXT, scan_date TEXT,
                            entry_price REAL, position_size_pct REAL, shares REAL,
                            stop_loss_pct REAL, take_profit_pct REAL, notes TEXT)"""))
                        conn.commit()
                    except Exception as e:
                        logger.warning(f"Concurrent DB init safely skipped (Phase 1): {e}")
                        conn.rollback()
                    try:
                        conn.execute(sqlalchemy.text("ALTER TABLE journal ADD COLUMN shares REAL"))
                        conn.commit()
                    except: conn.rollback()
                    try:
                        conn.execute(sqlalchemy.text("ALTER TABLE journal ADD COLUMN journal_name TEXT DEFAULT 'Default'"))
                        conn.commit()
                    except: conn.rollback()
                    try:
                        conn.execute(sqlalchemy.text("ALTER TABLE journal ADD COLUMN entry_date TEXT DEFAULT NULL"))
                        conn.commit()
                    except: conn.rollback()
                    try:
                        conn.execute(sqlalchemy.text("ALTER TABLE journal ADD COLUMN peak_price REAL DEFAULT NULL"))
                        conn.commit()
                    except: conn.rollback()
                    try:
                        conn.execute(sqlalchemy.text("""CREATE TABLE IF NOT EXISTS features (
                            id SERIAL PRIMARY KEY, ticker TEXT, sector TEXT, score REAL,
                            raw_score REAL,
                            f_score REAL, v_score REAL, m_score REAL, s_score REAL, p_score REAL,
                            tier TEXT, price REAL, mkt_cap_m REAL, reason_count INTEGER,
                            scan_date TEXT, scan_timestamp TEXT, run_type TEXT)"""))
                        try:
                            conn.execute(sqlalchemy.text("ALTER TABLE features ADD COLUMN IF NOT EXISTS raw_score REAL"))
                            conn.commit()
                        except Exception:
                            conn.rollback()
                        conn.execute(sqlalchemy.text("""CREATE TABLE IF NOT EXISTS results (
                            id SERIAL PRIMARY KEY, ticker TEXT, sector TEXT, score REAL,
                            raw_score REAL,
                            f_score REAL, v_score REAL, m_score REAL, s_score REAL, p_score REAL,
                            tier TEXT, reasons TEXT, price REAL, mkt_cap TEXT,
                            scan_date TEXT, scan_timestamp TEXT, run_type TEXT)"""))
                        conn.commit()
                    except Exception as e:
                        logger.warning(f"Concurrent DB init safely skipped (Phase 2b): {e}")
                        conn.rollback()
                    try:
                        conn.execute(sqlalchemy.text("ALTER TABLE results ADD COLUMN IF NOT EXISTS raw_score REAL"))
                        conn.commit()
                    except Exception as e:
                        logger.warning(f"Concurrent DB init safely skipped (Phase 2): {e}")
                        conn.rollback()
                    _DB_INITIALIZED = True
        return conn

    # ── SQLite fallback ───────────────────────────────────────────────────────
    if _SQLITE_CONN is not None:
        try:
            _SQLITE_CONN.execute("SELECT 1")  # verify still alive
        except Exception:
            _SQLITE_CONN = None  # was closed externally — force reopen
    if _SQLITE_CONN is None:
        _SQLITE_CONN = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn = _SQLITE_CONN
    if not _DB_INITIALIZED:
        with _DB_INIT_LOCK:
            if not _DB_INITIALIZED:
                try:
                    conn.execute("""CREATE TABLE IF NOT EXISTS memory (
                        ticker TEXT PRIMARY KEY, first_seen TEXT,
                        times_flagged INTEGER, last_score REAL)""")
                    conn.execute("""CREATE TABLE IF NOT EXISTS journal (
                        timestamp TEXT, ticker TEXT, action TEXT, scan_date TEXT,
                        entry_price REAL, position_size_pct REAL, shares REAL,
                        stop_loss_pct REAL, take_profit_pct REAL, notes TEXT)""")
                    conn.commit()
                except Exception as e:
                    logger.warning(f"Concurrent local DB init safely skipped (Phase 1): {e}")
                try:
                    conn.execute("ALTER TABLE journal ADD COLUMN shares REAL")
                    conn.commit()
                except: pass
                try:
                    conn.execute("ALTER TABLE journal ADD COLUMN journal_name TEXT DEFAULT 'Default'")
                    conn.commit()
                except: pass
                try:
                    conn.execute("""CREATE TABLE IF NOT EXISTS features (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT, sector TEXT, score REAL,
                        raw_score REAL,
                        f_score REAL, v_score REAL, m_score REAL, s_score REAL, p_score REAL,
                        tier TEXT, price REAL, mkt_cap_m REAL, reason_count INTEGER,
                        scan_date TEXT, scan_timestamp TEXT, run_type TEXT)""")
                    try:
                        conn.execute("ALTER TABLE features ADD COLUMN raw_score REAL")
                        conn.commit()
                    except Exception:
                        pass
                    conn.execute("""CREATE TABLE IF NOT EXISTS results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT, sector TEXT, score REAL,
                        raw_score REAL,
                        f_score REAL, v_score REAL, m_score REAL, s_score REAL, p_score REAL,
                        tier TEXT, reasons TEXT, price REAL, mkt_cap TEXT,
                        scan_date TEXT, scan_timestamp TEXT, run_type TEXT)""")
                    conn.commit()
                except Exception as e:
                    logger.warning(f"Concurrent local DB init safely skipped (Phase 2): {e}")
                try:
                    conn.execute("ALTER TABLE results ADD COLUMN raw_score REAL")
                    conn.commit()
                except Exception:
                    pass
                _DB_INITIALIZED = True
    return conn

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

        # Migrate journal only for local SQLite — never overwrite Supabase journal
        if not os.environ.get("DATABASE_URL"):
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
    MEMORY_FILE: str = "data/argus_memory.csv"
    WATCHLIST_FILE: str = "data/argus_watchlist.csv"
    MIN_SCORE: int = 60
    TOP_N: int = 12
    PRICE_FLOOR: float = 2.0
    PRICE_CEILING: float = None
    VOL_FLOOR: int = 500000
    RESULTS_FILE: str = "data/argus_results.csv"
    RESULTS_HISTORY_FILE: str = "data/argus_results_history.csv"
    FEATURES_FILE: str = "data/argus_feature_history.csv"
    JOURNAL_FILE: str = "data/argus_journal.csv"
    # Configurable scoring thresholds
    REV_GROWTH_HIGH: float = 0.30
    REV_GROWTH_LOW: float = 0.20
    ROCE_THRESHOLD: float = 0.20
    GROSS_MARGIN_HIGH: float = 0.60
    GROSS_MARGIN_LOW: float = 0.40
    INST_OWN_CEILING: float = 0.40
    MKT_CAP_MIN: float = 50e6
    MKT_CAP_MAX: float = 10e9

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
        "ticker", "sector", "score", "raw_score", "f_score", "v_score", "m_score", "s_score", "p_score",
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
            "ticker":         pick.get("ticker", ""),
            "sector":         pick.get("sector", "Unknown"),
            "score":          float(pick.get("score", 0)),
            "raw_score":      float(pick.get("raw_score", pick.get("score", 0))),
            "f_score":        float(pick.get("f_score", 0)),
            "v_score":        float(pick.get("v_score", 0)),
            "m_score":        float(pick.get("m_score", 0)),
            "s_score":        float(pick.get("s_score", 0)),
            "p_score":        float(pick.get("p_score", 0)),
            "tier":           pick.get("tier", ""),
            "price":          float(pick.get("price", 0)),
            "mkt_cap_m":      mkt_cap_m,
            "reason_count":   len(pick.get("reasons", [])),
            "scan_date":      scan_date,
            "scan_timestamp": scan_timestamp,
            "run_type":       run_type,
        })
    df = pd.DataFrame(rows)
    conn = get_db_connection()
    # Lazy migration: ensure raw_score column exists (handles pre-existing Supabase tables)
    try:
        import sqlalchemy as _sa
        conn.execute(_sa.text("ALTER TABLE features ADD COLUMN IF NOT EXISTS raw_score REAL"))
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE features ADD COLUMN raw_score REAL")
            conn.commit()
        except Exception:
            pass
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
                except Exception as e:
                    logger.debug(f"Skipping {t} in prefilter: {e}")
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
    """Phase 3: Macroeconomic & Market Regime Filter — enhanced with FRED macro data."""
    try:
        spy = yf.Ticker("SPY").history(period="1y")
        vix = yf.Ticker("^VIX").history(period="1mo")

        if spy.empty or vix.empty or len(spy) < 200:
            return {"regime": "Neutral", "multiplier": 1.0, "reason": "Insufficient data",
                    "spy_ma200_gap_pct": None, "vix_level": None, "vix_trend": None, "macro": {}}

        spy_price = spy["Close"].iloc[-1]
        spy_ma50 = spy["Close"].rolling(50).mean().iloc[-1]
        spy_ma200 = spy["Close"].rolling(200).mean().iloc[-1]
        vix_price = vix["Close"].iloc[-1]

        spy_ma200_gap_pct = ((spy_price - spy_ma200) / spy_ma200) * 100
        vix_week_ago = vix["Close"].iloc[-5] if len(vix) >= 5 else vix["Close"].iloc[0]
        vix_trend = "rising" if vix_price > vix_week_ago * 1.05 else ("falling" if vix_price < vix_week_ago * 0.95 else "stable")
        extra = {"spy_ma200_gap_pct": spy_ma200_gap_pct, "vix_level": vix_price, "vix_trend": vix_trend}

        # ── Base regime from SPY/VIX ──────────────────────────────────────────
        if vix_price > 30:
            regime, multiplier, reason = "Extreme Fear", 0.7, f"VIX elevated at {vix_price:.1f}"
        elif spy_price < spy_ma200:
            regime, multiplier, reason = "Bear", 0.8, "SPY below 200-day MA"
        elif spy_price > spy_ma50 and spy_price > spy_ma200:
            regime, multiplier, reason = "Bull", 1.1, "SPY above 50-day and 200-day MA"
        else:
            regime, multiplier, reason = "Neutral", 1.0, "SPY consolidating between moving averages"

        # ── FRED macro overlay ────────────────────────────────────────────────
        macro = {}
        try:
            from macro_data import build_macro_context
            macro = build_macro_context()
        except Exception as _me:
            logger.debug(f"Macro context unavailable: {_me}")

        yield_inverted = macro.get("yield_curve", {}).get("inverted", False)
        cpi_hot = macro.get("cpi", {}).get("accelerating", False)
        fg_val = macro.get("fear_greed", {}).get("value", 50)

        # Stagflation: bear + hot CPI + inverted yield curve (takes priority over Bear)
        if yield_inverted and cpi_hot and spy_price < spy_ma200 and regime not in ("Extreme Fear",):
            regime = "Stagflation"
            multiplier = 0.65
            reason = "Inverted yield curve + accelerating CPI + SPY below 200MA"
        else:
            # Incremental adjustments: max -0.15 combined penalty
            adj = 0.0
            if yield_inverted:
                adj -= 0.05
            if cpi_hot:
                adj -= 0.05
            if fg_val < 25:
                adj -= 0.05
            if adj != 0.0:
                multiplier = max(0.50, round(multiplier + adj, 2))
                reason += f" (macro adj {adj:+.2f})"

        return {"regime": regime, "multiplier": multiplier, "reason": reason, **extra, "macro": macro}

    except Exception as e:
        logger.warning(f"Market regime check failed: {e}")
        return {"regime": "Neutral", "multiplier": 1.0, "reason": "Data fetch error",
                "spy_ma200_gap_pct": None, "vix_level": None, "vix_trend": None, "macro": {}}

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
    high_block = ("\n*📌 High scoring picks*\n" + "\n".join([format_pick(p, memory_df) for p in high])) if high else ""
    footer = f"\n{'─'*30}\n_Scanned {scanned_count} tickers • Top {len(results)} picks shown_"
    
    return header + alerts_block + highest_block + high_block + footer

def run_scan(config, scan_limit=400, shuffle=True, update_memory=True, progress_callback=None, run_type: str = "manual"):
    """
    Execute Argus scan and return standardized payload.
    This is used by both scheduled runs and Streamlit manual runs.
    shuffle=True  → random subset each run (Random / Full modes)
    shuffle=False → always scans the top-weighted IWM tickers first (Fixed mode)
    """
    import concurrent.futures

    tickers = get_universe()
    if shuffle:
        random.shuffle(tickers)
    memory_df = load_memory(config.MEMORY_FILE)
    scan_date = datetime.now().strftime("%Y-%m-%d")
    scan_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    regime_info = get_market_regime()
    logger.info(f"Market Regime: {regime_info['regime']} ({regime_info['reason']})")

    valid_tickers = _prefilter_tickers(tickers, config, scan_limit=scan_limit)
    results = []
    total = len(valid_tickers)
    filtered_count = 0

    def scan_worker(ticker):
        try:
            import time
            time.sleep(0.4)
            return score_stock(ticker, memory_df, config, regime_info)
        except Exception:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ticker = {executor.submit(scan_worker, t): t for t in valid_tickers}

        for idx, future in enumerate(concurrent.futures.as_completed(future_to_ticker)):
            ticker = future_to_ticker[future]
            if progress_callback:
                progress_callback(ticker, idx, total)
            pick = future.result()
            if pick:
                results.append(pick)
            else:
                filtered_count += 1

    results.sort(key=lambda x: (x["score"], x.get("raw_score", x["score"])), reverse=True)
    _append_feature_rows(results, scan_date, scan_timestamp, run_type)
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
        "filtered_count": filtered_count,
    }

def _migrate_results_columns(conn, df):
    """Lazily add any DataFrame columns missing from the results table to prevent schema mismatch errors."""
    try:
        db_url = os.environ.get("DATABASE_URL")
        if db_url:
            import sqlalchemy as _sa
            for col in df.columns:
                if col == "id":
                    continue
                dtype = "TEXT" if df[col].dtype == object else "DOUBLE PRECISION"
                try:
                    conn.execute(_sa.text(f"ALTER TABLE results ADD COLUMN IF NOT EXISTS {col} {dtype}"))
                    conn.commit()
                except Exception:
                    try: conn.rollback()
                    except Exception: pass
        else:
            try:
                existing = {r[1] for r in conn.execute("PRAGMA table_info(results)").fetchall()}
            except Exception:
                existing = {r[1] for r in conn.execute("PRAGMA table_info(results)")}
            for col in df.columns:
                if col in ("id",) or col in existing:
                    continue
                dtype = "TEXT" if df[col].dtype == object else "REAL"
                try:
                    conn.execute(f"ALTER TABLE results ADD COLUMN {col} {dtype}")
                    conn.commit()
                except Exception as _ce:
                    logger.warning(f"Could not add column '{col}' to results table: {_ce}")
    except Exception as _me:
        logger.warning(f"_migrate_results_columns failed: {_me}")


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
            df["reasons"] = df["reasons"].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
        # Lazily add any new columns before inserting to prevent schema mismatch errors
        _migrate_results_columns(conn, df)
        df.to_sql("results", conn, if_exists="append", index=False)

        # Always append to history CSV for persistence across app restarts.
        # Streamlit Cloud wipes SQLite on redeploy; the committed CSV is the only
        # reliable record that a scan already ran today.
        if history_file:
            try:
                if os.path.exists(history_file):
                    _hist_existing = pd.read_csv(history_file)
                    _hist_combined = pd.concat([_hist_existing, df], ignore_index=True)
                else:
                    _hist_combined = df.copy()
                _hist_combined.to_csv(history_file, index=False)
            except Exception as _he:
                logger.warning(f"save_results: could not update history CSV: {_he}")

        if write_latest:
            df.to_csv(latest_file, index=False)
    conn.close()


def sync_journal_to_csv(journal_file):
    """Write the full journal DB table back to the CSV so git tracks latest state."""
    try:
        df = load_journal()
        df = df[df["ticker"] != "_INIT_"]
        if "journal_name" not in df.columns:
            df["journal_name"] = "Default"
        df.to_csv(journal_file, index=False)
    except Exception as e:
        logger.warning(f"sync_journal_to_csv failed: {e}")


def save_journal_entry(journal_file, entry, journal_name="Default"):
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
        "journal_name": entry.get("journal_name", journal_name),
    }])
    conn = get_db_connection()
    row.to_sql("journal", conn, if_exists="append", index=False)
    conn.close()
    sync_journal_to_csv(journal_file)

def load_journal(journal_file=None, journal_name=None):
    try:
        conn = get_db_connection()
        if journal_name and journal_name != "All":
            try:
                from sqlalchemy import text as sa_text
                df = pd.read_sql(
                    sa_text("SELECT * FROM journal WHERE journal_name = :jn"),
                    conn, params={"jn": journal_name}
                )
            except Exception:
                df = pd.read_sql(
                    "SELECT * FROM journal WHERE journal_name = ?",
                    conn, params=(journal_name,)
                )
        else:
            df = pd.read_sql("SELECT * FROM journal", conn)
        conn.close()
        if "journal_name" not in df.columns:
            df["journal_name"] = "Default"
        return df
    except Exception:
        return pd.DataFrame(columns=[
            "timestamp", "ticker", "action", "scan_date", "entry_price",
            "position_size_pct", "shares", "stop_loss_pct", "take_profit_pct",
            "notes", "journal_name",
        ])


def list_journals():
    """Return sorted list of unique journal names from the journal table."""
    try:
        conn = get_db_connection()
        df = pd.read_sql("SELECT DISTINCT journal_name FROM journal", conn)
        conn.close()
        names = df["journal_name"].dropna().unique().tolist()
        return sorted([n for n in names if n]) or ["Default"]
    except Exception:
        return ["Default"]


def delete_journal(journal_name, journal_file=None):
    """Delete all entries for a given journal name."""
    if not journal_name or journal_name == "Default":
        return False
    try:
        conn = get_db_connection()
        try:
            from sqlalchemy import text as sa_text
            conn.execute(sa_text("DELETE FROM journal WHERE journal_name = :jn"), {"jn": journal_name})
        except Exception:
            conn.execute("DELETE FROM journal WHERE journal_name = ?", (journal_name,))
        conn.commit()
        conn.close()
        if journal_file:
            sync_journal_to_csv(journal_file)
        return True
    except Exception as e:
        logger.error(f"Failed to delete journal '{journal_name}': {e}")
        return False

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

        buy_shares = buys["shares"].fillna(0) if "shares" in buys.columns else pd.Series([0] * len(buys), dtype=float)
        sell_shares = sells["shares"].fillna(0) if "shares" in sells.columns else pd.Series([0] * len(sells), dtype=float)
        net_shares = buy_shares.sum() - sell_shares.sum()

        # Skip if position is fully closed (prefer share count; fall back to trade count)
        if net_shares <= 0 and len(buys) <= len(sells):
            continue

        # Shares-weighted average buy price; fall back to simple mean if shares absent
        total_buy_shares = buy_shares.sum()
        if total_buy_shares > 0:
            avg_buy = float((buys["entry_price"] * buy_shares).sum() / total_buy_shares)
        else:
            avg_buy = float(buys["entry_price"].mean())

        latest_buy = buys.iloc[-1]
        sl_pct = float(latest_buy.get("stop_loss_pct", 0) or 5.0)
        tp_pct = float(latest_buy.get("take_profit_pct", 0) or 20.0)
        entry_date = buys["timestamp"].min()

        open_positions[ticker] = {
            "avg_buy":        avg_buy,
            "sl_pct":         sl_pct,
            "take_profit_pct": tp_pct,
            "entry_date":     entry_date,
        }

    tickers = list(open_positions.keys())
    if not tickers:
        return alerts

    # Batch-download 1y history for current prices + trailing stop peak computation
    try:
        if len(tickers) == 1:
            hist_all = yf.download(tickers[0], period="1y", progress=False)
            hist_by_ticker = {tickers[0]: hist_all}
        else:
            hist_all = yf.download(tickers, period="1y", group_by="ticker", progress=False)
            hist_by_ticker = {}
            for t in tickers:
                if isinstance(hist_all.columns, pd.MultiIndex) and t in hist_all.columns.get_level_values(0):
                    hist_by_ticker[t] = hist_all[t]
                elif not isinstance(hist_all.columns, pd.MultiIndex):
                    hist_by_ticker[t] = hist_all
    except Exception as e:
        logger.warning(f"Failed to fetch history for monitor: {e}")
        return alerts

    current_prices = {}
    for t, h in hist_by_ticker.items():
        if h is not None and not h.empty and "Close" in h.columns:
            current_prices[t] = float(h["Close"].dropna().iloc[-1])

    for ticker, pos in open_positions.items():
        if ticker not in current_prices:
            continue

        curr_price = float(current_prices[ticker])
        entry = float(pos["avg_buy"])
        if entry <= 0:
            continue

        sl_pct = float(pos.get("sl_pct", 5.0) or 5.0)
        tp_pct = float(pos.get("take_profit_pct", 20.0) or 20.0)

        static_sl = entry * (1 - sl_pct / 100)
        tp_price  = entry * (1 + tp_pct / 100)

        # Trailing stop: trail the peak close since entry at the same sl_pct distance.
        # If the stock has risen, trail_sl > static_sl — locking in gains.
        # If it never went above entry, static_sl is the floor.
        peak_price = entry
        entry_date = pos.get("entry_date")
        if ticker in hist_by_ticker and entry_date is not None:
            h = hist_by_ticker[ticker]
            try:
                h_since = h[h.index >= pd.Timestamp(entry_date).tz_localize(None)]
                if not h_since.empty:
                    peak_price = float(h_since["Close"].max())
            except Exception:
                pass

        trail_sl       = peak_price * (1 - sl_pct / 100)
        effective_sl   = max(static_sl, trail_sl)
        is_trailing    = trail_sl > static_sl

        pct_move = ((curr_price - entry) / entry) * 100

        if curr_price <= effective_sl:
            if is_trailing:
                alerts.append(
                    f"🔴 *TRAILING STOP TRIGGERED:* {ticker} hit ${curr_price:.2f} "
                    f"(Peak: ${peak_price:.2f}, Trail SL: ${effective_sl:.2f}, {pct_move:.1f}% from entry)"
                )
            else:
                alerts.append(
                    f"🔴 *STOP LOSS BROKEN:* {ticker} hit ${curr_price:.2f} "
                    f"(Entry: ${entry:.2f}, {pct_move:.1f}%)"
                )
        elif curr_price >= tp_price:
            alerts.append(
                f"✅ *TAKE PROFIT REACHED:* {ticker} hit ${curr_price:.2f} "
                f"(Entry: ${entry:.2f}, +{pct_move:.1f}%)"
            )

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
    possible_features = ["score", "raw_score", "f_score", "v_score", "m_score", "s_score", "p_score", "reason_count", "mkt_cap_m"]
    features = [f for f in possible_features if f in train.columns]

    global_prob = float(train["target_hit"].mean())

    # Temporal walk-forward split: train on oldest 80%, evaluate on newest 20%.
    # Sorted by scan_date so no future data leaks into training.
    train_sorted = train.sort_values("scan_date").reset_index(drop=True)
    split_idx = max(int(len(train_sorted) * 0.8), len(train_sorted) - 20)
    train_set = train_sorted.iloc[:split_idx]
    test_set  = train_sorted.iloc[split_idx:]

    X_train = train_set[features].fillna(0)
    y_train = train_set["target_hit"]
    X_test  = test_set[features].fillna(0)
    y_test  = test_set["target_hit"]
    X_all   = train_sorted[features].fillna(0)
    y_all   = train_sorted["target_hit"]

    clf = None
    feature_importance = {}

    if HAS_XGB and len(train_sorted) > 50:
        try:
            clf = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                eval_metric="logloss",
                use_label_encoder=False,
            )
            clf.fit(X_train, y_train)
            imp = clf.feature_importances_
            feature_importance = {features[i]: float(imp[i]) for i in range(len(features))}
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            clf = None

    # Compute pred_prob on the full dataset for downstream scenario stats
    if clf is not None:
        train_sorted["pred_prob"] = clf.predict_proba(X_all)[:, 1]
        # Brier score on held-out test set (walk-forward evaluation)
        if len(X_test) > 0:
            test_probs = clf.predict_proba(X_test)[:, 1]
            brier = float(((test_probs - y_test) ** 2).mean())
        else:
            brier = float(((train_sorted["pred_prob"] - y_all) ** 2).mean())
    else:
        # Fallback to empirical deciles (trained on train_set, applied to all)
        logger.info("Falling back to empirical decile model.")
        try:
            train_set = train_set.copy()
            train_set["score_bucket"] = pd.qcut(train_set["score"], q=10, duplicates="drop")
        except Exception:
            train_set = train_set.copy()
            train_set["score_bucket"] = pd.cut(train_set["score"], bins=5)

        bucket_stats_fallback = train_set.groupby("score_bucket", observed=False)["target_hit"].mean()

        def _bucket_prob(score):
            for bucket, prob in bucket_stats_fallback.items():
                try:
                    if score in bucket:
                        return prob
                except Exception:
                    pass
            return global_prob

        train_sorted["pred_prob"] = train_sorted["score"].apply(_bucket_prob)
        if len(test_set) > 0:
            brier = float(((train_sorted.iloc[split_idx:]["pred_prob"] - y_test) ** 2).mean())
        else:
            brier = float(((train_sorted["pred_prob"] - y_all) ** 2).mean())

    train = train_sorted  # alias for the rest of the function

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
        from sklearn.metrics import confusion_matrix
        if len(X_test) > 0:
            _cm_probs = clf.predict_proba(X_test)[:, 1]
            _cm_preds = (_cm_probs >= 0.5).astype(int)
            cm = confusion_matrix(y_test, _cm_preds).tolist()
        else:
            y_pred = (train["pred_prob"] >= 0.5).astype(int)
            cm = confusion_matrix(train["target_hit"], y_pred).tolist()

    return {
        "ready": True,
        "horizon_days": horizon_days,
        "target_return": target_return,
        "global_prob": global_prob,
        "samples": int(len(train)),
        "train_samples": int(len(train_set)),
        "test_samples": int(len(test_set)),
        "brier_score": brier,
        "bucket_stats": bucket_stats,
        "calibration": calibration,
        "clf": clf,
        "features": features,
        "feature_importance": feature_importance,
        "confusion_matrix": cm,
        "X_sample": train[features],
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

_vol_cache: dict = {}
_vol_cache_ts: dict = {}
_VOL_CACHE_TTL = 3600

_iwm_cache: dict = {}
_iwm_cache_ts: float = 0.0
_IWM_CACHE_TTL = 3600

def _volatility_proxy(ticker: str) -> float:
    """Return approximate 1-month volatility (%). Cached per ticker for 1 hour."""
    import time
    now = time.time()
    if ticker in _vol_cache and (now - _vol_cache_ts.get(ticker, 0)) < _VOL_CACHE_TTL:
        return _vol_cache[ticker]
    try:
        hist = yf.Ticker(ticker).history(period="6mo")
        if hist.empty or "Close" not in hist.columns:
            result = 8.0
        else:
            rets = hist["Close"].pct_change().dropna()
            result = 8.0 if rets.empty else max(4.0, min(16.0, float(rets.std() * (20 ** 0.5) * 100)))
    except Exception:
        result = 8.0
    _vol_cache[ticker] = result
    _vol_cache_ts[ticker] = now
    return result


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
        if isinstance(earnings, pd.DataFrame) and not earnings.empty:
            if 'EPS Estimate' in earnings.columns and 'Reported EPS' in earnings.columns:
                last_est = earnings['EPS Estimate'].dropna().iloc[-1]
                last_rep = earnings['Reported EPS'].dropna().iloc[-1] if len(earnings['Reported EPS'].dropna()) > 0 else 0
                if last_est > 0 and last_rep / last_est < 0.90:
                    flags.append("Earnings miss")
    except:
        pass
    return flags

def _score_fundamentals(info, stock, config=None):
    """Score: revenue growth (max 20), ROCE (max 7), gross margin (max 5), FCF (max 3) = 35 pts max."""
    score, reasons = 0, []

    rev_high  = getattr(config, "REV_GROWTH_HIGH",   0.30)
    rev_low   = getattr(config, "REV_GROWTH_LOW",    0.20)
    roce_thr  = getattr(config, "ROCE_THRESHOLD",    0.20)
    gm_high   = getattr(config, "GROSS_MARGIN_HIGH", 0.60)
    gm_low    = getattr(config, "GROSS_MARGIN_LOW",  0.40)

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

    if growth >= rev_high:   score += 20; reasons.append(f"Rev growth {growth*100:.0f}%")
    elif growth >= rev_low:  score += 12; reasons.append(f"Rev growth {growth*100:.0f}%")

    # ── ROCE ────────────────────────────────────────────────
    roce = info.get("returnOnCapitalEmployed", info.get("returnOnCapital", 0)) or 0
    if roce > roce_thr: score += 7; reasons.append(f"ROCE {roce*100:.0f}%")

    # ── Gross Margin ────────────────────────────────────────
    # Strongest single predictor of durable competitive advantage.
    gross_margin = info.get("grossMargins", 0) or 0
    if gross_margin > gm_high:   score += 5; reasons.append(f"Gross margin {gross_margin*100:.0f}%")
    elif gross_margin > gm_low:  score += 3; reasons.append(f"Gross margin {gross_margin*100:.0f}%")

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
        score += 3; reasons.append("FCF positive")

    return score, reasons

def _score_momentum(hist, ticker):
    """Score: 6mo momentum (8), 50MA (4), 200MA (4), volume spike (8), IWM RS (6) = 30 pts max."""
    score, reasons = 0, []
    if hist.empty or len(hist) < 50:
        return score, reasons

    price_now  = hist["Close"].iloc[-1]
    price_6mo  = hist["Close"].iloc[-126] if len(hist) >= 126 else hist["Close"].iloc[0]
    ma50       = hist["Close"].rolling(50).mean().iloc[-1]
    vol_avg    = hist["Volume"].rolling(30).mean().iloc[-1]
    vol_today  = hist["Volume"].iloc[-1]

    # 6-month price momentum
    if price_now > price_6mo * 1.15:
        score += 8; reasons.append("6mo Momentum")

    # Price above 50-day MA
    if price_now > ma50:
        score += 4; reasons.append("Above 50MA")

    # Price above 200-day MA (trend health check)
    if len(hist) >= 200:
        ma200 = hist["Close"].rolling(200).mean().iloc[-1]
        if price_now > ma200:
            score += 4; reasons.append("Above 200MA")
    else:
        logger.debug(f"{ticker}: fewer than 200 days of history — 200MA check skipped")

    # Unusual volume spike
    if vol_today > vol_avg * 2:
        score += 8; reasons.append("⚡ Volume spike")

    # Relative Strength vs IWM (Russell 2000 benchmark)
    # +6 pts if the stock has outperformed IWM by 20%+ over the last 3 months.
    try:
        # Use the last ~63 trading days as a proxy for 3 months
        if len(hist) >= 63:
            import time as _time
            stock_3mo_return = (price_now / hist["Close"].iloc[-63]) - 1

            global _iwm_cache, _iwm_cache_ts
            _now = _time.time()
            if not _iwm_cache or (_now - _iwm_cache_ts) > _IWM_CACHE_TTL:
                _iwm_cache = {"hist": yf.Ticker("IWM").history(period="3mo")}
                _iwm_cache_ts = _now
            iwm_hist = _iwm_cache.get("hist")

            if iwm_hist is not None and not iwm_hist.empty:
                iwm_return = (iwm_hist["Close"].iloc[-1] / iwm_hist["Close"].iloc[0]) - 1
                rs_delta   = stock_3mo_return - iwm_return
                if rs_delta >= 0.20:
                    score += 6; reasons.append(f"RS vs IWM +{rs_delta*100:.0f}%")
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
        f_score, f_reasons = _score_fundamentals(info, stock, config)
        score += f_score; reasons.extend(f_reasons)

        # 2. Valuation (max 10 pts: PEG 6 + P/S 4)
        v_score = 0
        peg = info.get("pegRatio", 2)
        if 0 < peg < 1: v_score += 6; reasons.append(f"PEG {peg:.1f}")
        ps = info.get("priceToSalesTrailing12Months", 10)
        if 0 < ps < 4: v_score += 4; reasons.append(f"P/S {ps:.1f}x")
        score += v_score

        # 3. Momentum (6mo, 50MA, 200MA, volume spike, IWM RS — max 30 pts)
        hist = stock.history(period="1y")  # 1y gives enough data for 200MA
        m_score, m_reasons = _score_momentum(hist, ticker)
        score += m_score; reasons.extend(m_reasons)

        # 4. Smart Money & Cap (max 20 pts: inst ownership 13 + cap range 7)
        s_score = 0
        inst_ceil = getattr(config, "INST_OWN_CEILING", 0.40)
        mkt_min   = getattr(config, "MKT_CAP_MIN", 50e6)
        mkt_max   = getattr(config, "MKT_CAP_MAX", 10e9)
        inst_own = info.get("heldPercentInstitutions", 1) or 1
        if inst_own < inst_ceil: s_score += 13; reasons.append(f"Inst. {inst_own*100:.0f}%")
        mkt_cap = info.get("marketCap", 0) or 0
        if mkt_min < mkt_cap < mkt_max: s_score += 7; reasons.append(f"Cap ${mkt_cap/1e6:.0f}M")
        score += s_score

        # 5. Persistence Bonus (max 5 pts — unchanged)
        p_score = 0
        prev = memory_df[memory_df["ticker"] == ticker]
        if not prev.empty and int(prev["times_flagged"].values[0]) >= 3:
            p_score += 5
            reasons.append("\U0001f501 Persistence bonus")
        score += p_score

        # Capture raw quality score before regime adjustment for MIN_SCORE gating.
        # The regime multiplier adjusts the *displayed* score to reflect market context
        # but must not make the threshold unachievable (e.g. 0.7 × max-100 = 70 < 72).
        quality_score = score

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

        # raw_score is the unclamped total — used as tiebreaker when multiple picks hit 100
        raw_score = score
        score = min(100, score)

        # Gate on pre-multiplier quality score, adjusted for the regime multiplier so that
        # borderline stocks are not rejected before the multiplier can push them over MIN_SCORE.
        # e.g. MIN_SCORE=75, Bull ×1.05 → gate = 71.4, letting quality_score 72 → 75.6 pass.
        _gate_mult = regime_info.get("multiplier", 1.0) if regime_info else 1.0
        _gate_threshold = config.MIN_SCORE / max(_gate_mult, 0.01)
        if quality_score < _gate_threshold:
            return None

        return {
            "ticker":     ticker,
            "sector":     info.get("sector", "Unknown"),
            "score":      score,
            "raw_score":  raw_score,
            "f_score":    f_score,
            "v_score":    v_score,
            "m_score":    m_score,
            "s_score":    s_score,
            "p_score":    p_score,
            "tier":       "\U0001f7e2 HIGH CONVICTION" if score >= 80 else "\U0001f7e1 WATCHLIST",
            "reasons":    reasons,
            "price":      round(hist["Close"].iloc[-1], 2) if not hist.empty else 0,
            "mkt_cap":    f"${mkt_cap/1e6:.0f}M"
        }
    except Exception as e:
        logger.warning(f"score_stock failed for {ticker}: {e}")
        return None

def get_universe():
    import ssl
    import io
    import urllib.request
    _ssl_ctx = ssl.create_default_context()
    _ssl_ctx.check_hostname = False
    _ssl_ctx.verify_mode = ssl.CERT_NONE
    _opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=_ssl_ctx))

    # Method 1: Try iShares IWM holdings CSV
    try:
        url = "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund"
        raw = _opener.open(url, timeout=15).read()
        df = pd.read_csv(io.BytesIO(raw), skiprows=9, on_bad_lines='skip')
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
    """Phase 4: Mean-variance optimization (Max Sharpe + Min Volatility) via scipy SLSQP."""
    if not tickers:
        return {"error": "No tickers provided."}

    try:
        data = yf.download(tickers, period="1y", interval="1d", progress=False)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame()
        returns = data.pct_change().dropna()
        if len(returns) < 50:
            return {"error": "Not enough trading days to optimize reliably."}

        mean_ret = returns.mean().values * 252
        cov = returns.cov().values * 252
        n = len(tickers)
        w0 = np.full(n, 1.0 / n)
        bounds = tuple((0.0, 1.0) for _ in range(n))
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]

        def _neg_sharpe(w):
            r = w @ mean_ret
            v = np.sqrt(w @ cov @ w)
            return -(r - risk_free_rate) / v if v > 0 else 0.0

        def _port_vol(w):
            return np.sqrt(w @ cov @ w)

        if HAS_SCIPY:
            res_sharpe = _scipy_minimize(_neg_sharpe, w0, method="SLSQP",
                                         bounds=bounds, constraints=constraints,
                                         options={"ftol": 1e-9, "maxiter": 1000})
            res_vol    = _scipy_minimize(_port_vol, w0, method="SLSQP",
                                         bounds=bounds, constraints=constraints,
                                         options={"ftol": 1e-9, "maxiter": 1000})
            ms_w = res_sharpe.x
            mv_w = res_vol.x
        else:
            # Monte Carlo fallback when scipy is unavailable
            rng = np.random.default_rng()
            sim_w = rng.dirichlet(np.ones(n), size=3000)
            sharpes = np.array([
                -(w @ mean_ret - risk_free_rate) / max(np.sqrt(w @ cov @ w), 1e-9)
                for w in sim_w
            ])
            vols = np.array([np.sqrt(w @ cov @ w) for w in sim_w])
            ms_w = sim_w[np.argmin(sharpes)]
            mv_w = sim_w[np.argmin(vols)]

        def _stats(w):
            r = float(w @ mean_ret)
            v = float(np.sqrt(w @ cov @ w))
            return r, v, (r - risk_free_rate) / v if v > 0 else 0.0

        ms_r, ms_v, ms_s = _stats(ms_w)
        mv_r, mv_v, mv_s = _stats(mv_w)

        return {
            "max_sharpe": {
                "return": ms_r, "volatility": ms_v, "sharpe": ms_s,
                "weights": {t: float(w) for t, w in zip(tickers, ms_w)},
            },
            "min_volatility": {
                "return": mv_r, "volatility": mv_v, "sharpe": mv_s,
                "weights": {t: float(w) for t, w in zip(tickers, mv_w)},
            },
        }
    except Exception as e:
        logger.error(f"Portfolio optimization failed: {e}")
        return {"error": str(e)}
