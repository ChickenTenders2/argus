import streamlit as st
import pandas as pd
import yfinance as yf
from engine import (
    Config,
    run_scan,
    save_results,
    build_prediction_model,
    add_predictions,
    add_risk_guidance,
    save_journal_entry,
    load_journal,
    list_journals,
    delete_journal,
    sync_journal_to_csv,
    get_market_regime,
    get_db_connection,
    generate_telegram_message,
    monitor_portfolio
)
import os
import re
import requests
import json
from datetime import datetime
import plotly.express as px

# ── Optional enhanced-UI libraries (all fail-safe) ───────
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
    HAS_AGGRID = True
except ImportError:
    HAS_AGGRID = False

try:
    from annotated_text import annotated_text
    HAS_ANNOTATED = True
except ImportError:
    HAS_ANNOTATED = False

try:
    from streamlit_lottie import st_lottie
    HAS_LOTTIE = True
except ImportError:
    HAS_LOTTIE = False

try:
    from streamlit_extras.metric_cards import style_metric_cards
    HAS_EXTRAS = True
except ImportError:
    HAS_EXTRAS = False

PREFS_FILE = "argus_prefs.json"


def format_reasons(r):
    """Deserialise a reasons value from the DB into a bullet-separated string."""
    if isinstance(r, list):
        return " • ".join(str(x) for x in r)
    if isinstance(r, str):
        if r.startswith("["):
            try:
                parsed = json.loads(r)
                return " • ".join(str(x) for x in parsed)
            except (json.JSONDecodeError, ValueError):
                pass
        if " • " in r:
            return r
    return str(r) if r else ""


METRIC_TOOLTIPS = {
    "peg": "Price/Earnings-to-Growth ratio. Below 1.0 suggests the stock may be undervalued relative to its earnings growth rate. The lower, the better.",
    "gross margin": "Gross profit as a % of revenue. Margins above 60% indicate strong pricing power and a durable competitive moat.",
    "rev growth": "Year-over-year revenue growth. ≥30% signals a high-growth business with strong momentum.",
    "fcf positive": "Free Cash Flow is positive — the company generates real cash after capital expenditures, a key sign of financial health and sustainability.",
    "roce": "Return on Capital Employed. Above 20% indicates the business is deploying capital very efficiently — a hallmark of quality compounders.",
    "p/s": "Price-to-Sales ratio. Below 4x suggests the stock is reasonably priced relative to its revenue — a useful valuation check for growth companies.",
    "6mo momentum": "Price is at least 15% higher than 6 months ago. Sustained upward momentum is one of the strongest predictors of continued outperformance.",
    "above 50ma": "Price is above the 50-day moving average. This is a bullish short-term trend signal, confirming buyers are in control.",
    "above 200ma": "Price is above the 200-day moving average. This confirms a healthy long-term uptrend and institutional interest.",
    "volume spike": "Today\u2019s volume is 2x the 30-day average. Unusual volume often signals institutional accumulation or a major catalyst driving breakout interest.",
    "rs vs iwm": "Relative Strength vs the Russell 2000 (IWM). Outperforming by 20%+ over 3 months signals this stock is a true market leader.",
    "inst.": "Low institutional ownership (<40%) means large funds have room to accumulate. Rising institutional interest can be a powerful price catalyst.",
    "cap ": "Market cap in the $50M\u2013$10B sweet spot — large enough to be stable, small enough for significant growth runway before institutional saturation.",
    "persistence bonus": "Argus has flagged this stock 3+ consecutive scans. Repeated high scores across different market days signal consistently strong fundamentals and momentum.",
    "ai sentiment": "AI-analysed news sentiment from Groq/Llama, ranging from -15 (very bearish) to +15 (very bullish). Reflects the qualitative news backdrop.",
    "regime adj": "Score adjusted for the current macro market regime (Bull/Bear/Extreme Fear). Argus multiplies scores up in bull markets and penalises in bear markets.",
}

_METRIC_TOOLTIP_CSS = """
<style>
.argus-badge-wrap { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; }
.argus-badge {
    display: inline-block; position: relative;
    background: #1a2e4a; color: #c9d6e8;
    padding: 3px 10px; border-radius: 12px;
    font-size: 0.78rem; cursor: default; white-space: nowrap;
}
.argus-badge:hover { background: #1e3f68; }
.argus-badge .argus-tip {
    visibility: hidden; opacity: 0;
    background: #1e2a3a; color: #e0e8f0;
    border: 1px solid #334466;
    text-align: left; border-radius: 6px;
    padding: 7px 11px; position: absolute;
    z-index: 9999; bottom: 130%; left: 50%;
    transform: translateX(-50%);
    min-width: 230px; max-width: 280px;
    font-size: 0.73rem; line-height: 1.45;
    box-shadow: 0 4px 12px rgba(0,0,0,0.6);
    white-space: normal; pointer-events: none;
    transition: opacity 0.15s;
}
.argus-badge:hover .argus-tip { visibility: visible; opacity: 1; }
</style>
"""

def _find_metric_tooltip(reason_str):
    """Return the tooltip text for a metric reason string by fuzzy-key lookup."""
    r_lower = reason_str.lower()
    for key, tip in METRIC_TOOLTIPS.items():
        if key in r_lower:
            return tip
    return ""

def render_metric_badges(reasons):
    """Render metric reason badges with CSS hover tooltips."""
    if not reasons:
        return
    st.markdown(_METRIC_TOOLTIP_CSS, unsafe_allow_html=True)
    badges_html = '<div class="argus-badge-wrap">'
    for r in reasons:
        r = r.strip()
        if not r:
            continue
        tip = _find_metric_tooltip(r)
        tip_html = f'<span class="argus-tip">{tip}</span>' if tip else ''
        badges_html += f'<span class="argus-badge">{r}{tip_html}</span>'
    badges_html += '</div>'
    st.markdown(badges_html, unsafe_allow_html=True)


def get_signal_label(score):
    """Return (label, color) conviction signal for a given Argus score."""
    try:
        score = float(score)
    except (TypeError, ValueError):
        return "🔵 Insufficient Data", "gray"
    if score >= 85:
        return "🟢 Strong Buy", "green"
    elif score >= 75:
        return "🔵 Buy", "blue"
    elif score >= 65:
        return "🟡 Moderate Buy", "orange"
    elif score >= 55:
        return "⚪ Hold / Watch", "gray"
    else:
        return "🔴 Avoid for Now", "red"


@st.cache_data(ttl=1800)
def cached_market_regime():
    """Cached wrapper — avoids two yfinance fetches on every Streamlit rerun."""
    return get_market_regime()


@st.cache_data(ttl=3600)
def cached_build_prediction_model(features_file, horizon_days, target_return):
    """Cached wrapper — avoids re-training XGBoost on every page navigation."""
    return build_prediction_model(
        features_file=features_file,
        horizon_days=horizon_days,
        target_return=target_return,
    )


@st.cache_data(ttl=86400)
def _load_lottie(url: str):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def show_aggrid(df, height=380, fit_columns=True, theme=None):
    """AgGrid table with fallback to st.dataframe when library is absent."""
    if not HAS_AGGRID or df.empty:
        st.dataframe(df, use_container_width=True)
        return
    if theme is None:
        try:
            _base = st.get_option("theme.base") or "dark"
        except Exception:
            _base = "dark"
        theme = "balham-dark" if _base == "dark" else "alpine"
    try:
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(sortable=True, filter=True, resizable=True, wrapText=True)
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20)
        gb.configure_grid_options(domLayout="normal")
        AgGrid(
            df,
            gridOptions=gb.build(),
            use_container_width=True,
            height=height,
            theme=theme,
            update_mode=GridUpdateMode.NO_UPDATE,
            allow_unsafe_jscode=False,
            fit_columns_on_grid_load=fit_columns,
        )
    except Exception:
        st.dataframe(df, use_container_width=True)


def get_cached_sector(ticker):
    try:
        conn = get_db_connection()
        try:
            sector_df = pd.read_sql("SELECT sector FROM features WHERE ticker = %(ticker)s ORDER BY id DESC LIMIT 1", conn, params={"ticker": ticker})
        except:
            sector_df = pd.read_sql("SELECT sector FROM features WHERE ticker = ? ORDER BY id DESC LIMIT 1", conn, params=(ticker,))
        conn.close()
        if not sector_df.empty:
            return sector_df.iloc[0]["sector"]
    except:
        pass
    # Basic fallback to yahoo finance info if totally unwatched
    try:
        tkr = yf.Ticker(ticker)
        return tkr.info.get("sector", "Unknown")
    except:
        return "Unknown"

def get_spy_return(start_date_str, end_date_str=None):
    try:
        spy = yf.Ticker("SPY")
        if end_date_str:
            hist = spy.history(start=start_date_str, end=end_date_str)
        else:
            hist = spy.history(start=start_date_str)
            
        if hist.empty:
            return 0.0
            
        start_price = hist["Close"].iloc[0]
        current_price = hist["Close"].iloc[-1]
        return ((current_price - start_price) / start_price) * 100
    except:
        return 0.0

def load_prefs():
    try:
        with open(PREFS_FILE, "r") as f:
            return json.load(f)
    except:
        return {"send_to_telegram": False}

def save_prefs(prefs):
    with open(PREFS_FILE, "w") as f:
        json.dump(prefs, f)

if "prefs" not in st.session_state:
    st.session_state.prefs = load_prefs()

def update_telegram_pref():
    st.session_state.prefs["send_to_telegram"] = st.session_state.send_to_telegram_cb
    save_prefs(st.session_state.prefs)

st.set_page_config(page_title="Argus Dashboard", layout="wide")
st.title("👁 Argus Investment Workstation")

@st.cache_data(ttl=3600)
def fetch_ticker_history(ticker, period="1y"):
    """Fetch historical price data with 1-hour caching to save API calls."""
    try:
        return yf.Ticker(ticker).history(period=period)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def fetch_financial_snapshot(ticker):
    """Fetch key financial metrics."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "P/E Ratio": info.get("trailingPE", "N/A"),
            "Market Cap": f"${info.get('marketCap', 0):,}" if info.get("marketCap") else "N/A",
            "52-Wk High": f"${info.get('fiftyTwoWeekHigh', 0):.2f}" if info.get("fiftyTwoWeekHigh") else "N/A",
            "52-Wk Low": f"${info.get('fiftyTwoWeekLow', 0):.2f}" if info.get("fiftyTwoWeekLow") else "N/A",
            "Div Yield": f"{(info.get('dividendYield', 0) * 100):.2f}%" if info.get("dividendYield") else "N/A"
        }
    except Exception:
        return {}

@st.cache_data(ttl=300)
def _fetch_gbpusd():
    """Fetch live GBP/USD exchange rate. Returns float (e.g. 1.27)."""
    try:
        fx = yf.download("GBPUSD=X", period="2d", progress=False, auto_adjust=True)
        if not fx.empty:
            col = fx["Close"] if "Close" in fx.columns else fx.iloc[:, 0]
            return float(col.dropna().iloc[-1])
    except Exception:
        pass
    return 1.27  # safe fallback

@st.cache_data(ttl=300)
def fetch_current_prices(tickers):
    """Bulk fetch current prices converted to GBP.
    - US/other tickers: yfinance USD price ÷ GBPUSD rate
    - UK (.L) tickers:  yfinance pence price ÷ 100
    """
    if not tickers: return {}
    prices = {}
    gbpusd = _fetch_gbpusd()
    try:
        data = yf.download(list(tickers), period="2d", progress=False, auto_adjust=True)
        if data.empty:
            return prices
        raw = {}
        if isinstance(data.columns, pd.MultiIndex):
            top_level = data.columns.get_level_values(0)
            if "Close" not in top_level:
                return prices
            close_df = data["Close"]
            for t in tickers:
                if t in close_df.columns:
                    val = close_df[t].dropna()
                    if not val.empty:
                        raw[t] = float(val.iloc[-1])
        else:
            t = tickers[0]
            if "Close" in data.columns and not data["Close"].dropna().empty:
                raw[t] = float(data["Close"].dropna().iloc[-1])
        for t, price in raw.items():
            if t.upper().endswith(".L"):
                prices[t] = round(price / 100, 4)   # pence → GBP
            else:
                prices[t] = round(price / gbpusd, 4)  # USD → GBP
    except Exception:
        pass
    return prices

@st.cache_data(ttl=14400)
def fetch_enrichment(ticker: str) -> dict:
    """Fetch earnings date, analyst price target, and insider activity via yfinance (4-hour cache)."""
    result = {"earnings_date": None, "analyst_target": None, "analyst_upside": None, "insider_net": None}
    try:
        t = yf.Ticker(ticker)
        info = t.info
        # Analyst consensus target
        target = info.get("targetMeanPrice")
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if target and price:
            result["analyst_target"] = round(float(target), 2)
            result["analyst_upside"] = round(((float(target) - float(price)) / float(price)) * 100, 1)
        # Next earnings date
        try:
            cal = t.calendar
            if isinstance(cal, dict):
                ed_list = cal.get("Earnings Date", [])
                if ed_list:
                    ed_first = ed_list[0] if isinstance(ed_list, list) else ed_list
                    ed_ts = pd.Timestamp(ed_first)
                    if ed_ts > pd.Timestamp.now():
                        result["earnings_date"] = ed_ts.strftime("%d %b")
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                if "Earnings Date" in cal.index:
                    ed_val = cal.loc["Earnings Date"].iloc[0]
                    ed_ts = pd.Timestamp(ed_val)
                    if ed_ts > pd.Timestamp.now():
                        result["earnings_date"] = ed_ts.strftime("%d %b")
        except Exception:
            pass
        # Insider net activity (last 90 days)
        try:
            ins_df = t.insider_transactions
            if ins_df is not None and not ins_df.empty:
                date_col = next((c for c in ins_df.columns if "date" in c.lower()), None)
                tx_col = next((c for c in ins_df.columns if "transaction" in c.lower() or "type" in c.lower()), None)
                if date_col:
                    ins_df[date_col] = pd.to_datetime(ins_df[date_col], errors="coerce")
                    cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
                    recent = ins_df[ins_df[date_col] >= cutoff]
                else:
                    recent = ins_df.head(20)
                if not recent.empty and tx_col:
                    col_vals = recent[tx_col].astype(str)
                    buys = col_vals.str.contains("Buy|Purchase|Acqui", case=False, na=False).sum()
                    sells = col_vals.str.contains("Sell|Sale|Dispos", case=False, na=False).sum()
                    if buys > sells and buys > 0:
                        result["insider_net"] = "🟢 Insider Buyer"
                    elif sells > buys and sells > 0:
                        result["insider_net"] = "🔴 Insider Seller"
        except Exception:
            pass
    except Exception:
        pass
    return result


@st.cache_data(ttl=3600)
def fetch_ticker_news(ticker: str, max_items: int = 5) -> list:
    """Fetch recent news headlines for a ticker via yfinance (1-hour cache)."""
    items = []
    try:
        news_raw = yf.Ticker(ticker).news
        if not news_raw:
            return items
        for item in (news_raw or [])[:max_items]:
            if not isinstance(item, dict):
                continue
            title, link, publisher, pub_time = "", "", "", ""
            if "content" in item:
                c = item["content"]
                title = c.get("title", "")
                url_info = c.get("canonicalUrl", {})
                link = url_info.get("url", "") if isinstance(url_info, dict) else str(url_info)
                provider = c.get("provider", {})
                publisher = provider.get("displayName", "") if isinstance(provider, dict) else str(provider)
                pub_date = c.get("pubDate", "")
                if pub_date:
                    try:
                        pub_time = pd.Timestamp(pub_date).strftime("%d %b")
                    except Exception:
                        pass
            else:
                title = item.get("title", "")
                link = item.get("link", "")
                publisher = item.get("publisher", "")
                ts = item.get("providerPublishTime", 0)
                if ts:
                    try:
                        pub_time = datetime.fromtimestamp(int(ts)).strftime("%d %b")
                    except Exception:
                        pass
            if title:
                items.append({"title": title, "link": link, "publisher": publisher, "pub_time": pub_time})
    except Exception:
        pass
    return items


def safe_line_chart(data, y_label="value"):
    """
    Renders an interactive line chart using Plotly Express for smooth zooming
    and a reliable 'Autoscale' / 'Reset Axes' button.
    """
    try:
        import plotly.express as px
        if isinstance(data, pd.Series):
            df = data.reset_index()
            x_col = df.columns[0]
            y_col = df.columns[1]
        else:
            df = data.copy()
            if df.index.name or not isinstance(df.index, pd.RangeIndex):
                df = df.reset_index()
            x_col = df.columns[0]
            y_col = df.columns[1]

        fig = px.line(df, x=x_col, y=y_col)
        fig.update_layout(
            xaxis_title="",
            yaxis_title=y_label.capitalize(),
            margin=dict(l=0, r=0, t=10, b=0),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning("Interactive chart unavailable; showing data table instead.")
        if isinstance(data, pd.Series):
            fallback = data.reset_index()
            fallback.columns = ["x", y_label]
            st.dataframe(fallback, use_container_width=True)
        else:
            st.dataframe(data, use_container_width=True)

def format_pct_columns(df, cols):
    """Coerce columns to numeric before % formatting to avoid dtype errors."""
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = (out[col] * 100).round(1)
    return out

def send_telegram_message(token, chat_id, message):
    from datetime import datetime
    try:
        with open("argus_alerts_log.txt", "a", encoding="utf-8") as f:
            f.write(f"--- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n{message}\n\n")
    except Exception:
        pass

    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
        return True
    except Exception:
        return False

def apply_preset():
    preset = st.session_state.preset_selector
    st.session_state.price_ceiling = 0.0
    if preset == "High Conviction":
        st.session_state.preset_desc = "**Use when:** You want the fewest, highest-quality picks only. Ultra-strict score filter (≥90) in a strong bull market where you are comfortable concentrating capital into 1–3 exceptional setups instead of diversifying broadly."
        st.session_state.min_score = 90
        st.session_state.horizon_days = 63
        st.session_state.target_return = 15
        st.session_state.risk_per_trade_pct = 1.50
        st.session_state.max_position_pct = 15.00
        st.session_state.vol_floor = 500000
        st.session_state.price_floor = 2.0
    elif preset == "Liquidity Focus":
        st.session_state.preset_desc = "**Use when:** You need to enter and exit positions quickly without slippage (e.g. trading large account sizes). Restricts picks to only highly liquid, large-cap names with ≥2M average daily volume."
        st.session_state.min_score = 65
        st.session_state.horizon_days = 63
        st.session_state.target_return = 10
        st.session_state.risk_per_trade_pct = 0.75
        st.session_state.max_position_pct = 10.00
        st.session_state.vol_floor = 2000000
        st.session_state.price_floor = 10.00
    elif preset == "Momentum Sprint":
        st.session_state.preset_desc = "**Use when:** The market is trending hard and you want to catch explosive short-term breakouts. 21-day horizon targets +30% moves. Best used after confirmed momentum weeks, not during chop or reversal."
        st.session_state.min_score = 72
        st.session_state.horizon_days = 21
        st.session_state.target_return = 30
        st.session_state.risk_per_trade_pct = 1.25
        st.session_state.max_position_pct = 8.00
        st.session_state.vol_floor = 500000
        st.session_state.price_floor = 2.0
    elif preset == "Capital Preservation":
        st.session_state.preset_desc = "**Use when:** VIX is elevated, SPY is below its 200-day MA, or you are in a confirmed bear market. Minimum risk per trade (0.15%), very high score threshold (≥85), and a 6-month horizon to weather volatility."
        st.session_state.min_score = 85
        st.session_state.horizon_days = 126
        st.session_state.target_return = 3
        st.session_state.risk_per_trade_pct = 0.15
        st.session_state.max_position_pct = 5.00
        st.session_state.vol_floor = 500000
        st.session_state.price_floor = 2.0
    elif preset == "Small Cap Hunter":
        st.session_state.preset_desc = "**Use when:** You want exposure to high-growth micro/small-cap names. Lowers price floor to $0.50 and volume floor to 100K. Best used in a bull market where risk appetite is high. Expect higher volatility."
        st.session_state.min_score = 68
        st.session_state.horizon_days = 84
        st.session_state.target_return = 40
        st.session_state.risk_per_trade_pct = 0.50
        st.session_state.max_position_pct = 5.00
        st.session_state.vol_floor = 100000
        st.session_state.price_floor = 0.50
    elif preset == "Earnings Season":
        st.session_state.preset_desc = "**Use when:** Earnings reports are imminent (typically Jan, Apr, Jul, Oct). 21-day horizon captures pre-earnings momentum and post-earnings gap moves. Tighter risk rules reduce exposure to surprise miss gaps."
        st.session_state.min_score = 65
        st.session_state.horizon_days = 21
        st.session_state.target_return = 12
        st.session_state.risk_per_trade_pct = 0.75
        st.session_state.max_position_pct = 6.00
        st.session_state.vol_floor = 500000
        st.session_state.price_floor = 2.0
    elif preset == "Swing Recovery":
        st.session_state.preset_desc = "**Use when:** The market has pulled back significantly (10–20%) but fundamentals remain intact. Targets beaten-down quality tickers expecting mean-reversion over 6 weeks. Score ≥70 filters out genuine deteriorations."
        st.session_state.min_score = 70
        st.session_state.horizon_days = 42
        st.session_state.target_return = 15
        st.session_state.risk_per_trade_pct = 0.60
        st.session_state.max_position_pct = 7.00
        st.session_state.vol_floor = 500000
        st.session_state.price_floor = 2.0
    elif preset == "Aggressive Growth":
        st.session_state.preset_desc = "**Use when:** Market is in a confirmed bull run and you are comfortable with higher risk per trade (1%). Targets +20% moves over 6 weeks. Best suited for growth-oriented accounts not needing capital preservation."
        st.session_state.min_score = 70
        st.session_state.horizon_days = 42
        st.session_state.target_return = 20
        st.session_state.risk_per_trade_pct = 1.0
        st.session_state.max_position_pct = 8.00
        st.session_state.vol_floor = 500000
        st.session_state.price_floor = 2.0
    elif preset == "Bear Market Defense":
        st.session_state.preset_desc = "**Use when:** SPY is below the 200-day MA or a recession is likely. Extremely low risk (0.25%), high score threshold (≥80), and a 3-month horizon. Focuses on defensive names that hold value in downturns."
        st.session_state.min_score = 80
        st.session_state.horizon_days = 84
        st.session_state.target_return = 5
        st.session_state.risk_per_trade_pct = 0.25
        st.session_state.max_position_pct = 5.00
        st.session_state.vol_floor = 500000
        st.session_state.price_floor = 2.0
        st.session_state.price_ceiling = 0.0
    elif preset == "Penny Stock High Risk ($1-$10)":
        st.session_state.preset_desc = "**Use when:** You want speculative, high-reward plays with a small, dedicated risk allocation. Constrained to $1–$10 stocks with the highest score-to-volatility ratio. Only deploy a small fraction of your total portfolio here."
        st.session_state.min_score = 60
        st.session_state.horizon_days = 21
        st.session_state.target_return = 50
        st.session_state.risk_per_trade_pct = 2.0
        st.session_state.max_position_pct = 5.00
        st.session_state.vol_floor = 100000
        st.session_state.price_floor = 1.0
        st.session_state.price_ceiling = 10.0
    elif preset == "Default":
        st.session_state.preset_desc = "**Use when:** Market conditions are neutral or you are unsure which preset to choose. Score ≥65, 63-day horizon, 0.75% risk per trade. A well-rounded starting point for most market environments."
        st.session_state.min_score = 65
        st.session_state.horizon_days = 63
        st.session_state.target_return = 10
        st.session_state.risk_per_trade_pct = 0.75
        st.session_state.max_position_pct = 8.00
        st.session_state.vol_floor = 500000
        st.session_state.price_floor = 2.0
        st.session_state.price_ceiling = 0.0

if "price_ceiling" not in st.session_state:
    st.session_state.price_ceiling = 0.0

if "min_score" not in st.session_state:
    st.session_state.min_score = 65
def nav_to_ticker(t):
    st.session_state["selected_ticker"] = t
    st.session_state["main_tabs"] = "Ticker Detail"


@st.dialog("⚡ Quick Log Trade")
def quick_log_dialog(ticker, price, stop_loss_pct, take_profit_pct):
    """Minimal 2-field dialog pre-filled from scan data."""
    st.markdown(f"**{ticker}** — Entry price: **${price:.2f}**")
    with st.form("quick_log_form"):
        col1, col2 = st.columns(2)
        action = col1.selectbox("Action", ["BUY", "SCALE_IN", "PASS"])
        shares = col2.number_input("Shares", min_value=0.0, value=1.0, step=1.0)
        notes = st.text_input("Notes (optional)", placeholder="e.g. Breakout entry")
        st.caption(
            f"Pre-filled from scan: Entry **${price:.2f}** · "
            f"Stop-loss **-{stop_loss_pct:.1f}%** · Take-profit **+{take_profit_pct:.1f}%**"
        )
        submitted = st.form_submit_button("💾 Save Entry", use_container_width=True)
        if submitted:
            _active_j = st.session_state.get("active_journal_selector", "Default")
            if _active_j == "All":
                _active_j = "Default"
            save_journal_entry(
                Config().JOURNAL_FILE,
                {
                    "ticker": ticker,
                    "action": action,
                    "scan_date": datetime.now().strftime("%Y-%m-%d"),
                    "entry_price": price,
                    "shares": shares,
                    "position_size_pct": 0.0,
                    "stop_loss_pct": stop_loss_pct,
                    "take_profit_pct": take_profit_pct,
                    "notes": notes,
                    "journal_name": _active_j,
                },
            )
            st.success(f"Logged {action} {shares:.0f}× {ticker} → **{_active_j}** portfolio.")
            st.rerun()


def display_cards(df):
    """Render a DataFrame of picks as visual cards instead of a wide table."""
    if df.empty:
        st.info("No picks to display.")
        return
        
    cols = st.columns(3)
    for i, (_, row) in enumerate(df.iterrows()):
        with cols[i % 3]:
            # Instead of a static container, use an expander collapsed by default
            tier_str = row.get("tier", "")
            tier_icon = "🟢" if "HIGH CONVICTION" in str(tier_str) else "🟡" if tier_str else ""
            label = f"{tier_icon} {row['ticker']} · Score: {row['score']}/100"
            with st.expander(label, expanded=False):
                if "tier" in row and pd.notna(row["tier"]):
                    color = "green" if "HIGH CONVICTION" in row["tier"] else "orange"
                    st.markdown(f"**:{color}[{row['tier']}]**")

                sig_label, sig_color = get_signal_label(row.get("score", 0))
                st.markdown(f"**:{sig_color}[{sig_label}]**")
                
                # Core Financials
                m1, m2 = st.columns(2)
                price = row.get("price", 0)
                mkt_cap = row.get("mkt_cap", "N/A")
                m1.metric("Price", f"${price:.2f}" if pd.notna(price) else "N/A")
                m2.metric("Mkt Cap", mkt_cap if pd.notna(mkt_cap) else "N/A")
                
                # ML Prob / Returns if available
                if "prob_upside" in row and pd.notna(row["prob_upside"]):
                    m3, m4 = st.columns(2)
                    m3.metric("ML Upside", f"{row['prob_upside']*100:.1f}%" if isinstance(row['prob_upside'], float) else row['prob_upside'])
                    if "scenario_bull" in row and pd.notna(row["scenario_bull"]):
                        bull_val = f"+{row['scenario_bull']*100:.1f}%" if isinstance(row['scenario_bull'], float) else row['scenario_bull']
                        m4.metric("Bull Target", bull_val)
                        
                if "Return (%)" in row and pd.notna(row["Return (%)"]):
                    ret_val = row["Return (%)"]
                    st.metric("Performance Since Scan", f"{ret_val:+.2f}%", delta=ret_val)

                # Metrics/Reasons display (bullet points or badges)
                if "reasons" in row and row["reasons"]:
                    st.markdown("##### Key Metrics")
                    reasons_raw = row["reasons"]
                    if isinstance(reasons_raw, list):
                        reasons = reasons_raw
                    elif isinstance(reasons_raw, str):
                        if reasons_raw.startswith("["):
                            try:
                                reasons = json.loads(reasons_raw)
                            except (json.JSONDecodeError, ValueError):
                                reasons = [reasons_raw]
                        elif " • " in reasons_raw:
                            reasons = reasons_raw.split(" • ")
                        else:
                            reasons = [reasons_raw]
                    else:
                        reasons = []

                    if isinstance(reasons, list):
                        render_metric_badges(reasons)
                    else:
                        render_metric_badges([str(reasons)]) if reasons else None

                # ── Enrichment: earnings, analyst target, insider ──
                try:
                    _enr = fetch_enrichment(row["ticker"])
                    _enr_parts = []
                    if _enr.get("earnings_date"):
                        _enr_parts.append(f"📅 Earnings: {_enr['earnings_date']}")
                    if _enr.get("analyst_target") and _enr.get("analyst_upside") is not None:
                        _up_icon = "🟢" if _enr["analyst_upside"] > 0 else "🔴"
                        _enr_parts.append(
                            f"{_up_icon} Target: ${_enr['analyst_target']} ({_enr['analyst_upside']:+.1f}%)"
                        )
                    if _enr.get("insider_net"):
                        _enr_parts.append(_enr["insider_net"])
                    if _enr_parts:
                        st.caption(" · ".join(_enr_parts))
                except Exception:
                    pass

                _bc1, _bc2 = st.columns(2)
                with _bc1:
                    st.button(
                        "Deep Dive",
                        key=f"dd_{row['ticker']}_{i}_{row.get('scan_date', 'temp')}",
                        use_container_width=True,
                        on_click=nav_to_ticker,
                        args=(row["ticker"],),
                    )
                with _bc2:
                    if st.button(
                        "⚡ Log Trade",
                        key=f"ql_{row['ticker']}_{i}_{row.get('scan_date', 'temp')}",
                        use_container_width=True,
                    ):
                        quick_log_dialog(
                            ticker=row["ticker"],
                            price=float(row.get("price") or 0),
                            stop_loss_pct=float(row.get("stop_loss_pct") or 8.0),
                            take_profit_pct=float(row.get("take_profit_pct") or 25.0),
                        )

if "horizon_days" not in st.session_state:
    st.session_state.horizon_days = 63
if "target_return" not in st.session_state:
    st.session_state.target_return = 10
if "risk_per_trade_pct" not in st.session_state:
    st.session_state.risk_per_trade_pct = 0.75
if "max_position_pct" not in st.session_state:
    st.session_state.max_position_pct = 8.00
if "vol_floor" not in st.session_state:
    st.session_state.vol_floor = 500000
if "price_floor" not in st.session_state:
    st.session_state.price_floor = 2.00
if "preset_desc" not in st.session_state:
    st.session_state.preset_desc = "A balanced setup suitable for normal market conditions."

# ── Auto-Preset: choose the best preset for today's regime once per session ──
if "auto_preset_applied" not in st.session_state:
    try:
        _r0 = cached_market_regime()
        _rn0 = _r0.get("regime", "Neutral")
        _pmap0 = {
            "Extreme Fear": "Capital Preservation",
            "Bear": "Bear Market Defense",
            "Neutral": "Default",
            "Bull": "High Conviction",
        }
        _ap0 = _pmap0.get(_rn0, "Default")
        st.session_state["preset_selector"] = _ap0
        apply_preset()
        st.session_state["auto_preset_label"] = (
            f"🤖 Auto-selected **{_ap0}** based on current **{_rn0}** regime."
        )
    except Exception:
        st.session_state["auto_preset_label"] = ""
    st.session_state["auto_preset_applied"] = True

with st.sidebar:
    st.header("Global Presets")
    
    preset_options = [
        "Default", "High Conviction", "Liquidity Focus", 
        "Momentum Sprint", "Capital Preservation", 
        "Small Cap Hunter", "Earnings Season", 
        "Swing Recovery", "Aggressive Growth", "Bear Market Defense", 
        "Penny Stock High Risk ($1-$10)"
    ]
    st.selectbox("Select Preset", preset_options, key="preset_selector", on_change=apply_preset)
    if st.session_state.get("auto_preset_label"):
        st.caption(st.session_state["auto_preset_label"])

    st.markdown(
        f"<div style='background:#2b2b2b;border-left:3px solid #888;border-radius:4px;padding:10px 14px;font-size:0.82rem;color:#ccc;margin-bottom:6px;'>{st.session_state.preset_desc}</div>",
        unsafe_allow_html=True,
    )

    st.header("Quick Scan Settings")
    min_score = st.slider("Minimum Score", 50, 95, key="min_score")
    scan_limit = st.slider("Universe size (tickers to scan)", 50, 400, 200, step=50)

    with st.expander("Advanced Filters & Constraints"):
        price_floor = st.number_input("Price Floor ($)", min_value=0.0, max_value=100.0, key="price_floor",
            help="Minimum stock price to include. Filters out very cheap penny stocks that may have high volatility or low quality. Default: $2.00.")
        price_ceiling = st.number_input("Price Ceiling ($) [0 = No Limit]", min_value=0.0, max_value=1000.0, key="price_ceiling",
            help="Maximum stock price to include. Set to 0 to disable. Useful for budget-constrained strategies (e.g. Penny Stock High Risk preset sets this to $10).")
        vol_floor = st.number_input("Volume Floor (avg daily)", step=100_000, min_value=0, key="vol_floor",
            help="Minimum average daily trading volume. Higher values ensure you can enter/exit positions without slippage. Default: 500,000 shares/day.")

    with st.expander("ML Prediction Horizons"):
        horizon_days = st.selectbox("Horizon days", [21, 42, 63, 84, 126], key="horizon_days",
            help="The future time window (in trading days) the ML model uses to assess whether a stock hit the target return. 63 days ≈ 1 quarter. Longer horizons capture slower trends; shorter ones target quick breakouts.")
        target_return = st.slider("Target return (%)", 1, 100, key="target_return",
            help="The minimum % gain the ML model considers a 'hit'. A stock is labeled a success if it returns ≥ this value within the Horizon Days. Higher targets are harder to hit but filter for stronger momentum.") / 100.0

    with st.expander("Advanced Risk Rules"):
        risk_per_trade_pct = st.slider("Risk per trade (% of portfolio)", 0.10, 5.00, step=0.05, key="risk_per_trade_pct",
            help="The maximum % of your total portfolio you are willing to lose if a position hits its stop-loss. Argus uses this to compute the suggested position size. Lower = more conservative. Default: 0.75%.")
        max_position_pct = st.slider("Max position size (%)", 1.0, 30.0, step=0.5, key="max_position_pct",
            help="Hard cap on how large any single position can be as a % of your total portfolio. Prevents over-concentration even when the risk formula suggests a larger size. Default: 8.0%.")

    st.divider()
    send_to_telegram = st.checkbox(
        "Send manual run to Telegram", 
        value=st.session_state.prefs.get("send_to_telegram", False), 
        key="send_to_telegram_cb", 
        on_change=update_telegram_pref
    )

    config = Config(MIN_SCORE=min_score, PRICE_FLOOR=price_floor, PRICE_CEILING=(price_ceiling if price_ceiling > 0 else None), VOL_FLOOR=vol_floor)

model = cached_build_prediction_model(
    features_file=config.FEATURES_FILE,
    horizon_days=horizon_days,
    target_return=target_return,
)

# Load data from SQLite
conn = get_db_connection()
try:
    history_df = pd.read_sql("SELECT * FROM results", conn)
except Exception:
    history_df = pd.DataFrame()
conn.close()

if os.path.exists(config.RESULTS_FILE):
    latest_df = pd.read_csv(config.RESULTS_FILE)
else:
    latest_df = pd.DataFrame()

# ── Auto-scan on load: fire once per session if no scan exists for today ──
_today_str = datetime.now().strftime("%Y-%m-%d")
_latest_scan_date = (
    str(latest_df["scan_date"].iloc[0])[:10]
    if not latest_df.empty and "scan_date" in latest_df.columns
    else ""
)
_auto_scan_ran = False
if _latest_scan_date != _today_str and not st.session_state.get("_auto_scan_done_today", False):
    with st.status("🔄 Running today's auto-scan…", expanded=True) as _auto_st:
        st.write(f"No scan found for today ({_today_str}). Starting automated scan…")
        try:
            _auto_payload = run_scan(config=config, scan_limit=scan_limit, update_memory=True)
            save_results(
                results=_auto_payload["results"],
                scan_date=_auto_payload["scan_date"],
                scan_timestamp=_auto_payload["scan_timestamp"],
                run_type="auto",
                latest_file=config.RESULTS_FILE,
                history_file=config.RESULTS_HISTORY_FILE,
                write_latest=True,
                feature_file=config.FEATURES_FILE,
            )
            st.session_state["_auto_scan_done_today"] = True
            _n_auto = len(_auto_payload["results"])
            _auto_st.update(label=f"✅ Auto-scan complete — {_n_auto} picks found.", state="complete")
            _auto_scan_ran = True
        except Exception as _ae:
            st.session_state["_auto_scan_done_today"] = True
            _auto_st.update(label=f"⚠️ Auto-scan failed: {_ae}", state="error")

if _auto_scan_ran:
    st.rerun()

# To restore Portfolio Optimizer, add "Portfolio Optimizer" back to this array
tab_options = ["Overview", "Ticker Detail", "Manual Run", "History", "Journal", "Prediction Model", "Alerts Log", "Help", "Prompts"]
active_tab = st.radio("Navigation", tab_options, key="main_tabs", horizontal=True, label_visibility="collapsed")
st.markdown("<br>", unsafe_allow_html=True)


if active_tab == "Overview":
    st.subheader("📡 Latest Scheduled Scan")

    # ── Today's Briefing — 3-column card ─────────────────────────────────────
    regime = cached_market_regime()
    regime_color_map = {"Bull": "green", "Neutral": "blue", "Bear": "orange", "Extreme Fear": "red"}
    r_color = regime_color_map.get(regime["regime"], "white")
    vix_val = regime.get("vix_level")
    vix_trend = regime.get("vix_trend", "stable")
    vix_arrow = "↑" if vix_trend == "rising" else ("↓" if vix_trend == "falling" else "→")
    vix_color = "red" if vix_val and vix_val > 25 else ("orange" if vix_val and vix_val > 18 else "green")
    vix_str = f"{vix_val:.1f} {vix_arrow}" if vix_val else "N/A"
    gap = regime.get("spy_ma200_gap_pct")
    gap_color = "green" if gap and gap > 3 else ("orange" if gap and gap > 0 else "red")
    gap_label = f"{gap:+.1f}%" if gap is not None else "N/A"
    scan_date_display = latest_df["scan_date"].iloc[0] if not latest_df.empty and "scan_date" in latest_df.columns else "N/A"

    try:
        _briefing_alerts = monitor_portfolio()
    except Exception:
        _briefing_alerts = []

    _bc1, _bc2, _bc3 = st.columns(3)

    with _bc1:
        with st.container(border=True):
            st.markdown("**📡 Market Regime**")
            st.markdown(f":{r_color}[**{regime['regime']}**] · *{regime['reason']}* · Mult **{regime['multiplier']}x**")
            st.caption(
                f"VIX: :{vix_color}[**{vix_str}**] · SPY vs 200MA: :{gap_color}[**{gap_label}**]"
            )
            st.caption(f"Last scan: **{scan_date_display}** · Refreshes every hour")

    with _bc2:
        with st.container(border=True):
            st.markdown("**🏆 Top Picks Today**")
            if not latest_df.empty:
                _top3 = latest_df.sort_values("score", ascending=False).head(3)
                for _, _tr in _top3.iterrows():
                    _sig_lbl, _sig_col = get_signal_label(_tr.get("score", 0))
                    _sig_icon = _sig_lbl.split()[0]
                    st.markdown(
                        f"**{_tr['ticker']}** &nbsp; :{_sig_col}[{_sig_icon} {_tr['score']:.0f}]"
                        + (f" &nbsp; *{_tr.get('sector', '')}*" if _tr.get("sector") else ""),
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No scan data yet — run a scan to see picks.")

    with _bc3:
        with st.container(border=True):
            st.markdown("**⚠️ Portfolio Alerts**")
            if _briefing_alerts:
                for _a in _briefing_alerts[:3]:
                    st.warning(_a, icon="🚨")
                if len(_briefing_alerts) > 3:
                    st.caption(f"…and {len(_briefing_alerts) - 3} more. Check Journal tab.")
            else:
                st.success("All positions within bounds ✅", icon="✅")

    # Bear-transition inline warnings
    if gap is not None:
        _vix_num = vix_val or 0
        if gap <= 0:
            st.warning("🐻 **Bear Market Active.** SPY crossed below its 200-day MA. Consider switching to **Capital Preservation** or **Bear Market Defense** preset.")
        elif gap <= 4 and vix_trend == "rising":
            st.warning(f"⚠️ **Bear Transition Risk:** SPY is only {gap:.1f}% above its 200-day MA with VIX rising. A ~{gap:.1f}% pullback would trigger regime shift. Reduce position sizes.")
        elif gap <= 8 and _vix_num > 20:
            st.info(f"📊 **Caution zone:** SPY {gap:.1f}% above 200MA with VIX at {_vix_num:.1f}. Not yet bearish, but elevated volatility warrants attention.")
    st.markdown("---")

    if latest_df.empty:
        st.info("No latest scheduled results yet.")
    else:
        latest_view = latest_df.copy()
        latest_view = add_predictions(latest_view, model)
        latest_view = add_risk_guidance(
            latest_view,
            model,
            risk_per_trade_pct=risk_per_trade_pct,
            max_position_pct=max_position_pct,
        )
        scan_date = latest_view["scan_date"].iloc[0] if "scan_date" in latest_view.columns else "Unknown"
        st.caption(f"Scan date: {scan_date} · {len(latest_view)} picks")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            with st.container(border=True):
                st.metric("Picks", str(len(latest_view)), help="Total Passed")
        with c2:
            with st.container(border=True):
                st.metric("High Conviction", str(int((latest_view["tier"] == "🟢 HIGH CONVICTION").sum())), help="Score >= 80")
        with c3:
            with st.container(border=True):
                st.metric("Avg Score", f"{latest_view['score'].mean():.1f}", help="Argus Rating")
        with c4:
            with st.container(border=True):
                if "prob_upside" in latest_view.columns and latest_view["prob_upside"].notna().any():
                    st.metric("Avg Upside Prob", f"{latest_view['prob_upside'].mean() * 100:.1f}%", help="ML Projected")
                else:
                    st.metric("Avg Upside Prob", "N/A", help="ML Projected")

        if HAS_EXTRAS:
            try:
                style_metric_cards(border_left_color="#00b4d8", border_color="#262730", background_color="#0e1117")
            except Exception:
                pass

        display_cols = [
            "ticker", "sector", "score", "tier", "reasons", "price", "mkt_cap",
            "prob_upside", "scenario_bear", "scenario_base", "scenario_bull",
            "confidence", "suggested_position_pct", "stop_loss_pct", "take_profit_pct", "entry_style",
        ]
        latest_view = format_pct_columns(
            latest_view,
            ["prob_upside", "scenario_bear", "scenario_base", "scenario_bull"],
        )
        
        subset_view = latest_view[[c for c in display_cols if c in latest_view.columns]].copy()
        
        # Display the visual card grid instead of a flat table
        display_cards(subset_view)

        with st.expander("📋 Copy Tickers"):
            _ticker_str = ", ".join(subset_view["ticker"].tolist())
            st.caption(f"{len(subset_view)} tickers — click the copy icon on the right →")
            st.code(_ticker_str, language=None)

        with st.expander("Show Raw Data Table"):
            if "reasons" in subset_view.columns:
                subset_view["reasons"] = subset_view["reasons"].apply(format_reasons)
            
            st.dataframe(
                subset_view, 
                use_container_width=True,
                column_config={
                    "score": st.column_config.ProgressColumn(
                        "Score",
                        help="Argus Score Rating",
                        format="%d",
                        min_value=0,
                        max_value=100,
                    ),
                    "prob_upside": st.column_config.ProgressColumn(
                        "Upside Prob (%)",
                        help="ML Upside Probability",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "reasons": st.column_config.TextColumn(
                        "Metrics & Reasons",
                        help="Fundamental and Technical Drivers",
                        width="large"
                    )
                }
            )

        st.markdown("### 🔍 Stock Deep Dive")
        c1, c2 = st.columns([1, 10])
        with c1:
            diveticker = st.selectbox("Select Ticker", subset_view["ticker"].tolist(), key="deep_dive_selectbox", label_visibility="collapsed")
        with c2:
            st.button(
                "Analyze", 
                key="btn_deep_dive_analyze",
                on_click=nav_to_ticker,
                args=(st.session_state.get("deep_dive_selectbox", ""),)
            )

if active_tab == "History":
    st.subheader("🗂 History")
    if history_df.empty:
        st.info("No history file yet.")
    else:
        history_df["scan_date"] = pd.to_datetime(history_df["scan_date"], errors="coerce")
        run_type_opts = ["all"] + sorted(history_df["run_type"].dropna().unique().tolist())
        selected_run_type = st.selectbox("Run type", run_type_opts)
        if selected_run_type != "all":
            filtered = history_df[history_df["run_type"] == selected_run_type].copy()
        else:
            filtered = history_df.copy()

        filtered = filtered.dropna(subset=["scan_date"])
        if filtered.empty:
            st.warning("No rows after filters.")
        else:
            filtered["scan_day"] = filtered["scan_date"].dt.date
            daily = (
                filtered.groupby("scan_day", as_index=False)
                .agg(picks=("ticker", "count"), top_score=("score", "max"), avg_score=("score", "mean"))
                .sort_values("scan_day", ascending=False)
            )
            daily["scan_day"] = daily["scan_day"].astype(str)
            daily["avg_score"] = daily["avg_score"].round(1)
            daily["top_score"] = daily["top_score"].round(1)
            show_aggrid(daily, height=280)

            st.caption("Score trend")
            trend = (
                filtered.groupby("scan_day", as_index=False)
                .agg(avg_score=("score", "mean"))
                .sort_values("scan_day")
            )
            safe_line_chart(trend.set_index("scan_day")["avg_score"], y_label="avg_score")

            date_options = [d.strftime("%Y-%m-%d") for d in sorted(filtered["scan_day"].unique(), reverse=True)]
            
            hc1, hc2 = st.columns([5, 1], vertical_alignment="bottom")
            with hc1:
                selected_day = st.selectbox("Day details", date_options)
            with hc2:
                if st.button("🗑️ Delete Day", help="Delete all scan history for the selected day", use_container_width=True):
                    try:
                        conn = get_db_connection()
                        # delete from results and features tables for that specific scan_date starting with selected_day
                        try:
                            from sqlalchemy import text
                            conn.execute(text("DELETE FROM results WHERE scan_date LIKE :sd"), {"sd": selected_day + "%"})
                            conn.execute(text("DELETE FROM features WHERE scan_date LIKE :sd"), {"sd": selected_day + "%"})
                        except:
                            conn.execute("DELETE FROM results WHERE scan_date LIKE ?", (selected_day + "%",))
                            conn.execute("DELETE FROM features WHERE scan_date LIKE ?", (selected_day + "%",))
                        conn.commit()
                        conn.close()
                        st.success(f"Deleted history for {selected_day}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting: {e}")

            day_df = filtered[filtered["scan_date"].dt.strftime("%Y-%m-%d") == selected_day].sort_values("score", ascending=False).copy()
            
            with st.spinner("Fetching current prices to calculate return..."):
                current_prices = fetch_current_prices(day_df["ticker"].dropna().unique().tolist())
                if current_prices and "price" in day_df.columns:
                    day_df["Current Price"] = day_df["ticker"].map(current_prices).round(2)
                    day_df["Return (%)"] = (((day_df["Current Price"] - day_df["price"]) / day_df["price"]) * 100).round(2)
                    
            if "reasons" in day_df.columns:
                day_df["reasons"] = day_df["reasons"].apply(format_reasons)
            
            # Show cards for better readability
            display_cards(day_df)
            
            with st.expander("Show History Table"):
                if HAS_AGGRID:
                    show_aggrid(day_df, height=320)
                else:
                    try:
                        _theme_base = st.get_option("theme.base") or "dark"
                    except Exception:
                        _theme_base = "dark"
                    if "Return (%)" in day_df.columns and _theme_base == "light":
                        st.dataframe(day_df.style.background_gradient(subset=["Return (%)"], cmap="RdYlGn"), use_container_width=True)
                    else:
                        st.dataframe(day_df, use_container_width=True)

if active_tab == "Ticker Detail":
    st.subheader("🔎 Ticker Detail")

    with st.container(border=True):
        st.markdown("**🔍 Search Any Ticker — Manual Lookup**")
        st.caption("Analyse any stock not in your scan history. Fetches live data and runs the full Argus scoring engine.")
        _mt_col1, _mt_col2 = st.columns([3, 1])
        _manual_ticker_input = _mt_col1.text_input("Enter ticker symbol", placeholder="e.g. PLTR, NVDA, AAPL", label_visibility="collapsed", key="manual_ticker_input").upper().strip()
        _run_manual = _mt_col2.button("🔍 Analyse", key="btn_manual_ticker_analyze", use_container_width=True)
    if _run_manual and _manual_ticker_input:
        with st.spinner(f"Fetching and scoring {_manual_ticker_input}..."):
            try:
                from engine import score_stock, load_memory
                _mem_df = load_memory()
                _manual_result = score_stock(_manual_ticker_input, _mem_df, config, None)
                if _manual_result:
                    _mr_df = pd.DataFrame([_manual_result])
                    _mr_df = add_predictions(_mr_df, model)
                    _mr_df = add_risk_guidance(_mr_df, model, risk_per_trade_pct=risk_per_trade_pct, max_position_pct=max_position_pct)
                    _sig_lbl, _sig_col = get_signal_label(_manual_result.get("score", 0))
                    st.success(f"**{_manual_ticker_input}** — Score: **{_manual_result['score']}/100** | **:{_sig_col}[{_sig_lbl}]**")
                    display_cards(_mr_df)
                    st.dataframe(_mr_df.T, use_container_width=True)
                else:
                    st.warning(f"{_manual_ticker_input} did not pass Argus filters (score too low, invalid ticker, or red flags detected).")
                    try:
                        _info = yf.Ticker(_manual_ticker_input).info
                        if _info and _info.get("symbol"):
                            snap = fetch_financial_snapshot(_manual_ticker_input)
                            if snap:
                                st.markdown("##### Basic Financial Snapshot")
                                st.dataframe(pd.DataFrame(list(snap.items()), columns=["Metric", "Value"]).set_index("Metric"), use_container_width=True)
                    except Exception:
                        pass
            except Exception as _e:
                st.error(f"Error analysing {_manual_ticker_input}: {_e}")
    st.divider()

    if history_df.empty:
        st.info("No history available yet.")
    else:
        ticker_options = sorted(history_df["ticker"].dropna().unique().tolist())
        # Try to use selected_ticker from session_state
        idx = 0
        if "selected_ticker" in st.session_state and st.session_state["selected_ticker"] in ticker_options:
            idx = ticker_options.index(st.session_state["selected_ticker"])
            
        ticker = st.selectbox("Select ticker", ticker_options, index=idx)
        st.session_state["selected_ticker"] = ticker
        
        # Load up the ticker history
        tdf = history_df[history_df["ticker"] == ticker].copy()
        tdf["scan_date"] = pd.to_datetime(tdf["scan_date"], errors="coerce")
        tdf = tdf.sort_values("scan_date", ascending=False)
        
        if not tdf.empty:
            latest_ticker_row = tdf.sort_values("scan_date", ascending=False).head(1).copy()
            latest_ticker_row = add_predictions(latest_ticker_row, model)
            latest_ticker_row = add_risk_guidance(
                latest_ticker_row,
                model,
                risk_per_trade_pct=risk_per_trade_pct,
                max_position_pct=max_position_pct,
            )
            
            # --- TOP SECTION: Score History & Strategy ---
            colA, colB = st.columns([2, 1])
            with colA:
                st.markdown("#### Argus Score History")
                safe_line_chart(tdf.set_index("scan_date")["score"], y_label="score")
                
            with colB:
                st.markdown("#### Buy & Sell Strategy")
                st.info("Actionable execution plan based on the latest signal.")
                score_val = latest_ticker_row["score"].iloc[0]
                upside_prob = latest_ticker_row.get("prob_upside", pd.Series([0])).iloc[0]
                entry = latest_ticker_row.get("entry_style", pd.Series([""])).iloc[0]
                sl_pct = latest_ticker_row.get("stop_loss_pct", pd.Series([0])).iloc[0]
                tp_pct = latest_ticker_row.get("take_profit_pct", pd.Series([0])).iloc[0]
                
                prob_str = f"{upside_prob*100:.1f}%" if upside_prob is not None else "N/A"
                _td_sig_lbl, _td_sig_col = get_signal_label(score_val)
                st.markdown(f"**Signal: :{_td_sig_col}[{_td_sig_lbl}]**")
                st.markdown(f"**Buy:** Score of **{score_val}** & Upside Prob **{prob_str}**. "
                            f"Entry style is **{entry}**.")
                st.markdown(f"**Sell:** Stop loss at **-{sl_pct:.1f}%**. "
                            f"Target taking profits around **+{tp_pct:.1f}%**.")
            
                st.markdown("#### Groq Investment Thesis")
                _thesis_key = f"thesis_{ticker}"
                _thesis_err_key = f"thesis_err_{ticker}"
                _groq_avail = bool(Config().GROQ_API_KEY)

                if _groq_avail and _thesis_key not in st.session_state and _thesis_err_key not in st.session_state:
                    try:
                        from llm import generate_ai_thesis as _gen_thesis
                        _api_key = Config().GROQ_API_KEY
                        _reasons_raw = latest_ticker_row["reasons"].iloc[0]
                        if isinstance(_reasons_raw, str):
                            if _reasons_raw.startswith("["):
                                try:
                                    _reasons_raw = json.loads(_reasons_raw)
                                except (json.JSONDecodeError, ValueError):
                                    _reasons_raw = [_reasons_raw]
                            else:
                                _reasons_raw = [_reasons_raw]
                        elif not isinstance(_reasons_raw, list):
                            _reasons_raw = []
                        with st.spinner(f"Generating AI thesis for {ticker}…"):
                            st.session_state[_thesis_key] = _gen_thesis(ticker, score_val, _reasons_raw, _api_key)
                    except Exception as _te:
                        st.session_state[_thesis_err_key] = str(_te)

                if st.session_state.get(_thesis_key):
                    st.info(st.session_state[_thesis_key])
                    if st.button("🔄 Regenerate", key=f"btn_regen_{ticker}"):
                        for _k in [_thesis_key, _thesis_err_key]:
                            st.session_state.pop(_k, None)
                        st.rerun()
                elif st.session_state.get(_thesis_err_key):
                    st.error(f"Thesis error: {st.session_state[_thesis_err_key]}")
                    if st.button("🔄 Retry", key=f"btn_retry_{ticker}"):
                        st.session_state.pop(_thesis_err_key, None)
                        st.rerun()
                elif not _groq_avail:
                    st.info("Set `GROQ_API_KEY` environment variable to enable auto AI thesis.")

            st.markdown("#### Execution Guidance")
            
            # --- MIDDLE SECTION: Price History & Detailed Snapshot ---
            colC, colD = st.columns([2, 1])
            with colC:
                st.markdown("#### Price & Score History (1 Year)")
                st.caption("Top: 1-year price. Bottom: Argus score over each scan — 🟢 ≥75 strong, 🟡 50–75 watch, 🔴 <50 weak.")
                hist = fetch_ticker_history(ticker, period="1y")

                try:
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots

                    tdf_scores = tdf.dropna(subset=["scan_date", "score"]).sort_values("scan_date")

                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.10,
                        subplot_titles=("Price ($)", "Argus Score (0–100)"),
                        row_heights=[0.62, 0.38],
                    )

                    if not hist.empty:
                        fig.add_trace(
                            go.Scatter(x=hist.index, y=hist["Close"], name="Price",
                                       line=dict(color="#2196F3", width=1.5), fill="tozeroy",
                                       fillcolor="rgba(33,150,243,0.07)"),
                            row=1, col=1,
                        )

                    if not tdf_scores.empty:
                        fig.add_hrect(y0=75, y1=100, fillcolor="rgba(76,175,80,0.12)", line_width=0, row=2, col=1)
                        fig.add_hrect(y0=50, y1=75,  fillcolor="rgba(255,193,7,0.10)",  line_width=0, row=2, col=1)
                        fig.add_hrect(y0=0,  y1=50,  fillcolor="rgba(244,67,54,0.08)",  line_width=0, row=2, col=1)
                        fig.add_hline(y=75, line_dash="dot", line_color="rgba(76,175,80,0.5)",  row=2, col=1)
                        fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,193,7,0.5)",  row=2, col=1)
                        fig.add_trace(
                            go.Scatter(
                                x=tdf_scores["scan_date"], y=tdf_scores["score"],
                                name="Argus Score", mode="lines+markers",
                                line=dict(color="orange", width=2),
                                marker=dict(size=7, color=tdf_scores["score"],
                                            colorscale=[[0,"#f44336"],[0.5,"#ffc107"],[1,"#4caf50"]],
                                            cmin=0, cmax=100, showscale=False),
                                text=[f"Score: {s:.0f}" for s in tdf_scores["score"]],
                                hovertemplate="%{text}<extra></extra>",
                            ),
                            row=2, col=1,
                        )

                    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                    fig.update_yaxes(title_text="Score", range=[0, 100], row=2, col=1)
                    fig.update_layout(
                        height=420, margin=dict(l=0, r=0, t=30, b=0),
                        hovermode="x unified", showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning("Could not render chart.")
                    if not hist.empty:
                        safe_line_chart(hist["Close"], y_label="close price")
                    else:
                        st.info("No market price data available for chart.")
                    
            with colD:                            
                st.markdown("#### Execution Guidance")
                st.dataframe(
                    latest_ticker_row[
                        [
                            "ticker",
                            "score",
                            "prob_upside",
                            "suggested_position_pct",
                            "stop_loss_pct",
                            "take_profit_pct",
                        ]
                    ].T,
                    use_container_width=True,
                )
                
                st.markdown("#### Financial Snapshot")
                snap = fetch_financial_snapshot(ticker)
                if snap:
                    snap_df = pd.DataFrame(list(snap.items()), columns=["Metric", "Value"]).set_index("Metric")
                    st.dataframe(snap_df, use_container_width=True)
                else:
                    st.info("Snapshot data unavailable.")

                # ── Enrichment: earnings date, analyst target, insider ──
                try:
                    _td_enr = fetch_enrichment(ticker)
                    _enr_lines = []
                    if _td_enr.get("earnings_date"):
                        _enr_lines.append(f"📅 **Next Earnings:** {_td_enr['earnings_date']}")
                    if _td_enr.get("analyst_target") and _td_enr.get("analyst_upside") is not None:
                        _u_icon = "🟢" if _td_enr["analyst_upside"] > 0 else "🔴"
                        _enr_lines.append(
                            f"{_u_icon} **Analyst Target:** ${_td_enr['analyst_target']} "
                            f"({_td_enr['analyst_upside']:+.1f}% upside)"
                        )
                    if _td_enr.get("insider_net"):
                        _enr_lines.append(f"👤 **Insider Activity:** {_td_enr['insider_net']}")
                    for _el in _enr_lines:
                        st.markdown(_el)
                except Exception:
                    pass
            
            st.markdown("---")

            # ── Recent News Headlines ─────────────────────────────────────────
            with st.expander("📰 Recent News Headlines"):
                _news_items = fetch_ticker_news(ticker, max_items=5)
                if _news_items:
                    for _ni in _news_items:
                        _pub_str = f" · *{_ni['publisher']}*" if _ni.get("publisher") else ""
                        _ts_str = f" · {_ni['pub_time']}" if _ni.get("pub_time") else ""
                        if _ni.get("link"):
                            st.markdown(f"- [{_ni['title']}]({_ni['link']}){_pub_str}{_ts_str}")
                        else:
                            st.markdown(f"- **{_ni['title']}**{_pub_str}{_ts_str}")
                else:
                    st.caption("No recent news available.")

            # --- BOTTOM SECTION: Data view & AI ---
            with st.expander("View Raw Scan History Table"):
                if "reasons" in tdf.columns:
                    tdf["reasons"] = tdf["reasons"].apply(format_reasons)
                
                try:
                    # Streamlit 1.35+ supports selection events
                    selection_event = st.dataframe(
                        tdf, 
                        use_container_width=True, 
                        on_select="rerun", 
                        selection_mode="single-row"
                    )
                    if selection_event and hasattr(selection_event, "selection") and selection_event.selection.rows:
                        selected_idx = selection_event.selection.rows[0]
                        selected_row = tdf.iloc[[selected_idx]].copy()
                        st.markdown("##### 📝 Details for Selected Scan")
                        display_cards(selected_row)
                except TypeError:
                    # Fallback for older Streamlit versions
                    st.dataframe(tdf, use_container_width=True)

@st.dialog("Add Journal Entry")
def add_journal_entry_dialog(cfg, active_journal="Default"):
    with st.form("journal_form"):
        col1, col2, col3 = st.columns(3)
        journal_ticker = col1.text_input("Ticker").upper().strip()
        journal_action = col2.selectbox("Action", ["BUY", "PASS", "SELL", "SCALE_IN", "TRIM"])
        journal_scan_date = col3.date_input("Bought Date", value=datetime.now())
        
        col4, col5, col6, col7 = st.columns(4)
        entry_price = col4.number_input("Entry price", min_value=0.0, value=0.0)
        shares = col5.number_input("Shares", min_value=0.0, value=0.0, step=1.0)
        stop_loss_pct = col6.number_input("Stop loss %", min_value=0.0, max_value=100.0, value=8.0)
        take_profit_pct = col7.number_input("Take profit %", min_value=0.0, max_value=300.0, value=25.0)
        
        position_size_pct = None
        if st.session_state.get("enable_journal_advanced"):
            st.markdown("**Advanced Tracking**")
            ac1, ac2 = st.columns(2)
            position_size_pct = ac1.number_input("Position size %", min_value=0.0, max_value=100.0, value=2.0)
            st.info("Tracking iterative entries and scale-outs is supported by selecting 'SCALE_IN' or 'TRIM' under actions.")
            
        notes = st.text_area("Notes")
        st.caption(f"Saving to portfolio: **{active_journal}**")
        submitted = st.form_submit_button("Save Journal Entry")
        if submitted:
            if not journal_ticker:
                st.error("Ticker is required.")
            else:
                save_journal_entry(
                    cfg.JOURNAL_FILE,
                    {
                        "ticker": journal_ticker,
                        "action": journal_action,
                        "scan_date": str(journal_scan_date),
                        "entry_price": entry_price,
                        "shares": shares,
                        "position_size_pct": position_size_pct if position_size_pct is not None else 0.0,
                        "stop_loss_pct": stop_loss_pct,
                        "take_profit_pct": take_profit_pct,
                        "notes": notes,
                        "journal_name": active_journal,
                    },
                )
                st.success("Journal entry saved.")
                st.rerun()

if active_tab == "Journal":
    st.subheader("📝 Decision Journal")

    # ── Portfolio (Journal) Management ──────────────────────────────────────
    all_journals = list_journals()
    journal_options = ["All"] + all_journals

    jm_col1, jm_col2, jm_col3, jm_col4 = st.columns([3, 2, 2, 2], vertical_alignment="bottom")
    with jm_col1:
        active_journal = st.selectbox(
            "Active Portfolio",
            journal_options,
            key="active_journal_selector",
            help="Select which portfolio to view. 'All' shows every entry across all portfolios.",
        )
    with jm_col2:
        new_journal_name = st.text_input("New portfolio name", placeholder="e.g. Robinhood", label_visibility="collapsed", key="new_journal_name_input")
    with jm_col3:
        if st.button("➕ Create Portfolio", use_container_width=True):
            _nj = new_journal_name.strip()
            if _nj and _nj not in all_journals:
                save_journal_entry(config.JOURNAL_FILE, {
                    "ticker": "_INIT_", "action": "PASS", "scan_date": datetime.now().strftime("%Y-%m-%d"),
                    "entry_price": 0, "shares": 0, "stop_loss_pct": 0, "take_profit_pct": 0,
                    "position_size_pct": 0, "notes": "Portfolio initialised", "journal_name": _nj,
                }, journal_name=_nj)
                st.success(f"Portfolio '{_nj}' created.")
                st.rerun()
            elif not _nj:
                st.warning("Enter a name first.")
            else:
                st.warning(f"'{_nj}' already exists.")
    with jm_col4:
        if active_journal not in ("All", "Default") and st.button("🗑️ Delete Portfolio", use_container_width=True, help="Permanently deletes all entries in this portfolio."):
            if delete_journal(active_journal, journal_file=config.JOURNAL_FILE):
                st.success(f"Portfolio '{active_journal}' deleted.")
                st.rerun()
            else:
                st.error("Could not delete portfolio.")

    st.markdown("---")
    st.checkbox("Enable Advanced Tracking (Position sizing & Scan History)", key="enable_journal_advanced")

    _current_journal = active_journal if active_journal != "All" else None

    _add_col, _csv_col = st.columns([1, 1])
    with _add_col:
        if st.button("➕ Add New Entry"):
            add_journal_entry_dialog(config, active_journal=active_journal if active_journal != "All" else "Default")
    with _csv_col:
        with st.expander("📥 Import from CSV"):
            st.caption(
                "**Only `ticker` and `action` are required.** All other columns are optional — "
                "missing values will be filled with blanks/zeros so you can edit them later.  \n"
                "**Full column list (case-insensitive):** "
                "`ticker` · `action` *(BUY/SELL/PASS/SCALE_IN/TRIM)* · `scan_date` *(YYYY-MM-DD)* · "
                "`entry_price` · `shares` · `stop_loss_pct` · `take_profit_pct` · `position_size_pct` · `notes`  \n"
                "File will be imported into the currently selected portfolio."
            )
            _uploaded_csv = st.file_uploader("Upload journal CSV", type=["csv"], label_visibility="collapsed", key="journal_csv_uploader")
            if _uploaded_csv is not None:
                try:
                    _import_df = pd.read_csv(_uploaded_csv)
                    _required_cols = {"ticker", "action"}
                    def _norm_col(c):
                        c = c.lower().replace(" ", "_").replace("%", "pct")
                        c = re.sub(r'[^a-z0-9_]', '', c)
                        return re.sub(r'_+', '_', c).strip('_')
                    _import_df.columns = [_norm_col(c) for c in _import_df.columns]
                    _missing = _required_cols - set(_import_df.columns)
                    if _missing:
                        st.error(f"Missing required columns: {', '.join(_missing)}. Only 'ticker' and 'action' are mandatory.")
                    else:
                        _optional_defaults = {
                            "scan_date": datetime.now().strftime("%Y-%m-%d"),
                            "entry_price": 0.0,
                            "shares": 0.0,
                            "stop_loss_pct": None,
                            "take_profit_pct": None,
                            "position_size_pct": None,
                            "notes": "",
                        }
                        for _col, _default in _optional_defaults.items():
                            if _col not in _import_df.columns:
                                _import_df[_col] = _default
                        _import_df["ticker"] = _import_df["ticker"].astype(str).str.upper().str.strip()
                        st.dataframe(_import_df.head(5), use_container_width=True)
                        _target_j = active_journal if active_journal != "All" else "Default"
                        if st.button(f"✅ Confirm Import ({len(_import_df)} rows) → {_target_j}", key="btn_confirm_csv_import"):
                            _import_df["journal_name"] = _target_j
                            _import_df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            conn = get_db_connection()
                            _import_df.to_sql("journal", conn, if_exists="append", index=False)
                            sync_journal_to_csv(config.JOURNAL_FILE)
                            conn.close()
                            st.success(f"Imported {len(_import_df)} entries into '{_target_j}'.")
                            st.rerun()
                except Exception as _csv_err:
                    st.error(f"Failed to parse CSV: {_csv_err}")

    journal_df = load_journal(config.JOURNAL_FILE, journal_name=_current_journal)
    if "journal_name" not in journal_df.columns:
        journal_df["journal_name"] = "Default"
    journal_df = journal_df[journal_df["ticker"] != "_INIT_"]

    if journal_df.empty:
        st.info("No journal entries yet.")
    else:
        _j_tickers = sorted(journal_df["ticker"].unique().tolist())
        with st.expander("📋 Copy Tickers"):
            st.caption(f"{len(_j_tickers)} tickers — click the copy icon on the right →")
            st.code(", ".join(_j_tickers), language=None)

        _tx_tab, _hold_tab = st.tabs(["📋 All Transactions", "🏦 Holdings Summary"])

        with _tx_tab:
            edited_df = st.data_editor(
                journal_df.sort_values("timestamp", ascending=False),
                num_rows="dynamic",
                use_container_width=True,
                key="journal_editor"
            )
            if not edited_df.equals(journal_df.sort_values("timestamp", ascending=False)):
                if st.button("💾 Save Edited Log"):
                    conn = get_db_connection()
                    if _current_journal:
                        try:
                            from sqlalchemy import text as _sa_text
                            conn.execute(_sa_text("DELETE FROM journal WHERE journal_name = :jn"), {"jn": _current_journal})
                        except Exception:
                            conn.execute("DELETE FROM journal WHERE journal_name = ?", (_current_journal,))
                        conn.commit()
                        edited_df.to_sql("journal", conn, if_exists="append", index=False)
                    else:
                        edited_df.to_sql("journal", conn, if_exists="replace", index=False)
                    sync_journal_to_csv(config.JOURNAL_FILE)
                    conn.close()
                    st.success("Log updated successfully!")
                    st.rerun()

        with _hold_tab:
            _hold_map = {}
            for _, _hr in journal_df.iterrows():
                _ht = str(_hr.get("ticker", "")).strip().upper()
                _ha = str(_hr.get("action", "")).strip().upper()
                _hs = float(_hr.get("shares") or 0)
                _hp = float(_hr.get("entry_price") or 0)
                if not _ht:
                    continue
                if _ht not in _hold_map:
                    _hold_map[_ht] = {"shares": 0.0, "cost": 0.0, "txns": 0}
                _hold_map[_ht]["txns"] += 1
                if _ha in ("BUY", "SCALE_IN"):
                    _hold_map[_ht]["shares"] += _hs
                    _hold_map[_ht]["cost"] += _hs * _hp
                elif _ha in ("SELL", "TRIM") and _hold_map[_ht]["shares"] > 0:
                    _rm = min(_hs, _hold_map[_ht]["shares"])
                    _ratio = _rm / _hold_map[_ht]["shares"]
                    _hold_map[_ht]["cost"] -= _hold_map[_ht]["cost"] * _ratio
                    _hold_map[_ht]["shares"] -= _rm

            _hold_rows = []
            for _ht, _hd in _hold_map.items():
                _avg_p = _hd["cost"] / _hd["shares"] if _hd["shares"] > 0.001 else 0.0
                _hold_rows.append({
                    "Ticker": _ht,
                    "Transactions": _hd["txns"],
                    "Shares Held": round(_hd["shares"], 4),
                    "Avg Entry (£)": round(_avg_p, 2),
                    "Total Cost (£)": round(_hd["shares"] * _avg_p, 2),
                })

            if _hold_rows:
                _hold_df = pd.DataFrame(_hold_rows)
                _open_hold = _hold_df[_hold_df["Shares Held"] > 0.001].copy()
                if not _open_hold.empty:
                    with st.spinner("Fetching live prices..."):
                        _live_p = fetch_current_prices(_open_hold["Ticker"].tolist())
                    _open_hold["Current (£)"] = _open_hold["Ticker"].map(_live_p).round(2)
                    _open_hold["Mkt Value (£)"] = (_open_hold["Shares Held"] * _open_hold["Current (£)"]).round(2)
                    _open_hold["P&L (£)"] = (_open_hold["Mkt Value (£)"] - _open_hold["Total Cost (£)"]).round(2)
                    _open_hold["P&L (%)"] = (
                        (_open_hold["Current (£)"] - _open_hold["Avg Entry (£)"]) /
                        _open_hold["Avg Entry (£)"].replace(0, float("nan")) * 100
                    ).round(2)
                    _tc = _open_hold["Total Cost (£)"].sum()
                    _tv = _open_hold["Mkt Value (£)"].sum()
                    _tp = _open_hold["P&L (£)"].sum()
                    _tpp = (_tp / _tc * 100) if _tc > 0 else 0
                    hc1, hc2, hc3, hc4 = st.columns(4)
                    hc1.metric("Open Positions", len(_open_hold))
                    hc2.metric("Total Cost", f"£{_tc:,.2f}")
                    hc3.metric("Market Value", f"£{_tv:,.2f}")
                    hc4.metric("Total P&L", f"£{_tp:+,.2f}", delta=f"{_tpp:+.1f}%")
                    _pnl_abs = _open_hold["P&L (%)"].abs().max() or 1
                    st.dataframe(
                        _open_hold.style.background_gradient(subset=["P&L (%)"], cmap="RdYlGn", vmin=-_pnl_abs, vmax=_pnl_abs),
                        use_container_width=True, hide_index=True
                    )
                _closed = _hold_df[_hold_df["Shares Held"] <= 0.001][["Ticker", "Transactions"]].copy()
                if not _closed.empty:
                    with st.expander(f"Closed / Exited Positions ({len(_closed)})"):
                        st.dataframe(_closed, use_container_width=True, hide_index=True)
            else:
                st.info("No transactions to summarise yet.")

        st.markdown("### 📊 Portfolio P&L Status")
        
        trades = []
        open_positions = []
        j_sorted = journal_df.sort_values("timestamp")
        
        for t in j_sorted["ticker"].unique():
            t_df = j_sorted[j_sorted["ticker"] == t]
            buys = t_df[t_df["action"].isin(["BUY", "SCALE_IN"])].copy()
            sells = t_df[t_df["action"].isin(["SELL", "TRIM"])].copy()

            buy_shares = buys["shares"].fillna(0)
            sell_shares = sells["shares"].fillna(0)

            total_buy_shares = buy_shares.sum()
            total_sell_shares = sell_shares.sum()

            buy_count = len(buys)
            sell_count = len(sells)

            if buy_count > 0:
                if total_buy_shares > 0:
                    avg_buy = float((buys["entry_price"] * buy_shares).sum() / total_buy_shares)
                else:
                    avg_buy = float(buys["entry_price"].mean())

                net_shares = max(0.0, total_buy_shares - total_sell_shares)
                if net_shares > 0 or (buy_count > sell_count and total_buy_shares == 0):
                    first_buy_date = buys["timestamp"].min()
                    first_buy_date_str = str(first_buy_date)[:10] if pd.notnull(first_buy_date) else datetime.now().strftime("%Y-%m-%d")
                    open_positions.append({"Ticker": t, "Avg Buy": avg_buy, "Shares": net_shares, "First Date": first_buy_date_str})

                if sell_count > 0:
                    if total_sell_shares > 0:
                        avg_sell = float((sells["entry_price"] * sell_shares).sum() / total_sell_shares)
                    else:
                        avg_sell = float(sells["entry_price"].mean())
                    ret_pct = ((avg_sell - avg_buy) / avg_buy) * 100 if avg_buy > 0 else 0.0
                    trades.append({"Ticker": t, "Avg Buy": avg_buy, "Avg Sell": avg_sell, "Return (%)": ret_pct})

        realized_tab, unrealized_tab = st.tabs(["Realized P&L", "Open Positions (Live)"])
        
        with realized_tab:
            if trades:
                pnl_df = pd.DataFrame(trades).round(2)
                wins = len(pnl_df[pnl_df["Return (%)"] > 0])
                total_closed = len(pnl_df)
                win_rate = (wins / total_closed) * 100
                
                mc1, mc2 = st.columns(2)
                mc1.metric("Closed Trades", total_closed, help="Trades with both BUY and SELL logs.")
                mc2.metric("Win Rate", f"{win_rate:.1f}%")
                _ret_abs = pnl_df["Return (%)"].abs().max() or 1
                st.dataframe(pnl_df.style.background_gradient(subset=["Return (%)"], cmap="RdYlGn", vmin=-_ret_abs, vmax=_ret_abs), use_container_width=True)
            else:
                st.info("No closed trades found.")
                
        with unrealized_tab:
            if open_positions:
                open_df = pd.DataFrame(open_positions).round(2)
                with st.spinner("Fetching live prices for open positions..."):
                    current_prices = fetch_current_prices(open_df["Ticker"].tolist())
                
                open_df["Current Price (£)"] = open_df["Ticker"].map(current_prices).round(2)
                open_df["Cost Basis (£)"] = (open_df["Shares"] * open_df["Avg Buy"]).round(2)
                open_df["Current Value (£)"] = (open_df["Shares"] * open_df["Current Price (£)"]).round(2)
                open_df["Unrealized (%)"] = (((open_df["Current Price (£)"] - open_df["Avg Buy"]) / open_df["Avg Buy"]) * 100).round(2)
                
                total_invested = open_df["Cost Basis (£)"].sum()
                total_current = open_df["Current Value (£)"].sum()
                net_return_pct = ((total_current - total_invested) / total_invested * 100) if total_invested > 0 else 0
                
                first_entry_date = open_df["First Date"].min()
                spy_return = get_spy_return(first_entry_date)
                
                st.markdown("### 📊 Portfolio Analytics")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Invested", f"£{total_invested:,.2f}")
                m2.metric("Current Value", f"£{total_current:,.2f}")
                m3.metric("Net Return", f"{net_return_pct:+.2f}%")
                m4.metric("SPY Benchmark", f"{spy_return:+.2f}%", help=f"S&P 500 Return since earliest active entry ({first_entry_date})")
                
                st.divider()
                
                if total_invested > 0:
                    with st.spinner("Mapping sectors..."):
                        open_df["Sector"] = open_df["Ticker"].apply(get_cached_sector)
                    
                    sector_totals = open_df.groupby("Sector")["Current Value (£)"].sum().reset_index()
                    fig = px.pie(sector_totals, values='Current Value (£)', names='Sector', title="Sector Exposure by Value (£)", hole=0.3)
                    
                    ui_col1, ui_col2 = st.columns([1, 1])
                    with ui_col1:
                        st.plotly_chart(fig, use_container_width=True)
                    with ui_col2:
                        st.markdown("<br><br>", unsafe_allow_html=True)
                        st.markdown("#### Open Positions")
                        display_df = open_df.drop(columns=["First Date", "Sector"])
                        _unr_abs = display_df["Unrealized (%)"].abs().max() or 1
                        st.dataframe(display_df.style.background_gradient(subset=["Unrealized (%)"], cmap="RdYlGn", vmin=-_unr_abs, vmax=_unr_abs), use_container_width=True, hide_index=True)
                else:
                    display_df = open_df.drop(columns=["First Date"])
                    _unr_abs = display_df["Unrealized (%)"].abs().max() or 1
                    st.dataframe(display_df.style.background_gradient(subset=["Unrealized (%)"], cmap="RdYlGn", vmin=-_unr_abs, vmax=_unr_abs), use_container_width=True, hide_index=True)
            else:
                st.info("No open positions found.")
                
        st.markdown("---")
        st.markdown("### 🗑️ Remove Ticker Completely")
        st.write("Permanently delete ALL journal entries and monitoring for a specific ticker.")
        
        del_col1, del_col2 = st.columns([3, 1], vertical_alignment="bottom")
        with del_col1:
            ticker_to_delete = st.selectbox("Select Ticker to Delete", [""] + sorted(journal_df["ticker"].unique().tolist()), key="del_ticker")
        with del_col2:
            if st.button("Delete All Logs", help="This action cannot be undone.", use_container_width=True):
                if ticker_to_delete:
                    conn = get_db_connection()
                    try:
                        from sqlalchemy import text
                        conn.execute(text("DELETE FROM journal WHERE ticker = :ticker"), {"ticker": ticker_to_delete})
                    except:
                        conn.execute("DELETE FROM journal WHERE ticker = ?", (ticker_to_delete,))
                    conn.commit()
                    sync_journal_to_csv(config.JOURNAL_FILE)
                    conn.close()
                    st.success(f"Successfully deleted {ticker_to_delete} from the Journal!")
                    st.rerun()
                else:
                    st.error("Please select a ticker.")

if active_tab == "Manual Run":
    st.subheader("⚙️ Manual Scan")
    st.caption("Use this for weekend/on-demand runs. Results are written to history + feature store.")
    if st.button("🚀 Run Global Scan", key="btn_run_scan"):
        
        st.info(f"Scan executing using preset: **{st.session_state.preset_selector}**")
        
        _lottie_ph = st.empty()
        if HAS_LOTTIE:
            _anim = _load_lottie("https://assets5.lottiefiles.com/packages/lf20_qp1q7mct.json")
            if _anim:
                with _lottie_ph:
                    st_lottie(_anim, height=130, key="scan_lottie", speed=1, loop=True)

        progress_bar = st.progress(0)
        status_text = st.empty()

        def scan_progress_cb(ticker, idx, total):
            progress = (idx + 1) / total
            progress_bar.progress(progress)
            status_text.text(f"Scanning {ticker}... ({idx + 1}/{total})")

        with st.spinner("Scanning tickers..."):
            payload = run_scan(config=config, scan_limit=scan_limit, update_memory=True, progress_callback=scan_progress_cb)
            status_text.text("Scan complete!")
            results = payload["results"]
            scan_date = payload["scan_date"]
            scan_timestamp = payload["scan_timestamp"]
            scanned_count = payload["scanned_count"]

            save_results(
                results=results,
                scan_date=scan_date,
                scan_timestamp=scan_timestamp,
                run_type="manual",
                latest_file=config.RESULTS_FILE,
                history_file=config.RESULTS_HISTORY_FILE,
                write_latest=False,
                feature_file=config.FEATURES_FILE,
            )

        _lottie_ph.empty()
        st.caption(f"Scanned {scanned_count} tickers")
        if results:
            regime = cached_market_regime()
            st.success(f"{len(results)} picks met requirements in {regime['regime']} regime.")
            df_res = pd.DataFrame(results).sort_values("score", ascending=False)
            df_res = add_predictions(df_res, model)
            df_res = add_risk_guidance(
                df_res,
                model,
                risk_per_trade_pct=risk_per_trade_pct,
                max_position_pct=max_position_pct,
            )
            df_res = format_pct_columns(
                df_res,
                ["prob_upside", "scenario_bear", "scenario_base", "scenario_bull"],
            )
            
            if "reasons" in df_res.columns:
                df_res["reasons"] = df_res["reasons"].apply(format_reasons)
                
            display_cards(df_res)
            
            with st.expander("Show Raw Discovery Table"):
                st.dataframe(df_res, use_container_width=True)

            st.subheader("High Conviction Charts (Score >= 80)")
            high_conviction = df_res[df_res["tier"] == "🟢 HIGH CONVICTION"]
            if not high_conviction.empty:
                cols = st.columns(3)
                for idx, row in high_conviction.reset_index(drop=True).iterrows():
                    with cols[idx % 3]:
                        st.write(f"**{row['ticker']}** - Score: {row['score']}")
                        hist = fetch_ticker_history(row["ticker"], period="6mo")
                        if not hist.empty:
                            safe_line_chart(hist["Close"], y_label="close")
                        else:
                            st.write("No history available")
            else:
                st.info("No high conviction picks found in this run.")
        else:
            st.warning("No tickers met the current criteria.")

        if send_to_telegram:
            alerts = monitor_portfolio()
            message = generate_telegram_message(
                results, 
                scanned_count, 
                title="Argus Manual Scan", 
                date_str=datetime.now().strftime('%d %b %Y %H:%M'),
                alerts=alerts
            )
            delivered = send_telegram_message(config.TELEGRAM_TOKEN, config.TELEGRAM_CHAT_ID, message)
            if delivered:
                st.success("Manual run sent to Telegram.")
            else:
                st.error("Could not send to Telegram. Check TELEGRAM_TOKEN and TELEGRAM_CHAT_ID.")

if active_tab == "Prediction Model":
    st.subheader("📈 ML Prediction Model Quality")
    if not model.get("ready"):
        st.info(model.get("reason", "Model is not ready yet."))
        st.caption("Keep running scans to accumulate matured samples (need ~50+).")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Matured Samples", model["samples"])
        col2.metric("Global Hit Rate", f"{model['global_prob'] * 100:.1f}%")
        col3.metric("Brier Score", f"{model['brier_score']:.3f}")
        st.caption(f"Target: +{target_return * 100:.0f}% in {horizon_days} days")

        if model.get("clf") is not None:
            st.success("🤖 XGBoost Model Active")
            
            cmat, cshap = st.columns([1, 1])
            with cmat:
                if "confusion_matrix" in model and model["confusion_matrix"]:
                    st.markdown("**Confusion Matrix (Default Threshold 0.5)**")
                    import plotly.figure_factory as ff
                    cm_data = model["confusion_matrix"]
                    fig = ff.create_annotated_heatmap(
                        z=cm_data,
                        x=['Predicted Negative', 'Predicted Positive'],
                        y=['Actual Negative', 'Actual Positive'],
                        colorscale='Blues'
                    )
                    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig, use_container_width=True)
            
            with cshap:
                if "feature_importance" in model and model["feature_importance"]:
                    st.markdown("**Feature Importance (XGB)**")
                    imp_dt = pd.DataFrame(list(model["feature_importance"].items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=False)
                    st.dataframe(imp_dt.head(10).style.background_gradient(subset=["Importance"], cmap="Oranges"), use_container_width=True)
            
            st.markdown("---")
            st.markdown("**SHAP Global Feature Impact**")
            try:
                import shap
                import matplotlib.pyplot as plt
                X_sample = model["X_sample"]
                explainer = shap.TreeExplainer(model["clf"])
                shap_values = explainer.shap_values(X_sample)
                fig_shap, ax_shap = plt.subplots(figsize=(8, 5))
                shap.summary_plot(shap_values, X_sample, plot_type="dot", show=False, max_display=12)
                st.pyplot(fig_shap, bbox_inches='tight')
            except Exception as e:
                st.warning(f"Could not generate SHAP plot: {e}")
        else:
            st.warning("📊 Falling back to empirical baseline model (add more granular history to activate ML).")

        bucket_stats = model["bucket_stats"].copy()
        for c in ["prob", "bear", "base", "bull"]:
            bucket_stats[c] = pd.to_numeric(bucket_stats[c], errors="coerce")
            bucket_stats[c] = (bucket_stats[c] * 100).round(1)
        st.markdown("**Score bucket outcomes**")
        st.dataframe(bucket_stats, use_container_width=True)

        calibration = model["calibration"].copy()
        calibration["actual_hit_rate"] = pd.to_numeric(calibration["actual_hit_rate"], errors="coerce")
        calibration["actual_hit_rate"] = (calibration["actual_hit_rate"] * 100).round(1)
        st.markdown("**Calibration table**")
        st.dataframe(calibration, use_container_width=True)

if active_tab == "Portfolio Optimizer":
    st.subheader("⚖️ Portfolio Optimizer & Dynamic Sizing")
    st.markdown("Phase 4: Uses efficient frontier logic to dynamically calculate the best weightings.")
    
    st.write("Select tickers to include in the optimization:")
    
    if not latest_df.empty:
        default_tickers = latest_df.head(5)['ticker'].tolist()
    else:
        default_tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]
        
    opt_tickers = st.multiselect("Optimization Tickers", latest_df["ticker"].tolist() if not latest_df.empty else default_tickers, default=default_tickers)
    
    if st.button("⚡ Optimize Portfolio", key="btn_opt"):
        if len(opt_tickers) < 2:
            st.error("Please select at least 2 tickers.")
        else:
            with st.spinner("Downloading 1-year history and optimizing..."):
                from engine import optimize_portfolio
                opt_result = optimize_portfolio(opt_tickers)
                
                if "error" in opt_result:
                    st.error(opt_result["error"])
                else:
                    ms = opt_result["max_sharpe"]
                    mv = opt_result["min_volatility"]
                    
                    st.success("Optimization Complete!")
                    
                    st.markdown("### Max Sharpe Ratio Portfolio")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        with st.container(border=True): st.metric("Annual Return", f"{ms['return']*100:.1f}%", help="Expected Yield")
                    with col2:
                        with st.container(border=True): st.metric("Volatility", f"{ms['volatility']*100:.1f}%", help="Annual Risk")
                    with col3:
                        with st.container(border=True): st.metric("Sharpe Ratio", f"{ms['sharpe']:.2f}", help="Risk Adjusted Return")
                    
                    st.markdown("#### Optimal Position Sizing (Max Sharpe)")
                    weights_df = pd.DataFrame(list(ms['weights'].items()), columns=['Ticker', 'Weight'])
                    weights_df['Weight'] = weights_df['Weight'].apply(lambda x: f"{x*100:.1f}%")
                    st.dataframe(weights_df, use_container_width=True)
                    
                    st.divider()
                    
                    st.markdown("### Minimum Volatility Portfolio")
                    col4, col5, col6 = st.columns(3)
                    with col4:
                        with st.container(border=True): st.metric("Annual Return", f"{mv['return']*100:.1f}%", help="Expected Yield")
                    with col5:
                        with st.container(border=True): st.metric("Volatility", f"{mv['volatility']*100:.1f}%", help="Annual Risk")
                    with col6:
                        with st.container(border=True): st.metric("Sharpe Ratio", f"{mv['sharpe']:.2f}", help="Risk Adjusted Return")
                    
                    st.markdown("#### Optimal Position Sizing (Min Volatility)")
                    weights_df2 = pd.DataFrame(list(mv['weights'].items()), columns=['Ticker', 'Weight'])
                    weights_df2['Weight'] = weights_df2['Weight'].apply(lambda x: f"{x*100:.1f}%")
                    st.dataframe(weights_df2, use_container_width=True)

if active_tab == "Alerts Log":
    st.subheader("🔔 Alerts Log")
    st.caption("A historical record of all Telegram push notifications generated by Argus.")
    try:
        import re
        with open("argus_alerts_log.txt", "r", encoding="utf-8") as f:
            logs = f.read()
        if logs.strip():
            entries = re.findall(
                r'--- (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ---\n(.*?)(?=--- \d{4}-\d{2}-\d{2}|\Z)',
                logs, re.DOTALL
            )
            if entries:
                for date_str, msg in reversed(entries):
                    msg = msg.strip()
                    if msg:
                        st.markdown(f"**{date_str}**")
                        st.info(msg)
            else:
                st.info("No alerts logged yet.")
        else:
            st.info("No alerts logged yet.")
    except FileNotFoundError:
        st.info("No alerts logged yet.")

if active_tab == "Help":
    st.subheader("❓ Help & Documentation")
    st.markdown('''
    ### Phase & Capabilities Overview
    * **Phase 1: ML Prediction Model:** Uses XGBoost to evaluate hit probabilities based on the accumulation of historical features.
    * **Phase 2: Live AI News Sentiment Scoring:** Connects to the Groq API (Llama 3.1) to dynamically read live news headlines and adjust the quantitative score (up to +/- 15 points) based on real-time sentiment.
    * **Phase 3: Macro Market Regime Filter:** Dynamically adjusts stock scores based on SPY moving averages and the VIX (e.g., boosting scores in Bull markets, penalizing in Bear markets).
    * **Phase 4: Portfolio Optimizer & Auto-Pilot Monitor:** 
        * Uses 1-year historical correlations to compute optimal portfolio weightings.
        * Features an **Active Portfolio Monitor** that tracks your logged `Journal` entries against live market prices, automatically triggering alerts when Stop-Loss (SL) or Take-Profit (TP) levels are breached.
    
    ---
    ### Pages Overview
    * **Overview:** Snapshot of the latest scan results with interactive visual metric cards. View the current macro Market Regime, last scan date, VIX level, and SPY vs 200-day MA proximity. Bear transition warnings appear automatically when conditions deteriorate.
    * **Ticker Detail:** Deep dive into specific tickers. Evaluate a two-panel Plotly chart (price on top, Argus score with colour bands on bottom — 🟢 ≥75, 🟡 50–75, 🔴 <50), quantitative execution guidance, and generate qualitative AI investment thesis reports (Groq).
    * **Portfolio Optimizer:** Select multiple tickers from the recent scan to calculate the statistically optimal portfolio sizing using the efficient frontier.
    * **Manual Run:** Instantly run a new global scan based on your sidebar settings. Includes Telegram notification support and Auto-Pilot portfolio monitoring.
    * **History:** Browse historical database scans with interactive sortable tables. Select any scan day to view a visual card grid of its scoring reasons.
    * **Journal:** Personal logbook featuring a **Live Portfolio Analytics Dashboard**. Track Total Invested, Current Value, Net Return %, and visualize Sector Diversification via interactive pie charts. Benchmarks portfolio performance against the S&P 500 (SPY). Powered by the Auto-Pilot monitor for live SL/TP tracking.
    * **Prediction Model:** ML diagnostics. Review XGBoost hit rates, Brier scores, and calibration accuracy across different market conditions.
    * **Alerts Log:** Record of all automated Telegram push notifications generated by both local manual runs and the GitHub Actions daily scan.
    * **Prompts:** Curated LLM prompts for deeper external research, plus market condition templates.

    ---
    ### 🤖 ML Prediction Model — When Does It Activate?

    The XGBoost model trains on rows in the `features` table where the **future return is already known** (i.e., the scan happened at least `Horizon Days` ago — default 63 days). Each scan of N tickers produces up to N feature rows. Once those rows mature past the horizon, they become labeled training samples.

    **Practical timeline:**
    * 1 scan of 200 tickers = up to 200 samples (but only usable after 63 days)
    * After ~1–2 months of daily GitHub Action scans, you will likely have 1,000–4,000 labeled samples — well above the 30-sample minimum
    * The model activates automatically as soon as 30+ matured samples exist; it improves steadily thereafter

    The model is **100% free** — it uses `yfinance` price data (free) and trains locally with XGBoost. No paid data feed is required.

    ---
    ### 📊 Upside Prob (%), Scenario Bear / Base / Bull — How Are They Calculated?

    All three are derived from the **local XGBoost model** and historical scan data — no paid API is needed:

    * **Upside Prob (%):** The ML model's predicted probability that this ticker will hit the Target Return within the Horizon Days. Calibrated using historical hit rates across score buckets.
    * **Scenario Bear:** The historical hit-rate for tickers in the same score bucket when market conditions were bearish (low-multiplier regimes).
    * **Scenario Base:** The overall historical hit-rate for this score bucket across all regimes.
    * **Scenario Bull:** The hit-rate observed in bullish regime periods (high-multiplier).

    These figures populate automatically once the ML model is active. Until then, they display as N/A.

    ---
    ### 🔔 Alerts Log — How It Links to Telegram

    The **Alerts Log** tab reads from `argus_alerts_log.txt`, which is updated in two ways:

    1. **GitHub Actions daily scan:** `argus.py` writes the full message to `argus_alerts_log.txt` and then commits it back to the repository. When you pull the latest code, the log file is updated with every automated run.
    2. **Local manual runs:** Running the scan via the Manual Run tab also appends to the same file on your local machine.

    > **Note:** The Alerts Log shows what was *sent* to Telegram. If a run failed to deliver to Telegram (bad token, rate limit, etc.), the log entry is still written so you can diagnose the issue.

    ---
    ### 📒 Journal Persistence — Your Entries Are Safe

    Journal entries are stored in `argus.db` (SQLite) **locally on your machine only**. The database is now excluded from Git tracking, so:

    * GitHub Actions daily runs will **never overwrite** your local database
    * Pulling new commits will **never wipe** your journal entries
    * Scan history is preserved via the committed CSV files (`argus_results_history.csv`) and automatically re-imported into your local database if it is ever missing

    ---
    ### 🎛 Global Presets — Quick Reference

    | Preset | Best for |
    |---|---|
    | **Default** | Neutral market, unsure which preset |
    | **High Conviction** | Bull market, concentrate into 1–3 best picks |
    | **Momentum Sprint** | Strong trend, catch short-term breakouts |
    | **Earnings Season** | Jan / Apr / Jul / Oct earnings cycles |
    | **Swing Recovery** | Market pulled back 10–20%, expect mean-reversion |
    | **Aggressive Growth** | Confirmed bull run, higher risk tolerance |
    | **Liquidity Focus** | Large accounts needing fast entry/exit |
    | **Capital Preservation** | Bear or Extreme Fear regime active |
    | **Bear Market Defense** | SPY below 200-day MA |
    | **Small Cap Hunter** | Bull market, high risk appetite |
    | **Penny Stock High Risk** | Speculative allocation only |

    ---
    ### Sidebar Settings Explained

    **Quick Scan Settings**
    * **Minimum Score:** The minimum Argus score required to pass (Default: 65).
    * **Universe Size:** How many top market cap tickers to scan (Default: 200).
    
    **Advanced Filters & Constraints**
    * **Price Floor ($):** Lowest acceptable stock price, filtering out highly volatile penny stocks.
    * **Volume Floor:** Minimum average daily trading volume to ensure liquidity (Default: 500,000).
    
    **ML Prediction Horizons**
    * **Horizon Days:** The timeframe over which the ML model predicts price movement (Default: 63 days, ~1 quarter).
    * **Target Return (%):** The target profit percentage the ML model evaluates the stock against (Default: 10%).
    
    **Advanced Risk Rules**
    * **Risk per Trade:** The % of your total portfolio you're willing to lose if stopped out (Default: 0.75%).
    * **Max Position Size:** Maximum portfolio allocation allowed in a single stock, forcing diversification (Default: 8.0%).
    ''')

if active_tab == "Prompts":
    st.subheader("🤖 AI Prompts for Research")
    st.caption("Copy these templates and paste them into AI tools (Perplexity, ChatGPT, Claude, Grok) for deeper analysis. Replace bracketed placeholders with your data.")

    st.markdown("### 💬 AI Chat Assistants")
    st.caption("Quick links to AI chat platforms for researching your stocks.")
    st.link_button("🔍 Ask Perplexity", "https://www.perplexity.ai/spaces/argus-AFO1vOM6R8mv1YzjC8vWzw", help="Opens your personal Argus Perplexity Space for in-depth stock research with live web search.")

    st.markdown("---")
    st.markdown("### 🌍 Market Condition Prompts")
    st.caption("Use these to get a fast read on the current macro environment and whether you should be cautious or aggressive.")

    st.markdown("#### M1. Today's Market Overview")
    st.code("""Today is [TODAY'S DATE]. Give me a concise macro market overview for a U.S. equity investor.
Cover:
1. S&P 500 and Nasdaq current trend vs their 50-day and 200-day moving averages.
2. VIX level — is fear elevated or complacent?
3. The most important macroeconomic headline from the past 48 hours (Fed, inflation, jobs, geopolitics).
4. Overall risk-on or risk-off sentiment verdict.
5. One sector you would overweight and one you would underweight right now, with brief reasoning.
Keep it professional, factual, and under 250 words.""", language="text")

    st.markdown("#### M2. Should I Be Worried or Excited Right Now?")
    st.code("""Today is [TODAY'S DATE]. Act as a senior portfolio manager reviewing current market conditions.
Tell me:
- The top 2 macro risks that could cause a meaningful correction in U.S. equities in the next 30–60 days.
- The top 2 bullish tailwinds that could drive a continued rally.
- Net verdict: should a medium-risk equity investor be adding, holding, or reducing exposure right now?
Be direct and concrete — avoid vague disclaimers. Give me a clear signal.""", language="text")

    st.markdown("#### M3. Argus Watchlist Impact Check")
    st.code("""The following stocks are on my active watchlist: [TICKER1, TICKER2, TICKER3, ...].
Today is [TODAY'S DATE]. For each ticker, briefly answer:
1. Is there any major news or catalyst in the last 5 days that directly affects this company?
2. Does any broad macro event (rate decisions, earnings cycle, sector rotation) meaningfully change the risk/reward for this stock?
3. Flag any ticker I should be especially cautious about right now, and why.
Format as a bullet list per ticker.""", language="text")

    st.markdown("#### M4. Bearish Signal Checklist")
    st.code("""Today is [TODAY'S DATE]. Run through the following bearish checklist for U.S. equities and rate each signal as Red / Yellow / Green:
- VIX trend (3-week change): rising sharply = Red
- SPY vs 200-day MA: below = Red, within 5% = Yellow, healthy buffer = Green
- Credit spreads (HYG vs LQD): widening = Red
- Yield curve (10Y–2Y): deeply inverted = Red
- Fed posture: hawkish surprise = Red, neutral/dovish = Green
- Recent earnings trend: majority misses = Red

Summarize the overall market health verdict in one sentence.""", language="text")

    st.markdown("#### M5. Sector Rotation Insight")
    st.code("""Today is [TODAY'S DATE]. Based on the current macro cycle stage (early recovery / mid-cycle / late cycle / contraction), which U.S. equity sectors are typically in favour and which are typically out of favour?

Map this to the current market conditions. Which of these sectors are showing relative strength vs the S&P 500 right now:
Technology, Healthcare, Financials, Energy, Consumer Staples, Consumer Discretionary, Industrials, Utilities, Real Estate, Materials, Communication Services?

Rank the top 3 to overweight and bottom 3 to underweight, with one-line justification each.""", language="text")

    st.markdown("---")
    st.markdown("### 📈 Individual Ticker Research Prompts")

    st.markdown("#### 1. Fundamental & Deep Research")
    st.code("""I am researching the stock [TICKER]. Please provide a comprehensive 
overview of its business model, primary revenue streams, recent 
financial performance (last two earnings reports), and any 
significant regulatory or macroeconomic headwinds it is 
currently facing.""", language="text")

    st.markdown("#### 2. Buy & Sell Strategy Validation")
    st.code("""I am considering taking a position in [TICKER]. Given the current 
macroeconomic climate and the stock's recent price action, what 
are the most critical support and resistance levels to watch? 
Suggest a logical stop-loss percentage and a realistic price 
target for a 3-month hold.""", language="text")

    st.markdown("#### 3. Upcoming Catalysts & Speculations")
    st.code("""What are the major upcoming catalysts, product launches, clinical 
trials, or earnings reports for [TICKER] in the next 3 to 6 months? 
How might these events impact the stock price, and what are 
the current market speculations or analyst sentiments 
surrounding them?""", language="text")

    st.markdown("#### 4. Competitive Moat Analysis")
    st.code("""Analyze the competitive landscape for [TICKER]. Who are its top 3 
direct competitors? Compare [TICKER]'s market share, unique 
competitive advantages (moat), and profit margins against these 
competitors. Is [TICKER] gaining or losing ground?""", language="text")
