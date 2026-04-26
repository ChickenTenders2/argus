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
    get_market_regime,
    get_db_connection,
    generate_telegram_message,
    monitor_portfolio
)
import os
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


def show_aggrid(df, height=380, fit_columns=True):
    """AgGrid table with fallback to st.dataframe when library is absent."""
    if not HAS_AGGRID or df.empty:
        st.dataframe(df, use_container_width=True)
        return
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
            theme="alpine",
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

@st.cache_data(ttl=3600)
def fetch_current_prices(tickers):
    """Bulk fetch current prices to calculate historical return."""
    if not tickers: return {}
    try:
        data = yf.download(list(tickers), period="5d", progress=False)
        prices = {}
        # Handle yf.download multiindex response based on number of tickers
        if len(tickers) == 1:
            t = tickers[0]
            if "Close" in data.columns:
                prices[t] = float(data["Close"].dropna().iloc[-1])
        else:
            if "Close" in data.columns:
                close_df = data["Close"]
                for t in tickers:
                    if t in close_df.columns:
                        prices[t] = float(close_df[t].dropna().iloc[-1])
        return prices
    except Exception as e:
        return {}

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
        st.session_state.preset_desc = "The opposite of wide scanning — ultra-strict score filter to surface only the absolute best setups."
        st.session_state.min_score = 90
        st.session_state.horizon_days = 63
        st.session_state.target_return = 15
        st.session_state.risk_per_trade_pct = 1.50
        st.session_state.max_position_pct = 15.00
        st.session_state.vol_floor = 500000
        st.session_state.price_floor = 2.0
    elif preset == "Liquidity Focus":
        st.session_state.preset_desc = "Raises the volume floor dramatically to only trade highly liquid, large-cap names."
        st.session_state.min_score = 65
        st.session_state.horizon_days = 63
        st.session_state.target_return = 10
        st.session_state.risk_per_trade_pct = 0.75
        st.session_state.max_position_pct = 10.00
        st.session_state.vol_floor = 2000000
        st.session_state.price_floor = 10.00
    elif preset == "Momentum Sprint":
        st.session_state.preset_desc = "Very short horizon, high target return — designed to catch explosive breakouts."
        st.session_state.min_score = 72
        st.session_state.horizon_days = 21
        st.session_state.target_return = 30
        st.session_state.risk_per_trade_pct = 1.25
        st.session_state.max_position_pct = 8.00
        st.session_state.vol_floor = 500000
        st.session_state.price_floor = 2.0
    elif preset == "Capital Preservation":
        st.session_state.preset_desc = "Designed for deep bear markets or extreme uncertainty. Minimal risk, long timeframe."
        st.session_state.min_score = 85
        st.session_state.horizon_days = 126
        st.session_state.target_return = 3
        st.session_state.risk_per_trade_pct = 0.15
        st.session_state.max_position_pct = 5.00
        st.session_state.vol_floor = 500000
        st.session_state.price_floor = 2.0
    elif preset == "Small Cap Hunter":
        st.session_state.preset_desc = "Lowers price and volume floors to scan micro/small-caps. Higher risk tolerance."
        st.session_state.min_score = 68
        st.session_state.horizon_days = 84
        st.session_state.target_return = 40
        st.session_state.risk_per_trade_pct = 0.50
        st.session_state.max_position_pct = 5.00
        st.session_state.vol_floor = 100000
        st.session_state.price_floor = 0.50
    elif preset == "Earnings Season":
        st.session_state.preset_desc = "Tightened around the 21-day horizon to capture pre/post earnings momentum."
        st.session_state.min_score = 65
        st.session_state.horizon_days = 21
        st.session_state.target_return = 12
        st.session_state.risk_per_trade_pct = 0.75
        st.session_state.max_position_pct = 6.00
        st.session_state.vol_floor = 500000
        st.session_state.price_floor = 2.0
    elif preset == "Swing Recovery":
        st.session_state.preset_desc = "Targets beaten-down tickers with strong fundamentals. Medium horizon, moderate score."
        st.session_state.min_score = 70
        st.session_state.horizon_days = 42
        st.session_state.target_return = 15
        st.session_state.risk_per_trade_pct = 0.60
        st.session_state.max_position_pct = 7.00
        st.session_state.vol_floor = 500000
        st.session_state.price_floor = 2.0
    elif preset == "Aggressive Growth":
        st.session_state.preset_desc = "Standard aggressive growth."
        st.session_state.min_score = 70
        st.session_state.horizon_days = 42
        st.session_state.target_return = 20
        st.session_state.risk_per_trade_pct = 1.0
        st.session_state.max_position_pct = 8.00
        st.session_state.vol_floor = 500000
        st.session_state.price_floor = 2.0
    elif preset == "Bear Market Defense":
        st.session_state.preset_desc = "Standard bear market defense."
        st.session_state.min_score = 80
        st.session_state.horizon_days = 84
        st.session_state.target_return = 5
        st.session_state.risk_per_trade_pct = 0.25
        st.session_state.max_position_pct = 5.00
        st.session_state.vol_floor = 500000
        st.session_state.price_floor = 2.0
        st.session_state.price_ceiling = 0.0
    elif preset == "Penny Stock High Risk ($1-$10)":
        st.session_state.preset_desc = "Extremely high risk, high reward plays constrained to penny stocks between $1 and $10."
        st.session_state.min_score = 60
        st.session_state.horizon_days = 21
        st.session_state.target_return = 50
        st.session_state.risk_per_trade_pct = 2.0
        st.session_state.max_position_pct = 5.00
        st.session_state.vol_floor = 100000
        st.session_state.price_floor = 1.0
        st.session_state.price_ceiling = 10.0
    elif preset == "Default":
        st.session_state.preset_desc = "A balanced setup suitable for normal market conditions."
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
                    # Highlight 'High Conviction' with a different color if needed
                    color = "green" if "HIGH CONVICTION" in row["tier"] else "orange"
                    st.markdown(f"**:{color}[{row['tier']}]**")
                
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
                        if HAS_ANNOTATED:
                            try:
                                tier_str = row.get("tier", "")
                                badge_bg = "#1a4731" if "HIGH CONVICTION" in str(tier_str) else "#1a2e4a"
                                annotated_text(*[(r.strip(), "", badge_bg) for r in reasons if r.strip()])
                            except Exception:
                                st.markdown("  \n".join(f"• {r}" for r in reasons))
                        else:
                            bullets = ""
                            for r in reasons:
                                parts = r.split(" ")
                                if len(parts) > 1 and parts[-1].replace('%', '').replace('.', '').replace('x', '').isdigit():
                                    name = " ".join(parts[:-1])
                                    val = parts[-1]
                                    bullets += f"• {name} **{val}**  \n"
                                else:
                                    bullets += f"• {r}  \n"
                            st.markdown(bullets)
                    else:
                        st.markdown(f"• {reasons}")
                
                st.button(
                    "Deep Dive", 
                    key=f"dd_{row['ticker']}_{i}_{row.get('scan_date', 'temp')}", 
                    use_container_width=True,
                    on_click=nav_to_ticker,
                    args=(row["ticker"],)
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
    
    st.info(st.session_state.preset_desc)

    st.header("Quick Scan Settings")
    min_score = st.slider("Minimum Score", 50, 95, key="min_score")
    scan_limit = st.slider("Universe size (tickers to scan)", 50, 400, 200, step=50)

    with st.expander("Advanced Filters & Constraints"):
        price_floor = st.number_input("Price Floor ($)", min_value=0.0, max_value=100.0, key="price_floor")
        price_ceiling = st.number_input("Price Ceiling ($) [0 = No Limit]", min_value=0.0, max_value=1000.0, key="price_ceiling")
        vol_floor = st.number_input("Volume Floor (avg daily)", step=100_000, min_value=0, key="vol_floor")

    with st.expander("ML Prediction Horizons"):
        horizon_days = st.selectbox("Horizon days", [21, 42, 63, 84, 126], key="horizon_days")
        target_return = st.slider("Target return (%)", 1, 100, key="target_return") / 100.0

    with st.expander("Advanced Risk Rules"):
        risk_per_trade_pct = st.slider("Risk per trade (% of portfolio)", 0.10, 5.00, step=0.05, key="risk_per_trade_pct")
        max_position_pct = st.slider("Max position size (%)", 1.0, 30.0, step=0.5, key="max_position_pct")

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

# To restore Portfolio Optimizer, add "Portfolio Optimizer" back to this array
tab_options = ["Overview", "Ticker Detail", "Manual Run", "History", "Journal", "Prediction Model", "Alerts Log", "Help", "Prompts"]
active_tab = st.radio("Navigation", tab_options, key="main_tabs", horizontal=True, label_visibility="collapsed")
st.markdown("<br>", unsafe_allow_html=True)


if active_tab == "Overview":
    st.subheader("📡 Latest Scheduled Scan")
    
    # Check Macro Market Regime
    regime = cached_market_regime()
    # Apply a color mapping to the regime text
    regime_color_map = {
        "Bull": "green",
        "Neutral": "blue",
        "Bear": "orange",
        "Extreme Fear": "red"
    }
    r_color = regime_color_map.get(regime["regime"], "white")
    st.markdown(f"**Current Market Regime:** :{r_color}[**{regime['regime']}**] *(Multiplier: {regime['multiplier']}x - {regime['reason']})*")
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
                    style = day_df.style
                    if "Return (%)" in day_df.columns:
                        style = style.background_gradient(subset=["Return (%)"], cmap="RdYlGn")
                    st.dataframe(style, use_container_width=True)

if active_tab == "Ticker Detail":
    st.subheader("🔎 Ticker Detail")
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
                st.markdown(f"**Buy:** Score of **{score_val}** & Upside Prob **{prob_str}**. "
                            f"Entry style is **{entry}**.")
                st.markdown(f"**Sell:** Stop loss at **-{sl_pct:.1f}%**. "
                            f"Target taking profits around **+{tp_pct:.1f}%**.")
            
                st.markdown("#### Groq Investment Thesis")
                if st.button("Generate Qualitative Analysis", key=f"btn_ai_{ticker}"):
                    with st.spinner(f"Analyzing {ticker} news and factors with Llama 3..."):
                        import os
                        from engine import Config
                        try:
                            from llm import generate_ai_thesis
                            
                            cfg = Config()
                            api_key = cfg.GROQ_API_KEY
                            if not api_key:
                                st.error("GROQ_API_KEY is not set in your environment variables. Please set it to use the AI analysis feature.")
                            else:
                                reasons = latest_ticker_row["reasons"].iloc[0]
                                if isinstance(reasons, str):
                                    if reasons.startswith("["):
                                        try:
                                            reasons = json.loads(reasons)
                                        except (json.JSONDecodeError, ValueError):
                                            reasons = [reasons]
                                    else:
                                        reasons = [reasons]
                                elif not isinstance(reasons, list):
                                    reasons = []
                                
                                thesis = generate_ai_thesis(ticker, score_val, reasons, api_key)
                                st.success("Analysis Complete!")
                                st.info(thesis)
                        except ImportError:
                            st.error("llm module not found. Make sure llm.py is properly set up.")
                        except Exception as e:
                            st.error(f"Error generating thesis: {e}")

            st.markdown("#### Execution Guidance")
            
            # --- MIDDLE SECTION: Price History & Detailed Snapshot ---
            colC, colD = st.columns([2, 1])
            with colC:
                st.markdown("#### Price & Score History (1 Year)")
                hist = fetch_ticker_history(ticker, period="1y")
                
                try:
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots
                    
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    if not hist.empty:
                        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], name="Price", line=dict(color='blue')), secondary_y=False)
                        
                    tdf_scores = tdf.dropna(subset=["scan_date", "score"]).sort_values("scan_date")
                    if not tdf_scores.empty:
                        fig.add_trace(go.Scatter(x=tdf_scores["scan_date"], y=tdf_scores["score"], name="Argus Score", mode="lines+markers", line=dict(dash='dot', color='orange')), secondary_y=True)
                        
                    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified")
                    fig.update_yaxes(title_text="Close Price", secondary_y=False)
                    fig.update_yaxes(title_text="Argus Score", secondary_y=True)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning("Could not render dual-axis chart.")
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
            
            st.markdown("---")
            
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
def add_journal_entry_dialog(cfg):
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
                    },
                )
                st.success("Journal entry saved.")
                st.rerun()

if active_tab == "Journal":
    st.subheader("📝 Decision Journal")
    
    st.checkbox("Enable Advanced Tracking (Position sizing & Scan History)", key="enable_journal_advanced")
    
    if st.button("➕ Add New Entry"):
        add_journal_entry_dialog(config)

    journal_df = load_journal(config.JOURNAL_FILE)
    if journal_df.empty:
        st.info("No journal entries yet.")
    else:
        st.markdown("### 📋 Logged Entries")
        
        # Add deletion capability via data_editor
        edited_df = st.data_editor(
            journal_df.sort_values("timestamp", ascending=False),
            num_rows="dynamic",
            use_container_width=True,
            key="journal_editor"
        )
        
        if not edited_df.equals(journal_df.sort_values("timestamp", ascending=False)):
            if st.button("💾 Save Edited Log"):
                conn = get_db_connection()
                edited_df.to_sql("journal", conn, if_exists="replace", index=False)
                conn.close()
                st.success("Log updated successfully!")
                st.rerun()
        
        st.markdown("### 📊 Portfolio P&L Status")
        
        # Calculate Realized and Unrealized P&L
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
                # Shares-weighted average buy price (fall back to simple mean if shares absent)
                if total_buy_shares > 0:
                    avg_buy = float((buys["entry_price"] * buy_shares).sum() / total_buy_shares)
                else:
                    avg_buy = float(buys["entry_price"].mean())

                # Check if position is open (net shares still held)
                net_shares = max(0.0, total_buy_shares - total_sell_shares)
                if net_shares > 0 or (buy_count > sell_count and total_buy_shares == 0):
                    first_buy_date = buys["timestamp"].min()
                    first_buy_date_str = str(first_buy_date)[:10] if pd.notnull(first_buy_date) else datetime.now().strftime("%Y-%m-%d")
                    open_positions.append({"Ticker": t, "Avg Buy": avg_buy, "Shares": net_shares, "First Date": first_buy_date_str})

                # Track realized trades if any sells exist
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
                
                st.dataframe(pnl_df.style.background_gradient(subset=["Return (%)"], cmap="RdYlGn"), use_container_width=True)
            else:
                st.info("No closed trades found.")
                
        with unrealized_tab:
            if open_positions:
                open_df = pd.DataFrame(open_positions).round(2)
                with st.spinner("Fetching live prices for open positions..."):
                    current_prices = fetch_current_prices(open_df["Ticker"].tolist())
                
                open_df["Current Price"] = open_df["Ticker"].map(current_prices).round(2)
                
                # Portfolio Math
                open_df["Cost Basis ($)"] = (open_df["Shares"] * open_df["Avg Buy"]).round(2)
                open_df["Current Value ($)"] = (open_df["Shares"] * open_df["Current Price"]).round(2)
                open_df["Unrealized (%)"] = (((open_df["Current Price"] - open_df["Avg Buy"]) / open_df["Avg Buy"]) * 100).round(2)
                
                # Top Level Analytics
                total_invested = open_df["Cost Basis ($)"].sum()
                total_current = open_df["Current Value ($)"].sum()
                net_return_pct = ((total_current - total_invested) / total_invested * 100) if total_invested > 0 else 0
                
                # SPY Benchmark
                first_entry_date = open_df["First Date"].min()
                spy_return = get_spy_return(first_entry_date)
                
                # Scoreboard UI
                st.markdown("### 📊 Portfolio Analytics")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Invested", f"${total_invested:,.2f}")
                m2.metric("Current Value", f"${total_current:,.2f}")
                m3.metric("Net Return", f"{net_return_pct:+.2f}%")
                m4.metric("SPY Benchmark", f"{spy_return:+.2f}%", help=f"S&P 500 Return since earliest active entry ({first_entry_date})")
                
                st.divider()
                
                # Sector mapping and visualization
                if total_invested > 0:
                    with st.spinner("Mapping sectors..."):
                        open_df["Sector"] = open_df["Ticker"].apply(get_cached_sector)
                    
                    sector_totals = open_df.groupby("Sector")["Current Value ($)"].sum().reset_index()
                    fig = px.pie(sector_totals, values='Current Value ($)', names='Sector', title="Sector Exposure by Dollar Value", hole=0.3)
                    
                    ui_col1, ui_col2 = st.columns([1, 1])
                    with ui_col1:
                        st.plotly_chart(fig, use_container_width=True)
                    with ui_col2:
                        st.markdown("<br><br>", unsafe_allow_html=True) # padding
                        st.markdown("#### Open Positions")
                        # Drop some internal cols for cleaner table
                        display_df = open_df.drop(columns=["First Date", "Sector"])
                        st.dataframe(display_df.style.background_gradient(subset=["Unrealized (%)"], cmap="RdYlGn"), use_container_width=True, hide_index=True)
                else:
                    display_df = open_df.drop(columns=["First Date"])
                    st.dataframe(display_df.style.background_gradient(subset=["Unrealized (%)"], cmap="RdYlGn"), use_container_width=True, hide_index=True)

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
                        # Fallback for raw sqlite3
                        conn.execute("DELETE FROM journal WHERE ticker = ?", (ticker_to_delete,))
                    conn.commit()
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
        with open("argus_alerts_log.txt", "r", encoding="utf-8") as f:
            logs = f.read()
            if logs.strip():
                # Display newest first
                blocks = [block.strip() for block in logs.split("---") if block.strip()]
                # Reconstruct and reverse
                reversed_logs = ""
                for block in reversed(blocks):
                    # check if the block starts with date (e.g. 2026-04-11)
                    if len(block) >= 19 and block[4] == '-' and block[7] == '-':
                        date_str = block[:19]
                        msg = block[19:].strip()
                        st.markdown(f"**{date_str}**")
                        st.info(msg)
                    else:
                        st.info(block)
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
    * **Overview:** Snapshot of the latest scan results with interactive visual metric cards. View the current macro Market Regime and top stock picks.
    * **Ticker Detail:** Deep dive into specific tickers. Evaluate Plotly interactive price charts, quantitative execution guidance, and generate qualitative AI investment thesis reports (Groq).
    * **Portfolio Optimizer:** Select multiple tickers from the recent scan to calculate the statistically optimal portfolio sizing using the efficient frontier.
    * **Manual Run:** Instantly run a new global scan based on your sidebar settings. Includes Telegram notification support and Auto-Pilot portfolio monitoring.
    * **History:** Browse historical database scans with interactive tables. Select any row to view a visual card grid of its scoring reasons.
    * **Journal:** Personal logbook featuring a **Live Portfolio Analytics Dashboard**. Track Total Invested, Current Value, Net Return %, and visualize Sector Diversification via interactive pie charts. Benchmarks portfolio performance against the S&P 500 (SPY). Powered by the Auto-Pilot monitor for live SL/TP tracking.
    * **Prediction Model:** ML diagnostics. Review XGBoost hit rates, Brier scores, and calibration accuracy across different market conditions.
    * **Alerts Log:** Record of all automated signals pushed via Telegram.
    * **Prompts:** Curated LLM prompts to assist with deeper external research.

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
    st.caption("Copy these templates and paste them into AI tools (like Perplexity, ChatGPT, or Claude) to deeper analyze tickers.")

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
