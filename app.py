import streamlit as st
import pandas as pd
import yfinance as yf
from engine import Config, score_stock, load_memory, get_universe

st.set_page_config(page_title="Argus Dashboard", layout="wide")
st.title("👁 Argus Investment Workstation")

with st.sidebar:
    st.header("Parameters")
    min_score = st.slider("Minimum Score", 50, 95, 65)
    price_floor = st.number_input("Price Floor ($)", 0.0, 100.0, 2.0)
    
    config = Config(MIN_SCORE=min_score, PRICE_FLOOR=price_floor)

if st.button("Run Global Scan"):
    tickers = get_universe()
    memory_df = load_memory(config.MEMORY_FILE)
    
    results = []
    progress = st.progress(0)
    for i, ticker in enumerate(tickers[:50]): # Limiting for demo
        pick = score_stock(ticker, memory_df, config)
        if pick: results.append(pick)
        progress.progress((i + 1) / 50)

    if results:
        df_res = pd.DataFrame(results)
        df_res = df_res.sort_values("score", ascending=False)
        st.dataframe(df_res)
        
        st.subheader("High Conviction Charts (Score >= 80)")
        high_conviction = df_res[df_res["tier"] == "🟢 HIGH CONVICTION"]
        
        if not high_conviction.empty:
            # Create columns for charts (up to 3 per row)
            cols = st.columns(3)
            for idx, row in high_conviction.reset_index(drop=True).iterrows():
                with cols[idx % 3]:
                    st.write(f"**{row['ticker']}** - Score: {row['score']}")
                    hist = yf.Ticker(row['ticker']).history(period="6mo")
                    if not hist.empty:
                        st.line_chart(hist['Close'])
                    else:
                        st.write("No history available")
        else:
            st.info("No high conviction picks found in this run.")
    else:
        st.warning("No tickers met the current criteria.")
