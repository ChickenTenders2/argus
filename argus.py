import yfinance as yf
import pandas as pd
import requests
import os
from datetime import datetime

# ── Config ──────────────────────────────────────────────
TELEGRAM_TOKEN   = "8673407952:AAGd20xArXTfZw27v2NtyaoGHMOeKebC0QA"
TELEGRAM_CHAT_ID = "1169760852"
MEMORY_FILE      = "argus_memory.csv"
WATCHLIST_FILE   = "argus_watchlist.csv"
MIN_SCORE        = 65
TOP_N            = 10

# ── Telegram ─────────────────────────────────────────────
def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    })

# ── Ticker Universe (Russell 2000 via iShares IWM) ───────
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
            print(f"✅ IWM download success: {len(tickers)} tickers")
            return tickers[:300]
        raise ValueError("Too few tickers parsed")
    except Exception as e:
        print(f"⚠️ IWM download failed: {e}")

    # Method 2: Try Wikipedia Russell 2000 component list
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Russell_2000_Index")
        for table in tables:
            for col in table.columns:
                if "ticker" in str(col).lower() or "symbol" in str(col).lower():
                    tickers = table[col].dropna().tolist()
                    tickers = [str(t).strip() for t in tickers if 1 < len(str(t).strip()) < 6]
                    if len(tickers) > 50:
                        print(f"✅ Wikipedia success: {len(tickers)} tickers")
                        return tickers[:300]
        raise ValueError("No ticker column found")
    except Exception as e:
        print(f"⚠️ Wikipedia failed: {e}")

    # Method 3: Try downloading IWM holdings via yfinance
    try:
        iwm = yf.Ticker("IWM")
        holdings = iwm.funds_data.top_holdings
        if holdings is not None and len(holdings) > 10:
            tickers = holdings.index.tolist()
            tickers = [str(t).strip() for t in tickers if 1 < len(str(t).strip()) < 6]
            print(f"✅ yfinance IWM holdings: {len(tickers)} tickers")
            return tickers
        raise ValueError("No holdings data")
    except Exception as e:
        print(f"⚠️ yfinance IWM failed: {e}")

    # Method 4: Hardcoded fallback
    print("⚠️ All sources failed — using hardcoded fallback list")
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

# ── Load / Save Memory ───────────────────────────────────
def load_memory():
    try:
        return pd.read_csv(MEMORY_FILE)
    except:
        return pd.DataFrame(columns=["ticker", "first_seen", "times_flagged", "last_score"])

def save_memory(df):
    df.to_csv(MEMORY_FILE, index=False)

# ── Scoring Engine ───────────────────────────────────────
def score_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info
        score = 0
        reasons = []
        red_flags = []

        # ── FUNDAMENTALS (max 50pts) ──
        rev_growth = info.get("revenueGrowth", 0) or 0
        if rev_growth >= 0.30:
            score += 25
            reasons.append(f"Rev growth {rev_growth*100:.0f}%")
        elif rev_growth >= 0.20:
            score += 15
            reasons.append(f"Rev growth {rev_growth*100:.0f}%")

        gross_margin = info.get("grossMargins", 0) or 0
        if gross_margin >= 0.50:
            score += 15
            reasons.append(f"Gross margin {gross_margin*100:.0f}%")
        elif gross_margin >= 0.30:
            score += 8
            reasons.append(f"Gross margin {gross_margin*100:.0f}%")

        fcf = info.get("freeCashflow", None)
        if fcf and fcf > 0:
            score += 10
            reasons.append("Positive FCF")

        # ── VALUATION (max 15pts) ──
        peg = info.get("pegRatio", None)
        if peg and 0 < peg < 1:
            score += 15
            reasons.append(f"PEG {peg:.1f} (undervalued)")
        elif peg and 1 <= peg < 2:
            score += 8
            reasons.append(f"PEG {peg:.1f}")

        # ── MOMENTUM (max 30pts) ──
        hist = stock.history(period="6mo")
        if not hist.empty and len(hist) > 50:
            price_now = hist["Close"].iloc[-1]
            price_6mo = hist["Close"].iloc[0]
            ma50      = hist["Close"].rolling(50).mean().iloc[-1]
            ma200     = hist["Close"].rolling(min(200, len(hist))).mean().iloc[-1]
            vol_avg   = hist["Volume"].rolling(30).mean().iloc[-1]
            vol_today = hist["Volume"].iloc[-1]

            if price_now > price_6mo * 1.15:
                score += 10
                reasons.append("Strong 6mo momentum")
            if price_now > ma50:
                score += 5
                reasons.append("Above 50MA")
            if price_now > ma200:
                score += 5
                reasons.append("Above 200MA")
            if vol_today > vol_avg * 2:
                score += 10
                reasons.append("⚡ Unusual volume spike")

        # ── SMART MONEY (max 20pts) ──
        inst_own = info.get("heldPercentInstitutions", 1) or 1
        if inst_own < 0.40:
            score += 20
            reasons.append(f"Low inst. ownership {inst_own*100:.0f}%")
        elif inst_own < 0.60:
            score += 10
            reasons.append(f"Moderate inst. ownership {inst_own*100:.0f}%")
        elif inst_own < 0.75:
            score += 5

        # ── MARKET CAP (max 10pts) ──
        mkt_cap = info.get("marketCap", 0) or 0
        if 50_000_000 < mkt_cap < 10_000_000_000:
            score += 10
            reasons.append(f"Small/mid cap ${mkt_cap/1e6:.0f}M")
        elif mkt_cap >= 10_000_000_000:
            score += 5

        # ── RED FLAG VETOES ──
        debt_eq = info.get("debtToEquity", 0) or 0
        if debt_eq > 500:
            red_flags.append("Extreme debt")

        short_pct = info.get("shortPercentOfFloat", 0) or 0
        if short_pct > 0.45:
            red_flags.append(f"Extreme short interest {short_pct*100:.0f}%")

        shares_out = info.get("sharesPercentSharesOut", 0) or 0
        if shares_out > 0.15:
            red_flags.append("Dilution risk")

        if red_flags:
            return None

        # ── TIER ──
        if score >= 80:
            tier = "🟢 HIGH CONVICTION"
        elif score >= 65:
            tier = "🟡 WATCHLIST"
        else:
            return None

        return {
            "ticker":  ticker,
            "score":   score,
            "tier":    tier,
            "reasons": reasons,
            "price":   round(hist["Close"].iloc[-1], 2) if not hist.empty else "N/A",
            "mkt_cap": f"${mkt_cap/1e6:.0f}M" if mkt_cap else "N/A"
        }

    except Exception:
        return None

# ── Format Telegram Message ──────────────────────────────
def format_pick(pick, memory_df):
    ticker     = pick["ticker"]
    prev       = memory_df[memory_df["ticker"] == ticker]
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

# ── Watchlist Monitor ────────────────────────────────────
def run_watchlist_monitor():
    try:
        wl = pd.read_csv(WATCHLIST_FILE)
        tickers = wl["ticker"].dropna().tolist()
    except:
        return

    if not tickers:
        return

    lines = [f"👁 *Argus Watchlist Update — {datetime.now().strftime('%d %b %Y')}*\n{'─'*30}"]
    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker).history(period="2d")
            if len(hist) >= 2:
                price   = round(hist["Close"].iloc[-1], 2)
                day_chg = ((hist["Close"].iloc[-1] - hist["Close"].iloc[-2]) / hist["Close"].iloc[-2]) * 100
                arrow   = "📈" if day_chg > 0 else "📉"
                chg_str = f"{'+' if day_chg > 0 else ''}{day_chg:.1f}%"
                lines.append(f"{arrow} *{ticker}* — ${price} ({chg_str} today)")
            else:
                lines.append(f"📌 *{ticker}* — data unavailable")
        except:
            lines.append(f"📌 *{ticker}* — data unavailable")

    send_telegram("\n".join(lines))

# ── Main ─────────────────────────────────────────────────
def main():
    print("👁 Argus scan starting...")
    tickers   = get_universe()
    memory_df = load_memory()
    results   = []

    for ticker in tickers:
        pick = score_stock(ticker)
        if pick:
            results.append(pick)

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:TOP_N]

    if not results:
        send_telegram(
            f"👁 *Argus Daily Scan — {datetime.now().strftime('%d %b %Y')}*\n"
            f"No high-conviction picks found today. Market may be choppy."
        )
        run_watchlist_monitor()
        return

    # ── Update memory ──
    today = datetime.now().strftime("%d %b %Y")
    for pick in results:
        t = pick["ticker"]
        if t in memory_df["ticker"].values:
            memory_df.loc[memory_df["ticker"] == t, "times_flagged"] += 1
            memory_df.loc[memory_df["ticker"] == t, "last_score"]    = pick["score"]
        else:
            new_row = pd.DataFrame([{
                "ticker":        t,
                "first_seen":    today,
                "times_flagged": 1,
                "last_score":    pick["score"]
            }])
            memory_df = pd.concat([memory_df, new_row], ignore_index=True)
    save_memory(memory_df)

    # ── Build & send Telegram message ──
    today_str = datetime.now().strftime("%d %b %Y")
    header    = f"👁 *Argus Daily Scan — {today_str}*\n{'─'*30}\n"
    body      = "\n".join([format_pick(p, memory_df) for p in results])
    footer    = f"\n{'─'*30}\n_Scanned {len(tickers)} tickers • Top {len(results)} picks shown_"
    send_telegram(header + body + footer)

    # ── Watchlist monitor ──
    run_watchlist_monitor()
    print("✅ Argus scan complete.")

if __name__ == "__main__":
    main()
