"""Reusable UI components for the Argus modern fintech dashboard."""

import streamlit as st
import json


def render_sparkline(values: list, color: str = "#00d4ff", width: int = 64, height: int = 18) -> str:
    """Return an inline SVG sparkline for a list of numeric values.

    Returns an HTML string ready for st.markdown(..., unsafe_allow_html=True).
    """
    if not values or len(values) < 2:
        return ""
    vals = [float(v) for v in values if v is not None]
    if not vals or len(vals) < 2:
        return ""
    mn, mx = min(vals), max(vals)
    rng = mx - mn if mx != mn else 1.0
    pts = []
    for i, v in enumerate(vals):
        x = i * width / (len(vals) - 1)
        y = height - ((v - mn) / rng) * (height - 2) - 1
        pts.append(f"{x:.1f},{y:.1f}")
    points_str = " ".join(pts)
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="vertical-align:middle;margin-left:6px">'
        f'<polyline points="{points_str}" fill="none" stroke="{color}" '
        f'stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>'
        f'</svg>'
    )


def render_catalyst_pills(reasons: list) -> str:
    """Return HTML for catalyst pills extracted from reasons list.

    Catalyst reasons are those from catalysts.py (insider cluster, 8-K, options flow).
    """
    catalyst_keywords = [
        "insider", "cluster", "8-k", "material agreement", "reg fd", "earnings beat",
        "options:", "iv rank", "bullish flow", "fda", "defense", "partnership",
        "pre-revenue", "cash runway", "squeeze setup", "tight float",
    ]
    pills = []
    for r in reasons:
        r_lower = str(r).lower()
        if any(kw in r_lower for kw in catalyst_keywords):
            pills.append(f'<span class="argus-catalyst-pill">{r}</span>')
    if not pills:
        return ""
    return '<div style="margin:4px 0">' + "".join(pills) + "</div>"


def render_velocity_badge(velocity: float) -> str:
    """Return an HTML badge showing score velocity direction."""
    if velocity is None:
        return ""
    if velocity >= 8:
        return f'<span class="argus-velocity-up">↑ +{velocity:.0f} pts</span>'
    elif velocity >= 3:
        return f'<span class="argus-velocity-up">↗ +{velocity:.0f}</span>'
    elif velocity <= -8:
        return f'<span class="argus-velocity-down">↓ {velocity:.0f} pts</span>'
    elif velocity <= -3:
        return f'<span class="argus-velocity-down">↘ {velocity:.0f}</span>'
    else:
        return f'<span class="argus-velocity-flat">→ {velocity:+.0f}</span>'


def render_rr_chip(stop_pct: float, tp_pct: float) -> str:
    """Return a risk/reward ratio chip HTML string."""
    if not stop_pct or not tp_pct or stop_pct <= 0:
        return ""
    ratio = tp_pct / stop_pct
    return f'<span class="argus-rr-chip">R/R {ratio:.1f}:1 ({stop_pct:.0f}%/{tp_pct:.0f}%)</span>'


def render_score_waterfall(pick: dict) -> None:
    """Render a score breakdown waterfall using st.markdown inside a card."""
    f = int(pick.get("f_score", 0) or 0)
    v = int(pick.get("v_score", 0) or 0)
    m = int(pick.get("m_score", 0) or 0)
    s = int(pick.get("s_score", 0) or 0)
    p = int(pick.get("p_score", 0) or 0)
    c = int(pick.get("c_score", 0) or 0)
    r = int(pick.get("r_score", 0) or 0)
    raw_pre = int(pick.get("raw_score_pre_regime", 0) or pick.get("raw_score", 0) or 0)
    final   = int(pick.get("score", 0) or 0)

    def _bar(pts: int, max_pts: int, color: str) -> str:
        pct = min(100, int(pts / max_pts * 100)) if max_pts > 0 else 0
        return (
            f'<div style="display:flex;align-items:center;gap:8px;margin:2px 0">'
            f'<div style="width:90px;font-size:0.72rem;color:#8ea0ba;text-align:right">'
            f'</div>'
            f'<div style="flex:1;background:rgba(255,255,255,0.06);border-radius:4px;height:6px">'
            f'<div style="width:{pct}%;background:{color};border-radius:4px;height:6px"></div>'
            f'</div>'
            f'<div style="width:32px;font-size:0.72rem;color:#e8eef7;text-align:right">{pts}</div>'
            f'</div>'
        )

    components = [
        ("Fundamentals", f, 18, "#26d97f"),
        ("Valuation",    v, 4,  "#00d4ff"),
        ("Momentum",     m, 30, "#ffb547"),
        ("Smart Money",  s, 22, "#bf5af2"),
        ("Catalyst",     c, 22, "#ff8c42"),
        ("Runner Sim",   r, 5,  "#ff4d6d"),
        ("Persistence",  p, 5,  "#8ea0ba"),
    ]

    rows = ""
    for name, pts, mx, col in components:
        pct = min(100, int(pts / mx * 100)) if mx > 0 else 0
        rows += (
            f'<div style="display:flex;align-items:center;gap:6px;margin:1px 0">'
            f'<div style="width:80px;font-size:0.70rem;color:#8ea0ba;text-align:right;flex-shrink:0">{name}</div>'
            f'<div style="flex:1;background:rgba(255,255,255,0.06);border-radius:3px;height:5px">'
            f'<div style="width:{pct}%;background:{col};border-radius:3px;height:5px"></div>'
            f'</div>'
            f'<div style="width:24px;font-size:0.70rem;color:#e8eef7;text-align:right;flex-shrink:0">{pts}</div>'
            f'</div>'
        )

    regime_note = ""
    if raw_pre and final != raw_pre:
        direction = "↑" if final > raw_pre else "↓"
        regime_note = f'<div style="font-size:0.70rem;color:#8ea0ba;margin-top:4px">Regime adj: {raw_pre} {direction} {final}</div>'

    html = (
        f'<div style="padding:6px 0">'
        f'<div style="font-size:0.70rem;color:#8ea0ba;margin-bottom:4px;font-weight:600;text-transform:uppercase;letter-spacing:0.05em">Score Breakdown</div>'
        f'{rows}'
        f'{regime_note}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_hero_card(pick: dict, rank: int = 1) -> None:
    """Render a large hero card for a top pick in the Overview."""
    from theme import REGIME_COLORS
    ticker    = pick.get("ticker", "")
    score     = int(pick.get("score", 0) or 0)
    sector    = pick.get("sector", "Unknown")
    reasons   = pick.get("reasons", [])
    velocity  = pick.get("score_velocity", 0) or 0
    price     = pick.get("price", 0) or 0

    if score >= 85:
        sig, sig_color = "Strong Buy", "#26d97f"
    elif score >= 75:
        sig, sig_color = "Buy", "#00d4ff"
    elif score >= 65:
        sig, sig_color = "Moderate Buy", "#ffb547"
    else:
        sig, sig_color = "Watch", "#8ea0ba"

    velocity_html = render_velocity_badge(velocity)
    catalyst_html = render_catalyst_pills(reasons if isinstance(reasons, list) else [])

    # Inline hero card HTML
    html = f"""
<div class="argus-hero-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start">
    <div>
      <div style="font-size:0.72rem;color:#8ea0ba;margin-bottom:2px">#{rank} · {sector}</div>
      <div class="argus-hero-ticker">{ticker}</div>
      <div class="argus-hero-signal" style="color:{sig_color}">{sig}</div>
    </div>
    <div style="text-align:right">
      <div class="argus-hero-score">{score}</div>
      <div style="font-size:0.72rem;color:#8ea0ba">/100</div>
    </div>
  </div>
  <div style="margin-top:8px;display:flex;align-items:center;gap:8px;flex-wrap:wrap">
    <span style="font-size:0.78rem;color:#e8eef7">💲{price:.2f}</span>
    {velocity_html}
  </div>
  {catalyst_html if catalyst_html else ""}
</div>
"""
    st.markdown(html, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button(f"🔍 Deep Dive", key=f"hero_dd_{ticker}_{rank}"):
            st.session_state["selected_ticker"] = ticker
            st.session_state["nav"] = "Ticker Detail"
            st.rerun()
    with c2:
        if st.button(f"📝 Log Trade", key=f"hero_log_{ticker}_{rank}"):
            st.session_state["journal_prefill_ticker"] = ticker
            st.session_state["nav"] = "Journal"
            st.rerun()


def render_sector_heatmap(df) -> None:
    """Render an 11×3 sector × score-bucket conviction heatmap using st.markdown."""
    import pandas as pd
    if df.empty:
        return

    sectors = [
        "Technology", "Healthcare", "Financial Services", "Consumer Cyclical",
        "Industrials", "Communication Services", "Real Estate",
        "Consumer Defensive", "Energy", "Basic Materials", "Utilities",
    ]

    buckets = [
        ("≥80 HC", lambda s: s >= 80, "rgba(38,217,127,0.6)"),
        ("65–79",  lambda s: 65 <= s < 80, "rgba(0,212,255,0.4)"),
        ("50–64",  lambda s: 50 <= s < 65, "rgba(255,181,71,0.3)"),
    ]

    header = (
        '<div style="display:grid;grid-template-columns:110px 1fr 1fr 1fr;gap:4px;margin-bottom:4px">'
        '<div style="font-size:0.68rem;color:#8ea0ba">Sector</div>'
    )
    for label, _, _ in buckets:
        header += f'<div style="font-size:0.68rem;color:#8ea0ba;text-align:center">{label}</div>'
    header += '</div>'

    rows_html = ""
    for sector in sectors:
        sec_df = df[df.get("sector", pd.Series()).eq(sector)] if "sector" in df.columns else pd.DataFrame()
        if sec_df.empty and "Sector" in df.columns:
            sec_df = df[df["Sector"] == sector]
        row = f'<div style="display:grid;grid-template-columns:110px 1fr 1fr 1fr;gap:4px;margin-bottom:2px">'
        row += f'<div style="font-size:0.70rem;color:#8ea0ba;align-self:center;overflow:hidden;white-space:nowrap;text-overflow:ellipsis">{sector[:14]}</div>'
        for _, fn, bg in buckets:
            if sec_df.empty:
                count = 0
            else:
                score_col = "score" if "score" in sec_df.columns else "Score"
                count = int((sec_df[score_col].apply(fn)).sum()) if score_col in sec_df.columns else 0
            cell_bg = bg if count > 0 else "rgba(255,255,255,0.03)"
            cell_color = "#e8eef7" if count > 0 else "#3a4a5a"
            row += (
                f'<div class="argus-hm-cell" style="background:{cell_bg};color:{cell_color}">'
                f'{count if count > 0 else "·"}</div>'
            )
        row += '</div>'
        rows_html += row

    st.markdown(
        f'<div style="margin:8px 0">{header}{rows_html}</div>',
        unsafe_allow_html=True,
    )


def render_catalyst_calendar(picks_df) -> None:
    """Render a mini earnings calendar for the next 7 days from HC picks."""
    if picks_df is None or picks_df.empty:
        return

    from datetime import datetime, timedelta
    today = datetime.now().date()
    week_out = today + timedelta(days=7)

    items = []
    for _, row in picks_df.iterrows():
        try:
            import yfinance as yf
            cal = yf.Ticker(str(row.get("ticker", ""))).calendar
            if isinstance(cal, dict) and "Earnings Date" in cal:
                ed = cal["Earnings Date"]
                if hasattr(ed, "date"):
                    ed = ed.date()
                elif isinstance(ed, str):
                    ed = datetime.strptime(ed[:10], "%Y-%m-%d").date()
                if today <= ed <= week_out:
                    items.append({"ticker": row.get("ticker"), "date": ed,
                                  "score": row.get("score", 0)})
        except Exception:
            pass

    if not items:
        return

    items.sort(key=lambda x: x["date"])
    st.markdown("**📅 Earnings this week (HC picks)**")
    for item in items[:5]:
        days = (item["date"] - today).days
        label = "today" if days == 0 else f"in {days}d"
        st.caption(f"**{item['ticker']}** — {item['date'].strftime('%a %b %d')} ({label}) · score {item['score']}")
