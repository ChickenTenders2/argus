"""Design tokens and CSS for the Argus modern fintech dashboard."""

# ── Colour palette ────────────────────────────────────────────────────────────
BG_PRIMARY   = "#0b1220"
BG_PANEL     = "#0f1a2e"
BG_CARD      = "#111e35"
ACCENT_CYAN  = "#00d4ff"
ACCENT_GREEN = "#26d97f"
ACCENT_WARN  = "#ffb547"
ACCENT_DANGER= "#ff5a7a"
TEXT_PRIMARY = "#e8eef7"
TEXT_MUTED   = "#8ea0ba"
BORDER_DIM   = "rgba(255,255,255,0.07)"

# Regime colours
REGIME_COLORS = {
    "Bull":         "#26d97f",
    "Neutral":      "#00d4ff",
    "Bear":         "#ffb547",
    "Extreme Fear": "#ff5a7a",
    "Stagflation":  "#bf5af2",
}

# Sector colours (Streamlit named colours)
SECTOR_COLORS = {
    "Technology":            "blue",
    "Healthcare":            "green",
    "Financial Services":    "violet",
    "Consumer Cyclical":     "orange",
    "Industrials":           "gray",
    "Communication Services":"blue",
    "Real Estate":           "red",
    "Consumer Defensive":    "green",
    "Energy":                "orange",
    "Basic Materials":       "gray",
    "Utilities":             "violet",
}

ENHANCED_CSS = """
<style>
/* ── Hide default Streamlit chrome ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* ── Import tabular-nums font feature ── */
* { font-variant-numeric: tabular-nums; }

/* ── Argus metric badges ── */
.argus-badge-wrap { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; }
.argus-badge {
    display: inline-block; position: relative;
    background: #1a2e4a; color: #c9d6e8;
    padding: 3px 10px; border-radius: 12px;
    font-size: 0.78rem; cursor: default; white-space: nowrap;
    border: 1px solid rgba(0,212,255,0.12);
}
.argus-badge:hover { background: #1e3f68; border-color: rgba(0,212,255,0.3); }
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

/* ── Catalyst pill ── */
.argus-catalyst-pill {
    display: inline-block;
    background: rgba(38,217,127,0.12);
    color: #26d97f;
    padding: 2px 8px; border-radius: 10px;
    font-size: 0.72rem; white-space: nowrap;
    border: 1px solid rgba(38,217,127,0.25);
    margin: 2px 1px;
}

/* ── Score velocity chip ── */
.argus-velocity-up   { color: #26d97f; font-size: 0.72rem; font-weight: 700; }
.argus-velocity-flat { color: #8ea0ba; font-size: 0.72rem; }
.argus-velocity-down { color: #ff5a7a; font-size: 0.72rem; font-weight: 700; }

/* ── Hero card ── */
.argus-hero-card {
    background: linear-gradient(135deg, #0f1a2e 0%, #111e35 100%);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.argus-hero-card:hover {
    border-color: rgba(0,212,255,0.5);
    box-shadow: 0 0 16px rgba(0,212,255,0.08);
}
.argus-hero-score {
    font-size: 2.2rem;
    font-weight: 800;
    color: #00d4ff;
    letter-spacing: -0.02em;
    line-height: 1;
}
.argus-hero-ticker {
    font-size: 1.3rem;
    font-weight: 700;
    color: #e8eef7;
}
.argus-hero-signal {
    font-size: 0.82rem;
    font-weight: 600;
    color: #26d97f;
    margin-top: 2px;
}
.argus-rr-chip {
    display: inline-block;
    background: rgba(255,181,71,0.12);
    color: #ffb547;
    padding: 2px 8px; border-radius: 10px;
    font-size: 0.72rem;
    border: 1px solid rgba(255,181,71,0.25);
}

/* ── Regime strip (enhanced) ── */
.argus-regime-strip {
    border-radius: 8px;
    padding: 8px 14px;
    margin-bottom: 12px;
    font-size: 0.88rem;
    border-left-width: 4px;
    border-left-style: solid;
}
.argus-regime-strip-label { font-weight: 700; font-size: 1rem; }
.argus-regime-chip {
    display: inline-block;
    background: rgba(255,255,255,0.08);
    padding: 2px 8px; border-radius: 10px;
    font-size: 0.72rem; margin: 0 3px;
    white-space: nowrap;
}

/* ── Sector heatmap cell ── */
.argus-hm-cell {
    display: flex; align-items: center; justify-content: center;
    border-radius: 6px; font-size: 0.75rem; font-weight: 700;
    height: 36px; cursor: default;
}

/* ── Metric hierarchy ── */
[data-testid="stMetricValue"] {
    font-size: 1.25rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.66rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    opacity: 0.55 !important;
}
[data-testid="stMetricDelta"] { font-size: 0.74rem !important; }

/* ── Border container hover glow ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(0,212,255,0.35) !important;
    box-shadow: 0 0 0 1px rgba(0,212,255,0.10);
}

/* ── Expander header weight ── */
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span { font-weight: 600 !important; }

/* ── Sidebar nav radio ── */
[data-testid="stSidebar"] [role="radiogroup"] label {
    border-radius: 6px;
    transition: background 0.15s;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.05);
}

/* ── Dividers ── */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* ── Regime info tooltip ── */
.argus-regime-tip {
    position: relative; display: inline-block;
    cursor: help; opacity: 0.6; font-size: 0.85rem;
    margin-left: 6px; vertical-align: middle;
}
.argus-regime-tip .argus-regime-tip-box {
    visibility: hidden; opacity: 0;
    background: #0d1b2e; color: #c9d6e8;
    text-align: left; border-radius: 7px;
    padding: 9px 13px; position: absolute;
    z-index: 9999; bottom: 140%; left: 50%;
    transform: translateX(-50%);
    width: 310px; font-size: 0.72rem; line-height: 1.55;
    border: 1px solid #2a4a6a;
    box-shadow: 0 3px 12px rgba(0,0,0,0.5);
    white-space: normal; pointer-events: none;
    transition: opacity 0.15s ease;
}
.argus-regime-tip:hover .argus-regime-tip-box { visibility: visible; opacity: 1; }
</style>
"""
