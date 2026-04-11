import streamlit as st

def apply_preset():
    preset = st.session_state.preset_selector
    if preset == "Aggressive Growth":
        st.session_state.min_score_val = 70
        st.session_state.horizon_val = 42
        st.session_state.target_return_val = 20
        st.session_state.risk_val = 1.0
    elif preset == "Bear Market Defense":
        st.session_state.min_score_val = 80
        st.session_state.horizon_val = 84
        st.session_state.target_return_val = 5
        st.session_state.risk_val = 0.25
    elif preset == "Default":
        st.session_state.min_score_val = 65
        st.session_state.horizon_val = 63
        st.session_state.target_return_val = 10
        st.session_state.risk_val = 0.75

if "min_score_val" not in st.session_state:
    st.session_state.min_score_val = 65
    st.session_state.horizon_val = 63
    st.session_state.target_return_val = 10
    st.session_state.risk_val = 0.75

with st.sidebar:
    st.header("Global Presets")
    st.selectbox("Select Preset", ["Default", "Aggressive Growth", "Bear Market Defense"], key="preset_selector", on_change=apply_preset)

    st.header("Quick Scan Settings")
    st.slider("Minimum Score", 50, 95, key="min_score_val")
    st.selectbox("Horizon days", [42, 63, 84], key="horizon_val")

