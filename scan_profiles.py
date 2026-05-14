"""Scan profile configurations for different daily slots.

Profiles:
  premarket  — lightweight: refresh catalyst scores + rescore yesterday's HC picks
               Runs ~06:00 ET (before US market open)
  full       — complete universe scan (current default behaviour)
               Runs ~09:30 ET (after US market open)
  catalyst   — single-ticker rescore triggered by EDGAR 8-K event
               Called from argus-catalyst.yml with CATALYST_TICKER env var
  postclose  — full scan + portfolio monitor + trailing stop check
               Runs ~16:30 ET (after US market close)
"""
import os
from dataclasses import dataclass, field


@dataclass
class ScanProfile:
    name: str
    scan_limit: int | None          # None = full universe
    run_type: str                   # stored in scan history
    send_telegram: bool             # whether to post results to Telegram
    update_memory: bool             # whether to update persistence memory
    run_portfolio_monitor: bool     # whether to check open journal positions
    min_score_override: int | None  # None = use config default
    universe_mode: str              # "core", "rockets", or "combined"
    description: str


PROFILES: dict[str, ScanProfile] = {
    "premarket": ScanProfile(
        name="premarket",
        scan_limit=None,
        run_type="premarket",
        send_telegram=True,
        update_memory=False,          # memory updated by postclose only
        run_portfolio_monitor=False,
        min_score_override=75,        # only alert on strong picks pre-market
        universe_mode="core",         # IWM only for speed
        description="Pre-market light scan — HC picks only",
    ),
    "full": ScanProfile(
        name="full",
        scan_limit=None,
        run_type="scheduled",
        send_telegram=True,
        update_memory=True,
        run_portfolio_monitor=False,
        min_score_override=None,
        universe_mode="combined",
        description="Full universe scan (default)",
    ),
    "catalyst": ScanProfile(
        name="catalyst",
        scan_limit=1,
        run_type="catalyst",
        send_telegram=True,
        update_memory=False,
        run_portfolio_monitor=False,
        min_score_override=70,
        universe_mode="core",
        description="Single-ticker 8-K catalyst rescore",
    ),
    "postclose": ScanProfile(
        name="postclose",
        scan_limit=None,
        run_type="postclose",
        send_telegram=True,
        update_memory=True,
        run_portfolio_monitor=True,
        min_score_override=None,
        universe_mode="combined",
        description="Post-close full scan + portfolio monitor",
    ),
}


def get_profile(name: str) -> ScanProfile:
    name = name.lower().strip()
    if name not in PROFILES:
        raise ValueError(f"Unknown scan profile '{name}'. Valid: {list(PROFILES.keys())}")
    return PROFILES[name]
