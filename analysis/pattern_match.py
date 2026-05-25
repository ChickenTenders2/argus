"""Pattern matcher — nearest-neighbour distance to historical explosive runners.

Curated runner profiles span two phases:
  - Post-momentum (T-30 before confirmed rip): stocks already showing momentum
  - Pre-explosion (T-30 before initial breakout): stocks in quiet accumulation

Using nearest-neighbour (minimum distance to any profile) rather than centroid
distance so the bimodal distribution is handled correctly — a pre-explosion stock
is compared to pre-explosion profiles, not a centroid between both phases.

Similarity score: 0–100 (100 = identical to nearest historical runner).

Public API:
  get_runner_similarity(pick: dict) -> float   (0–100)
  get_nearest_runners(pick: dict) -> list[dict]
"""
import logging
import numpy as np

logger = logging.getLogger("Argus.PatternMatch")

# Feature vector for each curated runner.
# Post-momentum profiles: T-30 snapshot when stock already had momentum (score 65-78).
# Pre-explosion profiles: T-30 snapshot during quiet accumulation before initial breakout.
_RUNNER_PROFILES = [
    # ── Post-momentum profiles (T-30 before confirmed 3-5x rip) ──────────────
    {
        "name": "Ondas Holdings (ONDS) — 2024 pre-run",
        "ticker": "ONDS",
        "features": {"f_score": 3, "v_score": 2, "m_score": 18, "s_score": 12, "p_score": 4,
                     "c_score": 0, "score": 65, "float_m": 8.5, "short_pct": 12},
    },
    {
        "name": "SoundHound AI (SOUN) — 2024 pre-run",
        "ticker": "SOUN",
        "features": {"f_score": 2, "v_score": 0, "m_score": 22, "s_score": 10, "p_score": 3,
                     "c_score": 0, "score": 68, "float_m": 200, "short_pct": 18},
    },
    {
        "name": "Rocket Lab (RKLB) — 2024 breakout",
        "ticker": "RKLB",
        "features": {"f_score": 4, "v_score": 2, "m_score": 20, "s_score": 14, "p_score": 5,
                     "c_score": 0, "score": 72, "float_m": 430, "short_pct": 10},
    },
    {
        "name": "AST SpaceMobile (ASTS) — 2024 run",
        "ticker": "ASTS",
        "features": {"f_score": 2, "v_score": 0, "m_score": 25, "s_score": 8, "p_score": 3,
                     "c_score": 5, "score": 70, "float_m": 50, "short_pct": 22},
    },
    {
        "name": "IonQ (IONQ) — 2024 quantum surge",
        "ticker": "IONQ",
        "features": {"f_score": 2, "v_score": 1, "m_score": 24, "s_score": 9, "p_score": 4,
                     "c_score": 3, "score": 69, "float_m": 180, "short_pct": 14},
    },
    {
        "name": "Hims & Hers (HIMS) — 2024 weight-loss run",
        "ticker": "HIMS",
        "features": {"f_score": 8, "v_score": 2, "m_score": 20, "s_score": 15, "p_score": 5,
                     "c_score": 2, "score": 74, "float_m": 150, "short_pct": 16},
    },
    {
        "name": "Dave Inc (DAVE) — 2024 fintech run",
        "ticker": "DAVE",
        "features": {"f_score": 6, "v_score": 3, "m_score": 19, "s_score": 11, "p_score": 4,
                     "c_score": 1, "score": 71, "float_m": 12, "short_pct": 25},
    },
    {
        "name": "Credo Technology (CRDO) — 2024 AI-infra run",
        "ticker": "CRDO",
        "features": {"f_score": 10, "v_score": 3, "m_score": 22, "s_score": 16, "p_score": 5,
                     "c_score": 2, "score": 78, "float_m": 120, "short_pct": 8},
    },
    # ── Pre-explosion profiles (T-30 before initial breakout from accumulation) ──
    {
        "name": "Ondas Holdings (ONDS) — 2021 pre-explosion",
        "ticker": "ONDS_pre",
        "features": {"f_score": 1, "v_score": 0, "m_score": 4, "s_score": 14, "p_score": 0,
                     "c_score": 2, "score": 21, "float_m": 8.0, "short_pct": 15},
    },
    {
        "name": "SoundHound AI (SOUN) — 2022 pre-explosion",
        "ticker": "SOUN_pre",
        "features": {"f_score": 0, "v_score": 0, "m_score": 7, "s_score": 12, "p_score": 0,
                     "c_score": 1, "score": 20, "float_m": 20.0, "short_pct": 20},
    },
    {
        "name": "Rocket Lab (RKLB) — 2023 pre-explosion",
        "ticker": "RKLB_pre",
        "features": {"f_score": 2, "v_score": 0, "m_score": 9, "s_score": 10, "p_score": 0,
                     "c_score": 1, "score": 22, "float_m": 50.0, "short_pct": 12},
    },
    {
        "name": "AST SpaceMobile (ASTS) — 2023 pre-explosion",
        "ticker": "ASTS_pre",
        "features": {"f_score": 0, "v_score": 0, "m_score": 5, "s_score": 9, "p_score": 0,
                     "c_score": 3, "score": 17, "float_m": 35.0, "short_pct": 18},
    },
    {
        "name": "IonQ (IONQ) — 2023 pre-explosion",
        "ticker": "IONQ_pre",
        "features": {"f_score": 1, "v_score": 0, "m_score": 8, "s_score": 10, "p_score": 0,
                     "c_score": 2, "score": 21, "float_m": 80.0, "short_pct": 14},
    },
    {
        "name": "Hims & Hers (HIMS) — 2023 pre-explosion",
        "ticker": "HIMS_pre",
        "features": {"f_score": 3, "v_score": 1, "m_score": 6, "s_score": 12, "p_score": 0,
                     "c_score": 1, "score": 23, "float_m": 60.0, "short_pct": 22},
    },
]

# Only use the 5 intrinsic score dimensions: p_score excluded (scan-history artifact
# that new picks always lack) and float/short excluded (unreliable from yfinance).
_FEATURE_KEYS = ["f_score", "v_score", "m_score", "s_score", "c_score"]


def _pick_to_vector(pick: dict) -> np.ndarray:
    return np.array([float(pick.get(k, 0) or 0) for k in _FEATURE_KEYS], dtype=float)


def _runner_to_vector(runner: dict) -> np.ndarray:
    f = runner["features"]
    return np.array([float(f.get(k, 0)) for k in _FEATURE_KEYS], dtype=float)


def _compute_std() -> np.ndarray:
    vecs = np.array([_runner_to_vector(r) for r in _RUNNER_PROFILES])
    std = vecs.std(axis=0)
    std[std < 0.1] = 1.0  # avoid division by zero on low-variance features
    return std


_STD = _compute_std()


def get_runner_similarity(pick: dict) -> float:
    """Return 0–100 similarity to the nearest historical runner profile.

    Uses nearest-neighbour distance (minimum over all profiles) rather than
    centroid distance so pre-explosion and post-momentum profiles are treated
    as separate archetypes — a quiet accumulation stock matches pre-explosion
    profiles without being penalised for being far from the post-momentum centroid.
    """
    try:
        v = _pick_to_vector(pick)
        min_dist = float("inf")
        for runner in _RUNNER_PROFILES:
            rv = _runner_to_vector(runner)
            diff = (v - rv) / _STD
            dist = float(np.sqrt((diff ** 2).mean()))
            if dist < min_dist:
                min_dist = dist
        similarity = max(0.0, 100.0 * (1.0 - min_dist / 3.0))
        return round(similarity, 1)
    except Exception as e:
        logger.debug(f"get_runner_similarity failed: {e}")
        return 0.0


def get_nearest_runners(pick: dict, top_n: int = 3) -> list:
    """Return the top_n most similar historical runners to this pick."""
    try:
        v = _pick_to_vector(pick)
        distances = []
        for runner in _RUNNER_PROFILES:
            rv = _runner_to_vector(runner)
            diff = (v - rv) / _STD
            dist = float(np.sqrt((diff ** 2).mean()))
            similarity = max(0.0, round(100.0 * (1.0 - dist / 3.0), 1))
            distances.append({
                "name":       runner["name"],
                "ticker":     runner["ticker"],
                "similarity": similarity,
                "distance":   round(dist, 2),
            })
        distances.sort(key=lambda x: x["similarity"], reverse=True)
        return distances[:top_n]
    except Exception as e:
        logger.debug(f"get_nearest_runners failed: {e}")
        return []
