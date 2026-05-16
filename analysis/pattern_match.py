"""Pattern matcher — measures Mahalanobis distance to historical explosive runners.

Curated runner profiles: each is a feature snapshot taken ~30 days before a 3-5x move.
Similarity score: 0–100 (100 = nearest match to runner cluster centroid).

Public API:
  get_runner_similarity(pick: dict) -> float   (0–100)
  get_nearest_runners(pick: dict) -> list[dict]
"""
import logging
import numpy as np

logger = logging.getLogger("Argus.PatternMatch")

# Feature vector for each curated runner at T-30 days before rip.
# Format: [f_score, v_score, m_score, s_score, p_score, c_score, score, float_m, short_pct_x100]
# Note: c_score and float_m may be 0 for older runners (pre-feature additions).
_RUNNER_PROFILES = [
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
]

_FEATURE_KEYS = ["f_score", "v_score", "m_score", "s_score", "p_score", "c_score", "score", "float_m", "short_pct"]

def _pick_to_vector(pick: dict) -> np.ndarray:
    float_m = (pick.get("float_shares", 0) or 0) / 1e6
    short_pct = (pick.get("short_pct", 0) or 0) * 100
    return np.array([
        float(pick.get("f_score", 0) or 0),
        float(pick.get("v_score", 0) or 0),
        float(pick.get("m_score", 0) or 0),
        float(pick.get("s_score", 0) or 0),
        float(pick.get("p_score", 0) or 0),
        float(pick.get("c_score", 0) or 0),
        float(pick.get("score",   0) or 0),
        float(float_m),
        float(short_pct),
    ], dtype=float)


def _runner_to_vector(runner: dict) -> np.ndarray:
    f = runner["features"]
    return np.array([f.get(k, 0) for k in _FEATURE_KEYS], dtype=float)


def _compute_centroid() -> np.ndarray:
    vecs = np.array([_runner_to_vector(r) for r in _RUNNER_PROFILES])
    return vecs.mean(axis=0)


def _compute_std() -> np.ndarray:
    vecs = np.array([_runner_to_vector(r) for r in _RUNNER_PROFILES])
    std = vecs.std(axis=0)
    std[std < 0.1] = 1.0  # avoid division by zero on low-variance features
    return std


_CENTROID = _compute_centroid()
_STD      = _compute_std()


def get_runner_similarity(pick: dict) -> float:
    """Return 0–100 similarity to the historical runner cluster. Higher = more runner-like."""
    try:
        v = _pick_to_vector(pick)
        diff = (v - _CENTROID) / _STD
        distance = float(np.sqrt((diff ** 2).mean()))
        # Convert distance to 0–100 similarity (distance=0 → 100, distance=3 → ~0)
        similarity = max(0.0, 100.0 * (1.0 - distance / 3.0))
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
