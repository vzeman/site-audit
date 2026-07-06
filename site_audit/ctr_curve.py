"""Shared organic CTR curve helpers for search-opportunity modeling."""

from __future__ import annotations

CTR_CURVE_V1: list[tuple[int, float]] = [
    (1, 0.28),
    (2, 0.15),
    (3, 0.10),
    (4, 0.07),
    (5, 0.055),
    (6, 0.045),
    (7, 0.038),
    (8, 0.032),
    (9, 0.028),
    (10, 0.025),
    (20, 0.010),
    (30, 0.004),
    (40, 0.0025),
    (50, 0.002),
    (100, 0.002),
]


def expected_ctr(position: float) -> float:
    """Return modeled organic CTR for a ranking position.

    The curve is a documented site-audit v1 approximation based on common
    industry CTR aggregates: steep top-3 drop-off, slower first-page decay,
    then roughly 0.4x per page until a 0.2% floor.
    """
    pos = _clamp_position(position)
    for idx, (left_pos, left_ctr) in enumerate(CTR_CURVE_V1):
        if pos == left_pos:
            return left_ctr
        if pos < left_pos:
            prev_pos, prev_ctr = CTR_CURVE_V1[idx - 1]
            span = left_pos - prev_pos
            ratio = (pos - prev_pos) / span if span else 0.0
            return prev_ctr + (left_ctr - prev_ctr) * ratio
    return CTR_CURVE_V1[-1][1]


def estimate_clicks_gain(impressions: float, current_position: float, target_position: float) -> float:
    """Estimate incremental clicks from moving to ``target_position``."""
    if target_position >= current_position:
        return 0.0
    gain = (expected_ctr(target_position) - expected_ctr(current_position)) * max(0.0, float(impressions or 0.0))
    return max(0.0, gain)


def _clamp_position(position: float) -> float:
    try:
        value = float(position)
    except (TypeError, ValueError):
        value = 100.0
    return max(1.0, min(100.0, value))
