# exposures.py
import numpy as np
import pandas as pd


def adjust_for_soft_exposures(
    pool: pd.DataFrame,
    base_points: pd.Series,
    used_counts: dict[int, int],
    lineup_index: int,
    soft_cap: float,
    strength: float,
) -> np.ndarray:
    """
    Soft exposure system (B):
    - soft_cap: target max exposure per player (0-1).
    - strength: how hard to push away from over-owned players (0-1).

    For lineup k (0-based), exposure_so_far = used / max(1, k).
    If exposure_so_far > soft_cap, we multiply the projection by:
         factor = max(0.1, 1 - strength * (exposure - soft_cap) / max(1e-6, 1-soft_cap))

    Returns adjusted objective vector.
    """
    if lineup_index == 0 or strength <= 0 or soft_cap >= 1.0:
        return base_points.values.astype(float)

    n_prev = float(lineup_index)
    obj = base_points.values.astype(float).copy()

    for i, row in enumerate(pool.itertuples()):
        pid = row.player_id
        prev_used = used_counts.get(pid, 0)
        if prev_used == 0:
            continue

        exposure = prev_used / n_prev  # 0..1
        if exposure <= soft_cap:
            continue

        # How far above cap are we, scaled
        over = exposure - soft_cap
        denom = max(1e-6, 1.0 - soft_cap)
        penalty_ratio = over / denom
        factor = max(0.1, 1.0 - strength * penalty_ratio)
        obj[i] *= factor

    return obj