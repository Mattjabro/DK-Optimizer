# stacks.py
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import pulp


@dataclass
class StackSettings:
    enable_qb_stacks: bool = True
    min_wrte_with_qb: int = 1           # min WR/TE from same team as QB
    min_opp_rb_wr_te: int = 1          # min RB/WR/TE from opponent team


def add_stacking_constraints(
    model: pulp.LpProblem,
    pool: pd.DataFrame,
    x: dict[int, pulp.LpVariable],
    settings: StackSettings,
):
    """
    NFL stacking rules implemented for PuLP.

    For each QB selected:
      • At least `min_wrte_with_qb` WR/TE from same team
      • At least `min_opp_rb_wr_te` RB/WR/TE from opponent

    All constraints added via PuLP's:
        model += (expression >= RHS)
    """

    if not settings.enable_qb_stacks:
        return

    if settings.min_wrte_with_qb <= 0 and settings.min_opp_rb_wr_te <= 0:
        return

    # Identify QBs
    qbs = pool[pool["Position"] == "QB"].itertuples()

    for qb in qbs:
        qb_pid = qb.player_id
        qb_team = qb.TeamAbbrev

        # If "Opponent" column doesn't exist, skip opponent stacking
        qb_opp = getattr(qb, "Opponent", None)

        # -------------------------------
        # SAME TEAM STACKING (WR/TE)
        # -------------------------------
        if settings.min_wrte_with_qb > 0:
            same_team_wrte = pool[
                (pool["TeamAbbrev"] == qb_team)
                & (pool["Position"].isin(["WR", "TE"]))
            ]

            if len(same_team_wrte) > 0:
                model += (
                    pulp.lpSum(x[row.player_id] for row in same_team_wrte.itertuples())
                    >= settings.min_wrte_with_qb * x[qb_pid]
                )

        # -------------------------------
        # OPPONENT BRING-BACK
        # -------------------------------
        if settings.min_opp_rb_wr_te > 0 and qb_opp is not None:
            opp_skill = pool[
                (pool["TeamAbbrev"] == qb_opp)
                & (pool["Position"].isin(["RB", "WR", "TE"]))
            ]

            if len(opp_skill) > 0:
                model += (
                    pulp.lpSum(x[row.player_id] for row in opp_skill.itertuples())
                    >= settings.min_opp_rb_wr_te * x[qb_pid]
                )