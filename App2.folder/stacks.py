# stacks.py
from dataclasses import dataclass
import pulp
import pandas as pd


@dataclass
class StackSettings:
    enable_qb_stacks: bool = True
    min_wrte_with_qb: int = 1          # A: same-team WR/TE with QB
    min_opp_rb_wr_te: int = 1         # A: bring-back from opponent


def add_stacking_constraints(
    model: pulp.LpProblem,
    pool: pd.DataFrame,
    x_vars: dict[int, pulp.LpVariable],
    settings: StackSettings,
) -> None:
    """
    Add QB stacking + bring-back constraints to the ILP model.
    - For each QB chosen:
        sum WR/TE same team >= min_wrte_with_qb * x_qb
        sum RB/WR/TE opp team >= min_opp_rb_wr_te * x_qb
    """
    if not settings.enable_qb_stacks:
        return

    qbs = pool[pool["Position"] == "QB"]

    for qb_row in qbs.itertuples():
        qb_id = qb_row.player_id
        qb_team = qb_row.TeamAbbrev
        qb_opp = getattr(qb_row, "Opponent", None)

        # Same-team WR/TE
        same_team_wrte = pool[
            (pool["TeamAbbrev"] == qb_team)
            & (pool["Position"].isin(["WR", "TE"]))
        ]
        if settings.min_wrte_with_qb > 0 and not same_team_wrte.empty:
            model += (
                pulp.lpSum(
                    x_vars[row.player_id] for row in same_team_wrte.itertuples()
                )
                >= settings.min_wrte_with_qb * x_vars[qb_id]
            )

        # Bring-back from opponent
        if qb_opp and settings.min_opp_rb_wr_te > 0:
            opp_pool = pool[
                (pool["TeamAbbrev"] == qb_opp)
                & (pool["Position"].isin(["RB", "WR", "TE"]))
            ]
            if not opp_pool.empty:
                model += (
                    pulp.lpSum(
                        x_vars[row.player_id] for row in opp_pool.itertuples()
                    )
                    >= settings.min_opp_rb_wr_te * x_vars[qb_id]
                )