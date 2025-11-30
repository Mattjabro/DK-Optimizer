# stacks.py
from __future__ import annotations

from dataclasses import dataclass
from ortools.linear_solver import pywraplp
import pandas as pd


@dataclass
class StackSettings:
    enable_qb_stacks: bool = True
    min_wrte_with_qb: int = 1           # min WR/TE from same team as QB
    min_opp_rb_wr_te: int = 1          # min RB/WR/TE from opponent team


def add_stacking_constraints(
    solver: pywraplp.Solver,
    pool: pd.DataFrame,
    x: dict[int, pywraplp.Variable],
    settings: StackSettings,
):
    """
    Very simple NFL stacking:

      - For each QB selected, require at least
          `min_wrte_with_qb` WR/TE from same team.
      - For each QB selected, require at least
          `min_opp_rb_wr_te` RB/WR/TE from opposing team.

    If `enable_qb_stacks` is False, nothing is added.
    """
    if not settings.enable_qb_stacks:
        return

    if settings.min_wrte_with_qb <= 0 and settings.min_opp_rb_wr_te <= 0:
        return

    # We assume columns: Position, TeamAbbrev, Opponent
    # Opponent is team code of opposing team (e.g. "NYJ").
    qbs = [row for row in pool.itertuples() if row.Position == "QB"]

    for qb in qbs:
        qb_pid = qb.player_id
        qb_team = qb.TeamAbbrev
        qb_opp = getattr(qb, "Opponent", None)

        # Same-team WR/TE
        if settings.min_wrte_with_qb > 0:
            same_team_wrte = [
                row.player_id
                for row in pool.itertuples()
                if row.TeamAbbrev == qb_team and row.Position in ("WR", "TE")
            ]
            if same_team_wrte:
                solver.Add(
                    sum(x[pid] for pid in same_team_wrte) >=
                    settings.min_wrte_with_qb * x[qb_pid]
                )

        # Opponent RB/WR/TE bring-back
        if settings.min_opp_rb_wr_te > 0 and qb_opp is not None:
            opp_skill = [
                row.player_id
                for row in pool.itertuples()
                if row.TeamAbbrev == qb_opp and row.Position in ("RB", "WR", "TE")
            ]
            if opp_skill:
                solver.Add(
                    sum(x[pid] for pid in opp_skill) >=
                    settings.min_opp_rb_wr_te * x[qb_pid]
                )