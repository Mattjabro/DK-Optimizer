# solver.py
from __future__ import annotations

import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

from stacks import StackSettings, add_stacking_constraints
from sims import simulate_points


def build_lineups(
    player_pool: pd.DataFrame,
    n_lineups: int = 20,
    min_salary: int = 0,
    max_salary: int = 50000,
    max_players_per_team: int = 3,
    min_diff_players_between_lineups: int = 1,
    stack_settings: StackSettings | None = None,
    use_randomness: bool = True,
    randomness_level: float = 0.6,
    sim_level: int = 3,  # kept for API compatibility, not used
    seed: int = 0,
):
    """
    OR-Tools version of your lineup builder.

    Constraints:
      - Salary between min_salary and max_salary
      - Exactly 9 players
      - 1 QB, 1 DST, >=2 RB, >=3 WR, >=1 TE
      - Exactly 7 RB/WR/TE total (2 RB + 3 WR + 1 TE + 1 FLEX)
      - Max players per team
      - Stacking rules via StackSettings
      - Min difference between lineups (min_diff_players_between_lineups)
      - Optional Monte Carlo randomness for projections
    """
    if stack_settings is None:
        stack_settings = StackSettings()

    # Clean pool
    pool = player_pool.copy()
    pool = pool[pool["ppg_projection"].notna()].copy()
    pool = pool.reset_index(drop=False).rename(columns={"index": "player_id"})

    rng = np.random.default_rng(seed)

    lineups: list[pd.DataFrame] = []
    used_set_list: list[list[int]] = []

    for ln in range(n_lineups):
        # -------------------------------------------------
        #  Simulated (or deterministic) objective vector
        # -------------------------------------------------
        if use_randomness and randomness_level > 0:
            sim_vec = simulate_points(pool, randomness_level=randomness_level, rng=rng)
            pool["sim_points"] = sim_vec
            objective_vec = pool["sim_points"].astype(float).values
        else:
            objective_vec = pool["ppg_projection"].astype(float).values

        # -------------------------------------------------
        #  OR-Tools solver setup (CBC)
        # -------------------------------------------------
        solver = pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
        if solver is None:
            raise RuntimeError("Could not create OR-Tools CBC solver.")

        # Decision vars: x[player_id] ∈ {0, 1}
        x: dict[int, pywraplp.Variable] = {}
        for row in pool.itertuples():
            x[row.player_id] = solver.BoolVar(f"x_{row.player_id}")

        # -------------------------------------------------
        #  Objective: maximize total projected/simulated points
        # -------------------------------------------------
        objective = solver.Objective()
        for i, row in enumerate(pool.itertuples()):
            pid = row.player_id
            objective.SetCoefficient(x[pid], float(objective_vec[i]))
        objective.SetMaximization()

        # -------------------------------------------------
        #  Salary constraints
        # -------------------------------------------------
        salary_expr = solver.Sum(row.Salary * x[row.player_id] for row in pool.itertuples())
        solver.Add(salary_expr <= max_salary)
        solver.Add(salary_expr >= min_salary)

        # -------------------------------------------------
        #  Roster size = 9
        # -------------------------------------------------
        solver.Add(solver.Sum(x[row.player_id] for row in pool.itertuples()) == 9)

        # -------------------------------------------------
        #  Position constraints
        # -------------------------------------------------
        def pos_sum(pos: str):
            return solver.Sum(
                x[row.player_id] for row in pool.itertuples() if row.Position == pos
            )

        # Exact
        solver.Add(pos_sum("QB") == 1)
        solver.Add(pos_sum("DST") == 1)

        # Minimums
        solver.Add(pos_sum("RB") >= 2)
        solver.Add(pos_sum("WR") >= 3)
        solver.Add(pos_sum("TE") >= 1)

        # FLEX rule: total RB/WR/TE = 7 (2 RB, 3 WR, 1 TE, 1 FLEX)
        solver.Add(
            solver.Sum(
                x[row.player_id]
                for row in pool.itertuples()
                if row.Position in ("RB", "WR", "TE")
            )
            == 7
        )

        # -------------------------------------------------
        #  Max per team
        # -------------------------------------------------
        for team in pool["TeamAbbrev"].unique():
            solver.Add(
                solver.Sum(
                    x[row.player_id]
                    for row in pool.itertuples()
                    if row.TeamAbbrev == team
                )
                <= max_players_per_team
            )

        # -------------------------------------------------
        #  Stacking constraints
        # -------------------------------------------------
        add_stacking_constraints(solver, pool, x, stack_settings)

        # -------------------------------------------------
        #  Uniqueness between lineups
        # -------------------------------------------------
        max_overlap = 9 - min_diff_players_between_lineups
        for prev_ids in used_set_list:
            solver.Add(
                solver.Sum(x[pid] for pid in prev_ids) <= max_overlap
            )

        # -------------------------------------------------
        #  Solve ILP
        # -------------------------------------------------
        status = solver.Solve()
        if status != pywraplp.Solver.OPTIMAL:
            # No more feasible lineups under current constraints
            break

        chosen_ids = [
            pid for pid, var in x.items() if var.solution_value() > 0.5
        ]
        used_set_list.append(chosen_ids)

        lu = pool[pool["player_id"].isin(chosen_ids)].copy()
        lineups.append(lu)

    return lineups