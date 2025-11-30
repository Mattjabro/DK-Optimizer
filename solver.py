# solver.py
from __future__ import annotations
import numpy as np
import pandas as pd
import pulp  # PuLP solver ONLY

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
    sim_level: int = 3,   # kept for compatibility
    seed: int = 0,
):
    """
    DK NFL optimizer using PuLP (CBC).
    Works on Streamlit Cloud.
    """

    if stack_settings is None:
        stack_settings = StackSettings()

    # Prep pool
    pool = player_pool.copy()
    pool = pool[pool["ppg_projection"].notna()].copy()
    pool = pool.reset_index(drop=False).rename(columns={"index": "player_id"})

    rng = np.random.default_rng(seed)

    lineups = []
    used_sets = []   # store player_id lists for uniqueness

    for ln in range(n_lineups):

        # -------------------------------------
        # Randomized projections (Monte-Carlo)
        # -------------------------------------
        if use_randomness and randomness_level > 0:
            sims = simulate_points(pool, randomness_level=randomness_level, rng=rng)
            pool["sim_points"] = sims
            obj_vec = pool["sim_points"].astype(float).values
        else:
            obj_vec = pool["ppg_projection"].astype(float).values

        # -------------------------------------
        # Create PuLP MILP model
        # -------------------------------------
        model = pulp.LpProblem(f"DK_Lineup_{ln}", pulp.LpMaximize)

        # Binary decision vars
        x = {
            row.player_id: pulp.LpVariable(f"x_{row.player_id}", 0, 1, pulp.LpBinary)
            for row in pool.itertuples()
        }

        # Objective
        model += pulp.lpSum(obj_vec[i] * x[row.player_id]
                            for i, row in enumerate(pool.itertuples()))

        # Salary constraints
        model += pulp.lpSum(row.Salary * x[row.player_id] for row in pool.itertuples()) <= max_salary
        model += pulp.lpSum(row.Salary * x[row.player_id] for row in pool.itertuples()) >= min_salary

        # Roster size = 9
        model += pulp.lpSum(x[row.player_id] for row in pool.itertuples()) == 9

        # Position helper
        def pos_sum(pos):
            return pulp.lpSum(x[row.player_id]
                              for row in pool.itertuples()
                              if row.Position == pos)

        # Exact position requirements
        model += pos_sum("QB") == 1
        model += pos_sum("DST") == 1

        # Minimums
        model += pos_sum("RB") >= 2
        model += pos_sum("WR") >= 3
        model += pos_sum("TE") >= 1

        # FLEX: total RB/WR/TE must be exactly 7
        model += pulp.lpSum(
            x[row.player_id]
            for row in pool.itertuples()
            if row.Position in ("RB", "WR", "TE")
        ) == 7

        # Max per team
        for team in pool["TeamAbbrev"].unique():
            model += pulp.lpSum(
                x[row.player_id]
                for row in pool.itertuples()
                if row.TeamAbbrev == team
            ) <= max_players_per_team

        # Stacking rules
        add_stacking_constraints(model, pool, x, stack_settings)

        # Uniqueness: overlap <= 9 - min_diff
        max_overlap = 9 - min_diff_players_between_lineups
        for prev in used_sets:
            model += pulp.lpSum(x[pid] for pid in prev) <= max_overlap

        # Solve with PuLP’s CBC solver
        status = model.solve(pulp.PULP_CBC_CMD(msg=False))

        if pulp.LpStatus[status] != "Optimal":
            break  # No more feasible solutions

        chosen = [pid for pid, var in x.items() if var.value() == 1]
        used_sets.append(chosen)

        lineup = pool[pool["player_id"].isin(chosen)].copy()
        lineups.append(lineup)

    return lineups