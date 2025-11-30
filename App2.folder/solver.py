# solver.py
from __future__ import annotations
import numpy as np
import pandas as pd
import pulp

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
    sim_level: int = 3,
    seed: int = 0,
):
    if stack_settings is None:
        stack_settings = StackSettings()

    pool = player_pool.copy()
    pool = pool[pool["ppg_projection"].notna()].copy()
    pool = pool.reset_index(drop=False).rename(columns={"index": "player_id"})

    rng = np.random.default_rng(seed)
    lineups = []
    used_set_list = []

    for ln in range(n_lineups):

        # Simulate projections
        if use_randomness and randomness_level > 0:
            sim_vec = simulate_points(pool, randomness_level=randomness_level, rng=rng)
            pool["sim_points"] = sim_vec
            objective_vec = pool["sim_points"].astype(float).values
        else:
            objective_vec = pool["ppg_projection"].astype(float).values

        # ILP model
        model = pulp.LpProblem(f"DK_LU_{ln}", pulp.LpMaximize)

        x = {
            row.player_id: pulp.LpVariable(f"x_{row.player_id}", 0, 1, pulp.LpBinary)
            for row in pool.itertuples()
        }

        # Objective
        model += pulp.lpSum(objective_vec[i] * x[row.player_id]
                            for i, row in enumerate(pool.itertuples()))

        # Salary range
        model += pulp.lpSum(row.Salary * x[row.player_id]
                            for row in pool.itertuples()) <= max_salary
        model += pulp.lpSum(row.Salary * x[row.player_id]
                            for row in pool.itertuples()) >= min_salary

        # Total roster = 9
        model += pulp.lpSum(x[row.player_id] for row in pool.itertuples()) == 9

        # DK roster constraints
        def pos_sum(pos):
            return pulp.lpSum(x[row.player_id]
                              for row in pool.itertuples()
                              if row.Position == pos)

        model += pos_sum("QB") == 1
        model += pos_sum("DST") == 1
        model += pos_sum("RB") >= 2
        model += pos_sum("WR") >= 3
        model += pos_sum("TE") >= 1

        # Flex count rule (RB/WR/TE = 7)
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

        # Stacking
        add_stacking_constraints(model, pool, x, stack_settings)

        # Uniqueness
        for prev in used_set_list:
            model += pulp.lpSum(x[pid] for pid in prev) <= 9 - min_diff_players_between_lineups

        # Solve
        status = model.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[status] != "Optimal":
            break

        chosen_ids = [pid for pid, var in x.items() if var.value() == 1]
        used_set_list.append(chosen_ids)

        lu = pool[pool["player_id"].isin(chosen_ids)].copy()
        lineups.append(lu)

    return lineups