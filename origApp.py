import streamlit as st
import pandas as pd
import numpy as np
import pulp


# -----------------------------
# Merging salaries + projections
# -----------------------------

def merge_players(salaries: pd.DataFrame, cheat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge DraftKings salaries with DFF cheatsheet projections.

    Rules:
    - Non-DST: merge on lowercased Name + Position + TeamAbbrev
               vs full_name + position + team in cheatsheet.
    - DST: merge on TeamAbbrev (DK) vs team (cheatsheet) where position == "DST".
    Returns:
        merged_player_pool, unmatched_salaries
    """

    salaries = salaries.copy()
    cheat = cheat.copy()

    # Build name helpers in cheatsheet
    cheat["first_name"] = cheat["first_name"].fillna("").astype(str)
    cheat["last_name"] = cheat["last_name"].fillna("").astype(str)
    cheat["full_name"] = (cheat["first_name"] + " " + cheat["last_name"]).str.strip()
    cheat["full_name_l"] = cheat["full_name"].str.lower()

    # Name helper in salaries
    salaries["Name_l"] = salaries["Name"].str.lower()

    # Split into DST / non-DST
    dst_sal = salaries[salaries["Position"] == "DST"].copy()
    non_dst_sal = salaries[salaries["Position"] != "DST"].copy()

    dst_proj = cheat[cheat["position"] == "DST"].copy()
    non_dst_proj = cheat[cheat["position"] != "DST"].copy()

    # --- DST merge: TeamAbbrev (DK) ↔ team (cheatsheet) ---
    dst_merged = dst_sal.merge(
        dst_proj,
        left_on="TeamAbbrev",
        right_on="team",
        how="left",
        suffixes=("", "_proj")
    )

    # --- Non-DST merge: Name + Position + TeamAbbrev ---
    non_dst_merged = non_dst_sal.merge(
        non_dst_proj,
        left_on=["Name_l", "Position", "TeamAbbrev"],
        right_on=["full_name_l", "position", "team"],
        how="left",
        suffixes=("", "_proj")
    )

    merged = pd.concat([non_dst_merged, dst_merged], ignore_index=True)

    # Anything without a projection is unmatched from the perspective of usable player pool
    unmatched = merged[merged["ppg_projection"].isna()][["Name", "Position", "TeamAbbrev", "Salary"]]

    # Filter to only rows with projections
    merged = merged[merged["ppg_projection"].notna()].copy()

    return merged, unmatched


# -----------------------------
# Boom/Bust distributions
# -----------------------------

def add_distribution_columns(df: pd.DataFrame, randomness_level: float = 1.0) -> pd.DataFrame:
    """
    Add distribution parameters (approximate SD + P10/P50/P90) for each player.

    Uses:
    - mean = ppg_projection
    - SD scaled by:
        - recent scoring (average of L5 & L10)
        - volatility (|L5 - L10|)
    randomness_level scales the SD (0 → deterministic, 1 → full).
    """

    df = df.copy()

    mean_proj = df["ppg_projection"].astype(float)

    # Use recent averages where available, fall back to projection
    if {"L5_fppg_avg", "L10_fppg_avg"}.issubset(df.columns):
        mean_recent = df[["L5_fppg_avg", "L10_fppg_avg"]].mean(axis=1, skipna=True)
        mean_recent = mean_recent.fillna(mean_proj)
        volatility = (df["L5_fppg_avg"] - df["L10_fppg_avg"]).abs().fillna(0.0)
    else:
        mean_recent = mean_proj
        volatility = pd.Series(0.0, index=df.index)

    # Heuristic SD: more recent production + more volatility = more spread
    base_sd = 0.35 * mean_recent + 0.30 * volatility
    base_sd = base_sd.replace(0, 1.0)  # never 0 if randomness > 0

    sd = base_sd * max(randomness_level, 0.0)

    df["dist_sd"] = sd

    # Approximate normal quantiles using z-scores
    z10, z90 = -1.2816, 1.2816
    df["p10"] = np.clip(mean_proj + z10 * sd, 0, None)
    df["p50"] = mean_proj  # median ~ mean for normal
    df["p90"] = np.clip(mean_proj + z90 * sd, 0, None)

    return df


def simulate_points(df: pd.DataFrame, randomness_level: float, rng: np.random.Generator) -> np.ndarray:
    """
    Draw one simulated outcome for each player from N(mean, sd) with truncation at 0.
    mean = ppg_projection
    sd   = dist_sd (from add_distribution_columns).
    """
    df = add_distribution_columns(df, randomness_level)
    mu = df["ppg_projection"].astype(float).values
    sd = df["dist_sd"].astype(float).values
    draws = rng.normal(mu, sd)
    return np.clip(draws, 0.0, None)


# -----------------------------
# Lineup ordering: QB, RB, RB, WR, WR, WR, TE, FLEX, DST
# -----------------------------

def order_lineup(lineup_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a 9-player valid DK NFL lineup, return rows ordered as:
    QB, RB, RB, WR, WR, WR, TE, FLEX, DST

    FLEX is whichever extra RB/WR/TE remains after filling:
    - 2 RB
    - 3 WR
    - 1 TE
    """

    df = lineup_df.copy()

    # Sort inside groups by projection for more natural ordering
    df = df.sort_values("ppg_projection", ascending=False)

    qb = df[df["Position"] == "QB"]
    dst = df[df["Position"] == "DST"]
    rb = df[df["Position"] == "RB"]
    wr = df[df["Position"] == "WR"]
    te = df[df["Position"] == "TE"]

    base_rb, base_wr, base_te = 2, 3, 1

    qb_idx = list(qb.index[:1])
    dst_idx = list(dst.index[:1])
    rb_idx_sorted = list(rb.index)
    wr_idx_sorted = list(wr.index)
    te_idx_sorted = list(te.index)

    base_rb_idx = rb_idx_sorted[:base_rb]
    base_wr_idx = wr_idx_sorted[:base_wr]
    base_te_idx = te_idx_sorted[:base_te]

    # Candidates beyond base for FLEX
    flex_candidate_idx = rb_idx_sorted[base_rb:] + wr_idx_sorted[base_wr:] + te_idx_sorted[base_te:]

    if flex_candidate_idx:
        flex_df = df.loc[flex_candidate_idx].sort_values("ppg_projection", ascending=False)
        flex_idx = list(flex_df.index[:1])
    else:
        # Fallback: any non-QB/DST not already chosen
        chosen_so_far = set(qb_idx + dst_idx + base_rb_idx + base_wr_idx + base_te_idx)
        remaining = df[~df.index.isin(chosen_so_far)]
        if not remaining.empty:
            flex_idx = [remaining.sort_values("ppg_projection", ascending=False).index[0]]
        else:
            flex_idx = []

    ordered_idx = qb_idx + base_rb_idx + base_wr_idx + base_te_idx + flex_idx + dst_idx

    # Deduplicate while preserving order
    seen = set()
    final_idx = []
    for idx in ordered_idx:
        if idx not in seen:
            seen.add(idx)
            final_idx.append(idx)

    ordered = df.loc[final_idx].copy()
    ordered.reset_index(drop=False, inplace=True)  # keep original index as "orig_index"
    ordered.rename(columns={"index": "orig_index"}, inplace=True)

    # Attach slot labels
    slots = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]
    ordered.insert(0, "Slot", slots[:len(ordered)])

    return ordered


# -----------------------------
# Optimizer
# -----------------------------

def build_lineups(
    player_pool: pd.DataFrame,
    n_lineups: int = 10,
    salary_cap: int = 50000,
    max_players_per_team: int = 3,
    use_randomness: bool = False,
    randomness_level: float = 0.5,
    seed: int = 0,
):
    """
    Build lineups from a merged player pool using ILP (PuLP).

    DK NFL constraints:
    - 9 players
    - 1 QB
    - 1 DST
    - 2+ RB
    - 3+ WR
    - 1+ TE
    - 7 total non-QB/DST (so 1 FLEX among RB/WR/TE)
    - max_players_per_team
    - salary <= salary_cap

    If use_randomness=True:
        Each lineup uses a new simulated projection vector based on
        ppg_projection + distribution parameters (boom/bust).
    """

    pool = player_pool.copy()
    pool = pool[pool["ppg_projection"].notna()].copy()

    # Give each player a stable id
    pool = pool.reset_index(drop=False).rename(columns={"index": "player_id"})

    rng = np.random.default_rng(seed)
    lineups = []
    used_lineup_sets = []

    for ln in range(n_lineups):
        # Choose which projection to optimize on
        if use_randomness:
            sim_points = simulate_points(pool, randomness_level, rng)
            pool["sim_points"] = sim_points
            objective_points = pool["sim_points"]
        else:
            objective_points = pool["ppg_projection"].astype(float)

        # Setup ILP
        model = pulp.LpProblem(f"DK_NFL_{ln}", pulp.LpMaximize)

        x = {
            row.player_id: pulp.LpVariable(f"x_{row.player_id}", lowBound=0, upBound=1, cat="Binary")
            for row in pool.itertuples()
        }

        # Objective
        model += pulp.lpSum(
            objective_points.iloc[i] * x[row.player_id]
            for i, row in enumerate(pool.itertuples())
        )

        # Salary cap
        model += pulp.lpSum(
            row.Salary * x[row.player_id]
            for row in pool.itertuples()
        ) <= salary_cap

        # Total roster spots
        model += pulp.lpSum(x[row.player_id] for row in pool.itertuples()) == 9

        # Position helpers
        def pos_sum(pos: str):
            return pulp.lpSum(
                x[row.player_id]
                for row in pool.itertuples()
                if row.Position == pos
            )

        # Position constraints
        model += pos_sum("QB") == 1
        model += pos_sum("DST") == 1
        model += pos_sum("RB") >= 2
        model += pos_sum("WR") >= 3
        model += pos_sum("TE") >= 1

        # Non-QB/DST = 7 (2 RB, 3 WR, 1 TE, 1 FLEX)
        model += pulp.lpSum(
            x[row.player_id]
            for row in pool.itertuples()
            if row.Position not in ("QB", "DST")
        ) == 7

        # Max players per team
        for team in pool["TeamAbbrev"].unique():
            model += pulp.lpSum(
                x[row.player_id]
                for row in pool.itertuples()
                if row.TeamAbbrev == team
            ) <= max_players_per_team

        # Uniqueness: prevent identical lineups
        for prev_lineup_ids in used_lineup_sets:
            model += pulp.lpSum(x[pid] for pid in prev_lineup_ids) <= len(prev_lineup_ids) - 1

        # Solve
        status = model.solve(pulp.PULP_CBC_CMD(msg=False))

        if pulp.LpStatus[status] != "Optimal":
            # No more unique optimal solutions
            break

        chosen_ids = [pid for pid, var in x.items() if var.value() == 1]
        used_lineup_sets.append(chosen_ids)

        lineup_df = pool[pool["player_id"].isin(chosen_ids)].copy()
        lineups.append(lineup_df)

    return lineups


# -----------------------------
# Streamlit App
# -----------------------------

def main():
    st.title("NFL DraftKings Lineup Builder 🏈")

    st.write("Upload **DKSalaries.csv** and the **DFF NFL cheatsheet CSV** to build lineups.")

    sal_file = st.file_uploader("Upload DraftKings Salaries CSV", type=["csv"])
    cheat_file = st.file_uploader("Upload DFF Cheatsheet CSV", type=["csv"])

    if not sal_file or not cheat_file:
        st.stop()

    salaries = pd.read_csv(sal_file)
    cheat = pd.read_csv(cheat_file)

    # Sidebar options
    st.sidebar.header("Lineup Settings")

    n_lineups = st.sidebar.number_input("Number of lineups", min_value=1, max_value=150, value=20, step=1)
    salary_cap = st.sidebar.number_input("Salary cap", min_value=40000, max_value=60000, value=50000, step=500)
    max_players_per_team = st.sidebar.number_input("Max players per team", min_value=3, max_value=9, value=3, step=1)

    use_randomness = st.sidebar.checkbox("Use boom/bust random simulation", value=True)
    randomness_level = st.sidebar.slider("Randomness level", 0.0, 1.5, 0.6, 0.1)
    seed = st.sidebar.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)

    # Merge
    with st.spinner("Merging salaries with projections..."):
        merged_pool, unmatched = merge_players(salaries, cheat)

    st.subheader("Player Pool Summary")
    st.write(f"Total players in salaries: **{len(salaries)}**")
    st.write(f"Players successfully matched with projections: **{len(merged_pool)}**")
    st.write(f"Players without projections (not used in optimization): **{len(unmatched)}**")

    if not unmatched.empty:
        with st.expander("Show unmatched players (no projections)"):
            st.dataframe(unmatched)

    # Add distribution columns for info (even if not using randomness)
    merged_pool_dist = add_distribution_columns(merged_pool, randomness_level=1.0)
    with st.expander("Sample of merged player pool with boom/bust stats"):
        st.dataframe(
            merged_pool_dist[[
                "Name", "Position", "TeamAbbrev", "Salary",
                "ppg_projection", "p10", "p50", "p90"
            ]].head(30)
        )

    if st.button("Build Lineups"):
        with st.spinner("Optimizing lineups..."):
            lineups = build_lineups(
                merged_pool_dist,
                n_lineups=n_lineups,
                salary_cap=salary_cap,
                max_players_per_team=max_players_per_team,
                use_randomness=use_randomness,
                randomness_level=randomness_level,
                seed=seed,
            )

        if not lineups:
            st.error("No feasible lineups found. Try relaxing constraints or lowering randomness.")
            return

        st.subheader("Generated Lineups")

        for i, lu in enumerate(lineups, start=1):
            ordered = order_lineup(lu)

            total_salary = int(ordered["Salary"].sum())
            total_proj = float(ordered["ppg_projection"].sum())

            st.markdown(f"### Lineup {i}")
            st.write(f"**Total Salary:** {total_salary}  |  **Total Projection (mean):** {total_proj:.2f}")

            show_cols = [
                "Slot",
                "Position",
                "Name",
                "TeamAbbrev",
                "Salary",
                "ppg_projection",
                "p10",
                "p90",
            ]
            available_cols = [c for c in show_cols if c in ordered.columns]
            st.dataframe(ordered[available_cols].reset_index(drop=True))

            st.markdown("---")


if __name__ == "__main__":
    main()