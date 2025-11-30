# ============================================================
# app.py — Updated: lineup display simplified, no index columns
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

from merge import merge_players
from sims import add_distribution_columns
from solver import build_lineups
from stacks import StackSettings
from utils import order_lineup


# ============================================================
#  DK TEMPLATE LOADER
# ============================================================

def load_dk_template(uploaded_file):
    raw = uploaded_file.read().decode("utf-8", errors="ignore")

    df = None
    try:
        df = pd.read_csv(StringIO(raw), sep="\t")
    except Exception:
        df = None

    if df is None or df.shape[1] == 1:
        try:
            df = pd.read_csv(StringIO(raw))
        except Exception:
            st.error("❌ Could not parse DK template as TSV or CSV.")
            return None

    if df.shape[1] < 13:
        st.error("❌ DK template has fewer than 13 required columns.")
        return None

    df = df.iloc[:, :13].copy()

    entry_col = df.columns[0]
    df = df[df[entry_col].notna()].copy()
    df = df[df[entry_col].astype(str).str.match(r"^\d+$")]

    df.reset_index(drop=True, inplace=True)
    return df


# ============================================================
#  PRESET HANDLER
# ============================================================

def apply_preset(preset, n_lineups, randomness_level, use_randomness, stack_settings):
    preset = preset.lower()
    ss = stack_settings

    if preset == "cash":
        return min(n_lineups, 50), 0.3, False, ss

    if preset == "single entry":
        return min(n_lineups, 20), max(randomness_level, 0.5), True, ss

    if preset == "150-max mme":
        return max(n_lineups, 75), max(randomness_level, 0.8), True, ss

    return n_lineups, randomness_level, use_randomness, ss


# ============================================================
#  MAIN APP
# ============================================================

def main():
    st.title("NFL DraftKings Optimizer 🏈")

    sal_file = st.file_uploader("Upload DraftKings Salaries CSV", type=["csv"])
    cheat_file = st.file_uploader("Upload DFF Cheatsheet CSV", type=["csv"])
    template_file = st.file_uploader("Upload DK Entry Template (CSV/TSV)", type=["csv", "tsv", "txt"])

    if not sal_file or not cheat_file:
        st.stop()

    salaries = pd.read_csv(sal_file)
    cheat = pd.read_csv(cheat_file)

    # Merge
    with st.spinner("Merging salaries with projections..."):
        merged_pool, unmatched = merge_players(salaries, cheat)

    # Sidebar
    st.sidebar.header("Settings")
    n_lineups = st.sidebar.number_input("Number of lineups", 1, 150, 20)
    min_salary = st.sidebar.number_input("Min salary", 0, 50000, 48000, 500)
    max_salary = st.sidebar.number_input("Max salary", 40000, 50000, 50000, 500)
    min_diff = st.sidebar.slider("Min different players", 0, 9, 2)

    preset = st.sidebar.selectbox("Preset", ["Custom", "Cash", "Single Entry", "150-max MME"])
    randomness_level = st.sidebar.slider("Volatility", 0.0, 1.5, 0.7, 0.05)
    use_randomness = st.sidebar.checkbox("Use randomness", True)

    enable_stacks = st.sidebar.checkbox("Enable QB stacking", True)
    min_wrte = st.sidebar.slider("Min WR/TE w/QB", 0, 4, 2)
    min_opp = st.sidebar.slider("Bring-back players", 0, 3, 1)

    max_team = st.sidebar.number_input("Max per team", 3, 9, 3)
    seed = st.sidebar.number_input("Random seed", 0, 999999, 42)

    stack_settings = StackSettings(
        enable_qb_stacks=enable_stacks,
        min_wrte_with_qb=min_wrte,
        min_opp_rb_wr_te=min_opp,
    )

    if preset != "Custom":
        n_lineups, randomness_level, use_randomness, stack_settings = apply_preset(
            preset, n_lineups, randomness_level, use_randomness, stack_settings
        )

    # ============================================================
    # PLAYER POOL EDITOR – only proj/exclude editable
    # ============================================================

    st.subheader("Player Pool Editor")

    if "exclude" not in merged_pool.columns:
        merged_pool["exclude"] = False

    editor_cols = ["Position", "Name", "TeamAbbrev", "Salary", "ppg_projection", "exclude"]

    edited = st.data_editor(
        merged_pool[editor_cols],
        use_container_width=True,
        column_config={
            "Position": st.column_config.TextColumn(disabled=True),
            "Name": st.column_config.TextColumn(disabled=True),
            "TeamAbbrev": st.column_config.TextColumn(disabled=True),
            "Salary": st.column_config.NumberColumn(disabled=True),
            "ppg_projection": st.column_config.NumberColumn(disabled=False),
            "exclude": st.column_config.CheckboxColumn(disabled=False),
        },
        key="player_editor",
        hide_index=True,
    )

    merged_pool["ppg_projection"] = edited["ppg_projection"]
    merged_pool["exclude"] = edited["exclude"]
    merged_pool = merged_pool[~merged_pool["exclude"]].copy()

    # ============================================================
    # BOOM/BUST TABLE — full and sortable
    # ============================================================

    merged_dist = add_distribution_columns(merged_pool, randomness_level=randomness_level)

    st.subheader("Player Boom/Bust Distribution")
    st.dataframe(
        merged_dist[
            ["Name", "Position", "TeamAbbrev", "Salary", "ppg_projection", "p10", "p50", "p90"]
        ],
        use_container_width=True,
        hide_index=True,
    )

    # ============================================================
    # BUILD LINEUPS
    # ============================================================

    if st.button("Build Lineups"):
        with st.spinner("Optimizing lineups..."):
            lineups = build_lineups(
                merged_dist,
                n_lineups=int(n_lineups),
                min_salary=int(min_salary),
                max_salary=int(max_salary),
                max_players_per_team=int(max_team),
                min_diff_players_between_lineups=int(min_diff),
                stack_settings=stack_settings,
                use_randomness=bool(use_randomness),
                randomness_level=float(randomness_level),
                sim_level=3,
                seed=int(seed),
            )

        if not lineups:
            st.error("❌ No feasible lineups found.")
            return

        st.success(f"Generated **{len(lineups)}** lineups")

        # Convert to ordered format
        ordered_lineups = [order_lineup(lu) for lu in lineups]

        # ============================================================
        # SHOW LINEUPS — simplified columns
        # ============================================================

        st.subheader("Generated Lineups")

        for i, ordered in enumerate(ordered_lineups, 1):
            st.markdown(f"### Lineup {i}")

            show_cols = ["Position", "Name", "TeamAbbrev", "ppg_projection", "Salary"]

            st.dataframe(
                ordered[show_cols].reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )

        # ============================================================
        # EXPOSURE TABLE
        # ============================================================

        st.subheader("Player Exposure (%)")

        all_players = []
        for ordered in ordered_lineups:
            all_players.extend(ordered["Name"].tolist())

        exposure_series = pd.Series(all_players).value_counts() / len(ordered_lineups) * 100
        exposure_series = exposure_series[exposure_series > 0]

        exposure_df = pd.DataFrame({
            "Name": exposure_series.index,
            "Exposure": exposure_series.values
        })

        meta = merged_pool[["Name", "Position", "TeamAbbrev"]]
        exposure_df = exposure_df.merge(meta, on="Name", how="left")
        exposure_df = exposure_df[["Position", "Name", "TeamAbbrev", "Exposure"]]
        exposure_df = exposure_df.sort_values("Exposure", ascending=False)

        st.dataframe(
            exposure_df,
            use_container_width=True,
            hide_index=True,
        )

        # ============================================================
        # DK TEMPLATE EXPORT
        # ============================================================

        if template_file:
            template_df = load_dk_template(template_file)
            if template_df is None or template_df.empty:
                st.error("❌ Template could not be loaded.")
                return

            id_map = dict(zip(salaries["Name"], salaries["ID"]))

            def get_id(name):
                return str(id_map.get(name, ""))

            output = template_df.copy()

            QB, RB1, RB2, WR1, WR2, WR3, TE, FLEX, DST = 4, 5, 6, 7, 8, 9, 10, 11, 12

            rows_to_fill = min(len(output), len(ordered_lineups))

            for i in range(rows_to_fill):
                ordered = ordered_lineups[i]

                def pick(df, pos, k):
                    tmp = df[df["Position"] == pos].reset_index(drop=True)
                    return tmp.loc[k, "Name"] if len(tmp) > k else ""

                flex = ordered[ordered["Slot"] == "FLEX"]["Name"]
                flex = flex.iloc[0] if len(flex) else ""

                output.iat[i, QB] = get_id(pick(ordered, "QB", 0))
                output.iat[i, RB1] = get_id(pick(ordered, "RB", 0))
                output.iat[i, RB2] = get_id(pick(ordered, "RB", 1))
                output.iat[i, WR1] = get_id(pick(ordered, "WR", 0))
                output.iat[i, WR2] = get_id(pick(ordered, "WR", 1))
                output.iat[i, WR3] = get_id(pick(ordered, "WR", 2))
                output.iat[i, TE] = get_id(pick(ordered, "TE", 0))
                output.iat[i, FLEX] = get_id(flex)
                output.iat[i, DST] = get_id(pick(ordered, "DST", 0))

            st.subheader("DK Upload Preview")
            st.dataframe(output.head(rows_to_fill), use_container_width=True, hide_index=True)

            st.download_button(
                "⬇️ Download DK Upload CSV",
                data=output.to_csv(index=False).encode("utf-8"),
                file_name="dk_lineups.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()