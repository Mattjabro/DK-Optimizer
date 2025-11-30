# merge.py
import pandas as pd


def _build_full_name(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["first_name"] = df["first_name"].fillna("").astype(str)
    df["last_name"] = df["last_name"].fillna("").astype(str)
    df["full_name"] = (df["first_name"] + " " + df["last_name"]).str.strip()
    df["full_name_l"] = df["full_name"].str.lower()
    return df


def _add_name_lower(salaries: pd.DataFrame) -> pd.DataFrame:
    salaries = salaries.copy()
    salaries["Name_l"] = salaries["Name"].str.lower()
    return salaries


def _parse_opponent_from_gameinfo(row: pd.Series) -> str | None:
    """
    DK GameInfo usually looks like "DAL@NYG 4:25PM ET".
    We use the first token "DAL@NYG" and infer opponent from TeamAbbrev.
    """
    gi = str(row.get("GameInfo", ""))
    team = str(row.get("TeamAbbrev", "")).strip()
    if not gi or "@" not in gi or not team:
        return None

    first_token = gi.split()[0]
    if "@" not in first_token:
        return None
    away, home = first_token.split("@", 1)
    away = away.strip()
    home = home.strip()

    if team == away:
        return home
    if team == home:
        return away
    # Fallback: if mismatch, just return the "other" one
    return home if team == away else away


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

    cheat = _build_full_name(cheat)
    salaries = _add_name_lower(salaries)

    # Split DST / non-DST
    dst_sal = salaries[salaries["Position"] == "DST"].copy()
    non_dst_sal = salaries[salaries["Position"] != "DST"].copy()

    dst_proj = cheat[cheat["position"] == "DST"].copy()
    non_dst_proj = cheat[cheat["position"] != "DST"].copy()

    # DST merge: TeamAbbrev ↔ team
    dst_merged = dst_sal.merge(
        dst_proj,
        left_on="TeamAbbrev",
        right_on="team",
        how="left",
        suffixes=("", "_proj")
    )

    # Non-DST merge: Name_l + Position + TeamAbbrev ↔ full_name_l + position + team
    non_dst_merged = non_dst_sal.merge(
        non_dst_proj,
        left_on=["Name_l", "Position", "TeamAbbrev"],
        right_on=["full_name_l", "position", "team"],
        how="left",
        suffixes=("", "_proj")
    )

    merged = pd.concat([non_dst_merged, dst_merged], ignore_index=True)

    # Opponent from GameInfo
    merged["Opponent"] = merged.apply(_parse_opponent_from_gameinfo, axis=1)

    # Unmatched
    unmatched = merged[merged["ppg_projection"].isna()][["Name", "Position", "TeamAbbrev", "Salary"]]

    # Restrict to usable players
    merged = merged[merged["ppg_projection"].notna()].copy()

    return merged, unmatched