# sims.py
import numpy as np
import pandas as pd


def add_distribution_columns(df: pd.DataFrame, randomness_level: float = 1.0) -> pd.DataFrame:
    """
    Adds boom/bust distribution info for each player.
    Includes Level 3 adjustments:
        - Recent performance (L5, L10)
        - Player volatility
        - Spread influence (scripted game flow)
        - Over/Under (pace / scoring environment)
        - Implied Team Total (scoring expectation)
    Returns df with:
        dist_sd, p10, p50, p90
    """

    df = df.copy()

    # -----------------------
    # Base mean projection
    # -----------------------
    mean_proj = df["ppg_projection"].astype(float)

    # -----------------------
    # Recent production & volatility
    # -----------------------
    L5 = df.get("L5_fppg_avg", pd.Series(np.nan, index=df.index)).astype(float)
    L10 = df.get("L10_fppg_avg", pd.Series(np.nan, index=df.index)).astype(float)

    mean_recent = (L5.fillna(mean_proj) + L10.fillna(mean_proj)) / 2
    volatility = (L5 - L10).abs().fillna(0.0)

    # SD base (recency + volatility)
    base_sd = 0.35 * mean_recent + 0.30 * volatility
    base_sd = base_sd.clip(lower=1.0)

    sd = base_sd * max(randomness_level, 0.0)

    # -----------------------
    # Level 3 Game Environment
    # -----------------------
    OU = df.get("over_under", pd.Series(45.0, index=df.index)).astype(float)
    ITS = df.get("implied_team_score", pd.Series(21.0, index=df.index)).astype(float)
    spread = df.get("spread", pd.Series(0.0, index=df.index)).astype(float)

    # Over/Under: higher OU → more plays → more fantasy points
    ou_mean_mult = 1.0 + 0.015 * (OU - 45)
    ou_sd_mult   = 1.0 + 0.020 * (OU - 45)

    # Implied team total: increases mean projection
    its_mean_mult = 1.0 + 0.02 * ((ITS - 21) / 7)

    # Game script effect (spread):
    # - Favorite RB/DST benefit when team is leading
    # - Under-dog WR/TE benefit when trailing
    script_mult = np.ones(len(df))

    for idx, (i, row) in enumerate(df.iterrows()):
        pos = row["Position"]
        sp = spread.loc[i] if i in spread.index else 0.0

        # Favorite → RB/DST boost
        if pos == "RB":
            script_mult[idx] *= (1.0 + 0.03 * (-sp))
        if pos == "DST":
            script_mult[idx] *= (1.0 + 0.025 * (-sp))

        # Under-dog → WR/TE forced passing
        if pos in ["WR", "TE"]:
            script_mult[idx] *= (1.0 + 0.02 * (sp))

        # QB mild influence
        if pos == "QB":
            script_mult[idx] *= (1.0 + 0.01 * (sp * 0.5))

    # Final adjusted mean
    enhanced_mean = (
        mean_proj.values *
        ou_mean_mult.values *
        its_mean_mult.values *
        script_mult
    )

    # Final adjusted SD
    enhanced_sd = (sd.values *
                   ou_sd_mult.values *
                   np.sqrt(script_mult))

    # Clip SD to prevent insanity
    enhanced_sd = np.clip(enhanced_sd, 1.0, None)

    # -----------------------
    # Add to DF
    # -----------------------
    df["dist_sd"] = enhanced_sd
    z10, z90 = -1.2816, 1.2816

    df["p10"] = np.clip(enhanced_mean + z10 * enhanced_sd, 0, None)
    df["p50"] = enhanced_mean
    df["p90"] = np.clip(enhanced_mean + z90 * enhanced_sd, 0, None)

    return df


def simulate_points(df: pd.DataFrame, randomness_level: float, rng: np.random.Generator):
    """
    Draws Monte Carlo simulated scores:
        final_score ~ N(p50, dist_sd)
    With clipping at 0.
    """

    # First build the distribution columns
    df2 = add_distribution_columns(df, randomness_level)

    mu = df2["p50"].astype(float).values
    sd = df2["dist_sd"].astype(float).values

    draws = rng.normal(mu, sd)
    return np.clip(draws, 0, None)