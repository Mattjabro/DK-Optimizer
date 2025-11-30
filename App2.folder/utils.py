# utils.py
import pandas as pd


def order_lineup(lineup_df: pd.DataFrame) -> pd.DataFrame:
    """
    Order a 9-player DK lineup as:
      QB, RB, RB, WR, WR, WR, TE, FLEX, DST
    """
    df = lineup_df.copy()
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

    flex_candidates = rb_idx_sorted[base_rb:] + wr_idx_sorted[base_wr:] + te_idx_sorted[base_te:]
    if flex_candidates:
        flex_df = df.loc[flex_candidates].sort_values("ppg_projection", ascending=False)
        flex_idx = [flex_df.index[0]]
    else:
        chosen = set(qb_idx + dst_idx + base_rb_idx + base_wr_idx + base_te_idx)
        rem = df[~df.index.isin(chosen)]
        if rem.empty:
            flex_idx = []
        else:
            flex_idx = [rem.sort_values("ppg_projection", ascending=False).index[0]]

    ordered_idx = qb_idx + base_rb_idx + base_wr_idx + base_te_idx + flex_idx + dst_idx

    seen = set()
    final_idx = []
    for idx in ordered_idx:
        if idx not in seen:
            seen.add(idx)
            final_idx.append(idx)

    ordered = df.loc[final_idx].copy()
    ordered.reset_index(drop=False, inplace=True)
    ordered.rename(columns={"index": "orig_index"}, inplace=True)

    slots = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]
    ordered.insert(0, "Slot", slots[: len(ordered)])

    return ordered