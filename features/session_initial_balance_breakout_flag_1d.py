# JUPYTER CELL — feature: session_initial_balance_breakout_flag_1d
FEATURE_CODE = "session_initial_balance_breakout_flag_1d"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Session Initial Balance Breakout Flag (1-day)

    Requirements:
      - Index must be a DatetimeIndex.
      - Data is assumed intraday with multiple bars per calendar day.

    Logic (bar-count based Initial Balance):
      For each calendar day:
        - Let n = number of bars in that day.
        - Initial Balance (IB) = first max(int(0.25 * n), 1) bars.
        - IB_high = max(high over IB bars)
        - IB_low  = min(low  over IB bars)

      For each bar in that day:
        - If bar is after the IB segment AND
             (high > IB_high OR low < IB_low)
          then breakout_flag = 1
          else 0.

      Output:
        Per-bar integer flag in {0, 1}.
    """

    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    if not isinstance(g.index, pd.DatetimeIndex):
        raise ValueError("session_initial_balance_breakout_flag_1d requires a DatetimeIndex.")

    high = g["high"].astype(float)
    low  = g["low"].astype(float)

    dates = g.index.normalize()

    temp = pd.DataFrame({"high": high, "low": low, "date": dates})
    grouped = temp.groupby("date", sort=False)

    breakout_flag = pd.Series(0, index=g.index, dtype=int)

    for date_val, grp in grouped:
        n = len(grp)
        if n <= 1:
            continue

        # IB defined as first 25% of bars in that day
        ib_count = max(int(round(0.25 * n)), 1)

        ib_slice = grp.iloc[:ib_count]
        ib_high = ib_slice["high"].max()
        ib_low  = ib_slice["low"].min()

        # Bars after IB
        after_ib = grp.iloc[ib_count:]

        cond_break = (after_ib["high"] > ib_high) | (after_ib["low"] < ib_low)
        breakout_flag.loc[after_ib.index] = cond_break.astype(int)

    return breakout_flag.rename(FEATURE_CODE)