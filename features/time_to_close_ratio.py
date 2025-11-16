# JUPYTER CELL — feature: time_to_close_ratio
FEATURE_CODE = "time_to_close_ratio"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Time-to-Close Ratio (per calendar day, bar-count based)

    Requirements:
      - Index must be a DatetimeIndex.
      - Data is intraday with multiple bars per calendar day.

    Logic:
      For each calendar day with n bars:

        Bars are ordered: j = 0, 1, ..., n-1

        time_to_close_ratio_j = (n - 1 - j) / max(1, n - 1)

      Interpretation:
        - First bar of the day  → ~1.0
        - Last bar of the day   → 0.0
        - Linearly decreasing in between.

      This is a bar-count based approximation of "how much of the session is left".
    """

    g = df.copy()

    if not isinstance(g.index, pd.DatetimeIndex):
        raise ValueError("time_to_close_ratio requires a DatetimeIndex.")

    dates = g.index.normalize()
    temp = pd.DataFrame({"date": dates})
    ratios = pd.Series(0.0, index=g.index, name=FEATURE_CODE)

    # Group by calendar day
    for date_val, grp_idx in temp.groupby("date").groups.items():
        idx = grp_idx  # index positions for this day
        n = len(idx)
        if n == 1:
            # Only one bar: we consider it as "at close"
            ratios.iloc[idx] = 0.0
            continue

        j = np.arange(n, dtype=float)
        denom = max(1.0, float(n - 1))
        day_ratios = (denom - j) / denom

        ratios.iloc[idx] = day_ratios

    return ratios