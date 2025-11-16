# JUPYTER CELL — feature: smc_liquidity_void_depth_50
FEATURE_CODE = "smc_liquidity_void_depth_50"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    SMC Liquidity Void Depth (50-bar window)

    Logic:
      A simple quantitative proxy for liquidity void / imbalance depth.

      For each bar:
        - gap_up   = low > high.shift(1)
                     gap_up_depth = low - high.shift(1)
        - gap_down = high < low.shift(1)
                     gap_down_depth = low.shift(1) - high

        - void_depth_bar = max(gap_up_depth, gap_down_depth, 0)

      Then over a 50-bar rolling window:
        - max_void_50 = rolling max(void_depth_bar, 50)
        - range50     = rolling range of high/low over 50 bars
        - feature     = max_void_50 / (range50 + eps)

      Output:
        Float in [0, 1+] indicating relative depth of the largest void
        within the last 50 bars.
    """

    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    high = g["high"].astype(float)
    low  = g["low"].astype(float)

    prev_high = high.shift(1)
    prev_low  = low.shift(1)

    # Gap up void
    gap_up = (low > prev_high)
    gap_up_depth = (low - prev_high).where(gap_up, 0.0)

    # Gap down void
    gap_down = (high < prev_low)
    gap_down_depth = (prev_low - high).where(gap_down, 0.0)

    void_depth_bar = np.maximum(gap_up_depth, gap_down_depth).fillna(0.0)

    win = 50
    max_void_50 = void_depth_bar.rolling(win).max()

    high50 = high.rolling(win).max()
    low50  = low .rolling(win).min()
    range50 = (high50 - low50).replace(0.0, np.nan)

    eps = 1e-9
    depth_norm = max_void_50 / (range50 + eps)
    depth_norm = depth_norm.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return pd.Series(depth_norm.values, index=g.index, name=FEATURE_CODE)