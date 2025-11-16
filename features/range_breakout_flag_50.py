# JUPYTER CELL — feature: range_breakout_flag_50
FEATURE_CODE = "range_breakout_flag_50"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Rolling Range Breakout Flag (50)
    Description:
      Discrete flag indicating breakout vs. the previous 50 bars (excluding current):
        +1 if close_t > max(high_{t-50..t-1})  (up-breakout)
        -1 if close_t < min(low_{t-50..t-1})   (down-breakout)
         0 otherwise (inside the prior 50-bar range)
    Formula / method (brief):
      prev_high_t = max(high_{t-50..t-1}) = rolling_max(high, 50) on shifted series
      prev_low_t  = min(low_{t-50..t-1})  = rolling_min(low, 50) on shifted series
      flag_t = 1 if close_t > prev_high_t; -1 if close_t < prev_low_t; else 0
    Input:
      df: DataFrame with DatetimeIndex (ascending), columns:
           open, high, low, close, volume (case-insensitive)
    Output:
      pd.Series (int), same index as df.index, length == len(df),
      name == FEATURE_CODE. Initial NaNs from rolling windows are OK (will map to 0).
    Constraints:
      - No look-ahead (uses only current and past data; window excludes current via shift).
      - Vectorized (rolling + numpy where).
      - Uses only numpy and pandas.
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]  # normalize

    # Prior window (exclude current) via shift(1)
    prev_high = g["high"].shift(1).rolling(50, min_periods=50).max().astype(float)
    prev_low  = g["low"].shift(1).rolling(50, min_periods=50).min().astype(float)

    c = g["close"].astype(float)

    up_break   = c > prev_high
    down_break = c < prev_low

    # Map to {-1, 0, +1}; NaNs in prev_* yield False in comparisons -> 0
    flag = np.where(up_break, 1, np.where(down_break, -1, 0)).astype(int)

    s = pd.Series(flag, index=g.index, name=FEATURE_CODE)
    return s