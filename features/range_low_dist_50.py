# JUPYTER CELL — feature: range_low_dist_50
FEATURE_CODE = "range_low_dist_50"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Rolling Range Low Distance (50)
    Description:
      Relative distance from the current close to the rolling 50-bar lowest low.
      Positive values mean close is above the range-low (typical), negative values
      mean close is below the range-low (rare).
    Formula / method (brief):
      RL_t = min(low_{t-49..t})
      dist_t = (close_t - RL_t) / close_t
    Input:
      df: DataFrame with DatetimeIndex (ascending), columns:
           open, high, low, close, volume (case-insensitive)
    Output:
      pd.Series (float), same index as df.index, length == len(df),
      name == FEATURE_CODE. Initial NaNs from rolling windows are OK.
    Constraints:
      - No look-ahead (uses only current and past data).
      - Vectorized (rolling min).
      - Uses only numpy and pandas.
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]  # normalize

    # Rolling 50-bar lowest low (includes current bar)
    rl = g["low"].rolling(50, min_periods=50).min().astype(float)

    # Relative distance from range-low
    s = (g["close"].astype(float) - rl) / g["close"].astype(float)
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s