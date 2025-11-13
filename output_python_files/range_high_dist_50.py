# JUPYTER CELL — feature: range_high_dist_50
FEATURE_CODE = "range_high_dist_50"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Rolling Range High Distance (50)
    Description:
      Relative distance from the current close to the rolling 50-bar highest high.
      Positive values mean close is above the range-high (rare), negative values
      mean close is below the range-high (typical).
    Formula / method (brief):
      RH_t = max(high_{t-49..t})
      dist_t = (close_t - RH_t) / close_t
    Input:
      df: DataFrame with DatetimeIndex (ascending), columns:
           open, high, low, close, volume (case-insensitive)
    Output:
      pd.Series (float), same index as df.index, length == len(df),
      name == FEATURE_CODE. Initial NaNs from rolling windows are OK.
    Constraints:
      - No look-ahead (uses only current and past data).
      - Vectorized (rolling max).
      - Uses only numpy and pandas.
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]  # normalize

    # Rolling 50-bar highest high (includes current bar)
    rh = g["high"].rolling(50, min_periods=50).max().astype(float)

    # Relative distance from range-high
    s = (g["close"].astype(float) - rh) / g["close"].astype(float)
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s