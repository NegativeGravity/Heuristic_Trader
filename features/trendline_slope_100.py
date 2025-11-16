# JUPYTER CELL — feature: trendline_slope_100
FEATURE_CODE = "trendline_slope_100"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Trendline Slope (100)
    Description:
      Rolling OLS slope of close vs. time over a 100-bar window. This is the same
      idea as reg_lin_slope_W, but with W=100; it measures the per-bar trend
      (positive uptrend, negative downtrend).
    Formula / method (brief):
      slope_t = cov(t,y) / var(t), where cov/var computed via rolling means.
    Input:
      df: DataFrame with DatetimeIndex (ascending), columns:
           open, high, low, close, volume (case-insensitive)
    Output:
      pd.Series (float), same index as df.index, name == FEATURE_CODE.
    Constraints:
      - No look-ahead; vectorized with rolling means.
      - Numpy & pandas only.
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    y = g["close"].astype(float)
    n = len(g)
    W = 100

    t = pd.Series(np.arange(n, dtype=float), index=g.index)

    t_mean = t.rolling(W, min_periods=W).mean()
    y_mean = y.rolling(W, min_periods=W).mean()
    cov = (t * y).rolling(W, min_periods=W).mean() - t_mean * y_mean
    var = (t * t).rolling(W, min_periods=W).mean() - t_mean * t_mean

    slope = cov / var.replace(0.0, np.nan)
    slope = slope.astype(float)
    slope.name = FEATURE_CODE
    return slope