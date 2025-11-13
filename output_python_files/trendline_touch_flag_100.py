# JUPYTER CELL — feature: trendline_touch_flag_100
FEATURE_CODE = "trendline_touch_flag_100"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Trendline Touch Flag (100)
    Description:
      Binary flag indicating whether the current close is "touching" the rolling
      regression trendline (close vs. time) over a 100-bar window, within a tolerance
      proportional to the window's residual standard deviation.
    Formula / method (brief):
      For each window W=100:
        - Compute slope/intercept of OLS(y~t); reg_line_t = slope*t + intercept
        - resid = close - reg_line_t
        - resid_std ≈ sqrt(var_y * (1 - r^2))
        - flag = 1 if |resid| <= tol_mult * resid_std  else 0
      Here tol_mult = 0.25 (change if you prefer).
    Input:
      df: DataFrame with DatetimeIndex (ascending), columns:
           open, high, low, close, volume (case-insensitive)
    Output:
      pd.Series (int), values in {0,1}, same index as df.index, name == FEATURE_CODE.
    Constraints:
      - No look-ahead; vectorized with rolling means.
      - Numpy & pandas only.
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    y = g["close"].astype(float)
    n = len(g)
    W = 100
    tol_mult = 0.25

    t = pd.Series(np.arange(n, dtype=float), index=g.index)

    t_mean = t.rolling(W, min_periods=W).mean()
    y_mean = y.rolling(W, min_periods=W).mean()
    ty_mean = (t * y).rolling(W, min_periods=W).mean()
    t2_mean = (t * t).rolling(W, min_periods=W).mean()
    y2_mean = (y * y).rolling(W, min_periods=W).mean()

    cov_ty = ty_mean - t_mean * y_mean
    var_t  = t2_mean - t_mean * t_mean
    var_y  = y2_mean - y_mean * y_mean

    slope = cov_ty / var_t.replace(0.0, np.nan)
    intercept = y_mean - slope * t_mean
    reg_line = slope * t + intercept

    # residual & residual std
    resid = y - reg_line
    r = cov_ty / (np.sqrt(var_t) * np.sqrt(var_y))
    resid_std = np.sqrt(np.clip(var_y * (1.0 - r * r), a_min=0.0, a_max=None))

    flag = (resid.abs() <= (tol_mult * resid_std)).astype(int)
    flag.name = FEATURE_CODE
    return flag