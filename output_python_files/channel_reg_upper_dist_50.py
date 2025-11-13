# JUPYTER CELL — feature: channel_reg_upper_dist_50
FEATURE_CODE = "channel_reg_upper_dist_50"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Rolling Regression Channel — Upper Distance (50)
    Description:
      Relative distance from the current close to the UPPER regression channel line
      built on a 50-bar rolling linear regression of close vs. time. The channel
      uses the regression line ± 1×(residual_std). Here residual_std is approximated
      by sqrt( var_y * (1 - r^2) ), where r is the rolling correlation between time and close.
      (You can change the multiplier if you want wider/narrower channels.)
    Formula / method (brief):
      For each window W=50:
        - t = 0..N-1 (global index as float); y = close
        - slope = cov(t,y)/var(t)
          with cov, var computed via rolling means (no apply/loops).
        - intercept = mean(y) - slope*mean(t)
        - reg_line_t = slope * t + intercept
        - resid_std ≈ sqrt( var_y * (1 - r^2) ), r = cov / sqrt(var_t*var_y)
        - upper = reg_line_t + 1 * resid_std
        - dist = (close - upper) / close
    Input:
      df: DataFrame with DatetimeIndex (ascending), columns:
           open, high, low, close, volume (case-insensitive)
    Output:
      pd.Series (float), same index as df.index, name == FEATURE_CODE.
      Initial NaNs from rolling windows are OK.
    Constraints:
      - No look-ahead.
      - Vectorized with rolling means.
      - Numpy & pandas only.
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    y = g["close"].astype(float)
    n = len(g)
    W = 50

    # time index as float (global positions 0..n-1)
    t = pd.Series(np.arange(n, dtype=float), index=g.index)

    # rolling means
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

    # correlation r and residual std ≈ sqrt(var_y * (1 - r^2))
    r = cov_ty / (np.sqrt(var_t) * np.sqrt(var_y))
    resid_std = np.sqrt(np.clip(var_y * (1.0 - r * r), a_min=0.0, a_max=None))

    # channel upper with multiplier m=1.0 (change if desired)
    m = 1.0
    upper = reg_line + m * resid_std

    s = (y - upper) / y
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s