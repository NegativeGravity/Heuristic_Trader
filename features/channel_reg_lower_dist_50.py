# JUPYTER CELL — feature: channel_reg_lower_dist_50
FEATURE_CODE = "channel_reg_lower_dist_50"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Rolling Regression Channel — Lower Distance (50)
    Description:
      Relative distance from the current close to the LOWER regression channel line
      built on a 50-bar rolling linear regression of close vs. time. The channel
      uses the regression line ± 1×(residual_std), where residual_std ≈ sqrt(var_y*(1 - r^2)).
    Formula / method (brief):
      Same as the upper version, but:
        lower = reg_line_t - 1 * resid_std
        dist = (close - lower) / close
    Input/Output/Constraints:
      Same as channel_reg_upper_dist_50.
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    y = g["close"].astype(float)
    n = len(g)
    W = 50

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

    r = cov_ty / (np.sqrt(var_t) * np.sqrt(var_y))
    resid_std = np.sqrt(np.clip(var_y * (1.0 - r * r), a_min=0.0, a_max=None))

    m = 1.0
    lower = reg_line - m * resid_std

    s = (y - lower) / y
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s