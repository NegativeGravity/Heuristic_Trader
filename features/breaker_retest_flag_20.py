# JUPYTER CELL — feature: breaker_retest_flag_20
FEATURE_CODE = "breaker_retest_flag_20"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Breaker Retest Flag (20)
    Approximation:
      Flag == 1 when close is near a 20-bar extreme
      (interpreted as a retest of a prior breaker zone).
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    close = g["close"].astype(float)
    high = g["high"].astype(float)
    low = g["low"].astype(float)

    rh = high.rolling(20, min_periods=20).max()
    rl = low.rolling(20, min_periods=20).min()

    tol = 0.001  # 0.1% tolerance around extremum
    near_high = (np.abs(close - rh) / close.replace(0.0, np.nan)) <= tol
    near_low = (np.abs(close - rl) / close.replace(0.0, np.nan)) <= tol

    flag = (near_high | near_low).astype(int)
    s = pd.Series(flag, index=g.index, name=FEATURE_CODE)
    return s