# JUPYTER CELL — feature: trend_sma_cross_flag_5_20
FEATURE_CODE = "trend_sma_cross_flag_5_20"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    SMA Crossover Flag (5 vs 20)
    Description:
      Discrete flag for moving-average crossovers:
        +1 when SMA(5) crosses ABOVE SMA(20) at current bar,
        -1 when SMA(5) crosses BELOW SMA(20) at current bar,
         0 otherwise.
    Formula / method (brief):
      s5  = SMA(close, 5), s20 = SMA(close, 20)
      d   = s5 - s20
      cross_up   if d>0 and d.shift(1)<=0
      cross_down if d<0 and d.shift(1)>=0
      flag = +1/-1/0
    Input / Output / Constraints:
      As per base structure; vectorized; no look-ahead.
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    s5  = g["close"].rolling(5,  min_periods=5).mean()
    s20 = g["close"].rolling(20, min_periods=20).mean()
    d = s5 - s20

    cross_up   = (d > 0) & (d.shift(1) <= 0)
    cross_down = (d < 0) & (d.shift(1) >= 0)

    flag = np.where(cross_up, 1, np.where(cross_down, -1, 0)).astype(int)
    s = pd.Series(flag, index=g.index, name=FEATURE_CODE)
    return s