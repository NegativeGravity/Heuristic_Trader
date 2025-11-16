# JUPYTER CELL — feature: breaker_block_distance_20
FEATURE_CODE = "breaker_block_distance_20"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Breaker Block Distance (20)
    Idea:
      Approximate distance from current close to a recent "breaker" level
      using 20-bar rolling extremes.
      - If short-term direction is up -> use rolling 20-bar high as breaker
      - If short-term direction is down -> use rolling 20-bar low
      Distance is normalized by close.
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    close = g["close"].astype(float)
    high = g["high"].astype(float)
    low = g["low"].astype(float)

    dir_sign = np.sign(close - close.shift(1)).fillna(0.0)
    rh = high.rolling(20, min_periods=20).max()
    rl = low.rolling(20, min_periods=20).min()

    breaker_level = np.where(dir_sign >= 0, rh.shift(1), rl.shift(1))
    breaker_level = pd.Series(breaker_level, index=g.index)

    dist = (close - breaker_level) / close.replace(0.0, np.nan)
    s = dist.astype(float)
    s.name = FEATURE_CODE
    return s