# JUPYTER CELL — feature: equal_lows_tightness_20
FEATURE_CODE = "equal_lows_tightness_20"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Equal Lows Tightness (20)
    Same idea as highs, for lows:
      tightness = (max_low_20 - min_low_20) / close
      Lower values = tighter support zone.
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    l = g["low"].astype(float)
    c = g["close"].astype(float)

    max_l = l.rolling(20, min_periods=5).max()
    min_l = l.rolling(20, min_periods=5).min()

    tightness = (max_l - min_l) / c.replace(0.0, np.nan)
    s = tightness.astype(float)
    s.name = FEATURE_CODE
    return s