# JUPYTER CELL — feature: equal_highs_tightness_20
FEATURE_CODE = "equal_highs_tightness_20"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Equal Highs Tightness (20)
    Measures how tight 20-bar highs are:
      tightness = (max_high_20 - min_high_20) / close
      Lower values = tighter equal-highs zone.
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    h = g["high"].astype(float)
    c = g["close"].astype(float)

    max_h = h.rolling(20, min_periods=5).max()
    min_h = h.rolling(20, min_periods=5).min()

    tightness = (max_h - min_h) / c.replace(0.0, np.nan)
    s = tightness.astype(float)
    s.name = FEATURE_CODE
    return s