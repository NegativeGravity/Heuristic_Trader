# JUPYTER CELL — feature: displacement_strength_10
FEATURE_CODE = "displacement_strength_10"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Displacement Strength (10)
    Idea:
      Measures impulsiveness of price move vs average volatility:
      strength = |close - close[-1]| / ATR(10)
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    c = g["close"].astype(float)
    h = g["high"].astype(float)
    l = g["low"].astype(float)
    prev_c = c.shift(1)

    tr = (h - l).combine((h - prev_c).abs(), np.maximum).combine((l - prev_c).abs(), np.maximum)
    atr = tr.rolling(10, min_periods=1).mean().replace(0.0, np.nan)

    s = (c - prev_c).abs() / atr
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s