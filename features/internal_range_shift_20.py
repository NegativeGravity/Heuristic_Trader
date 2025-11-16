# JUPYTER CELL — feature: internal_range_shift_20
FEATURE_CODE = "internal_range_shift_20"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Internal Range Shift (20)
    Position of close inside 20-bar range, differenced:
      pos_t = (close - low20) / (high20 - low20)
      shift = pos_t - pos_{t-1}
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    c = g["close"].astype(float)
    h = g["high"].astype(float)
    l = g["low"].astype(float)

    hi20 = h.rolling(20, min_periods=5).max()
    lo20 = l.rolling(20, min_periods=5).min()
    rng = (hi20 - lo20).replace(0.0, np.nan)

    pos = (c - lo20) / rng
    shift = pos - pos.shift(1)

    s = shift.astype(float)
    s.name = FEATURE_CODE
    return s