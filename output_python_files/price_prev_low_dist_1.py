# JUPYTER CELL — feature: price_prev_low_dist_1
FEATURE_CODE = "price_prev_low_dist_1"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Price Previous Low Distance (1 bar)
    Description:
      Flags 1 if the close is close to the low of the previous bar.
      Proximity is determined within a small range (ε = 0.01).
    Formula / method (brief):
      prev_low = low of previous bar
      flag = 1 if abs(close_t - prev_low) / close_t <= ε else 0
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    prev_low = g["low"].shift(1)

    epsilon = 0.01  # proximity range
    flag = (abs(g["close"] - prev_low) / g["close"] <= epsilon).astype(int)

    s = pd.Series(flag, index=g.index, name=FEATURE_CODE)
    return s