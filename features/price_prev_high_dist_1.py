# JUPYTER CELL — feature: price_prev_high_dist_1
FEATURE_CODE = "price_prev_high_dist_1"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Price Previous High Distance (1 bar)
    Description:
      Flags 1 if the close is close to the high of the previous bar.
      Proximity is determined within a small range (ε = 0.01).
    Formula / method (brief):
      prev_high = high of previous bar
      flag = 1 if abs(close_t - prev_high) / close_t <= ε else 0
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    prev_high = g["high"].shift(1)

    epsilon = 0.01  # proximity range
    flag = (abs(g["close"] - prev_high) / g["close"] <= epsilon).astype(int)

    s = pd.Series(flag, index=g.index, name=FEATURE_CODE)
    return s