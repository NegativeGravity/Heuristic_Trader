# JUPYTER CELL — feature: liq_daily_zone_touch_flag_1d
FEATURE_CODE = "liq_daily_zone_touch_flag_1d"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Liquidity Zone Daily Touch Flag (1d)
    Description:
      Flags 1 if the close touches the liquidity zone (high-low range) of the prior day.
      Proximity is determined within a small range (ε = 0.01).
    Formula / method (brief):
      daily_high = high of previous day
      daily_low = low of previous day
      flag = 1 if close_t is between (daily_low - ε) and (daily_high + ε)
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    high_prev = g["high"].shift(1)
    low_prev = g["low"].shift(1)

    epsilon = 0.01  # proximity range
    flag = ((g["close"] >= low_prev - epsilon) & (g["close"] <= high_prev + epsilon)).astype(int)

    s = pd.Series(flag, index=g.index, name=FEATURE_CODE)
    return s