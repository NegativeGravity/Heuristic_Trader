# JUPYTER CELL — feature: liq_weekly_zone_touch_flag_1w
FEATURE_CODE = "liq_weekly_zone_touch_flag_1w"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Liquidity Zone Weekly Touch Flag (1w)
    Description:
      Flags 1 if the close touches the liquidity zone (high-low range) of the prior week.
      Proximity is determined within a small range (ε = 0.01).
    Formula / method (brief):
      weekly_high = high of previous week
      weekly_low = low of previous week
      flag = 1 if close_t is between (weekly_low - ε) and (weekly_high + ε)
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    high_prev = g["high"].shift(5)  # Assumes 5 trading days per week
    low_prev = g["low"].shift(5)  # Assumes 5 trading days per week

    epsilon = 0.01  # proximity range
    flag = ((g["close"] >= low_prev - epsilon) & (g["close"] <= high_prev + epsilon)).astype(int)

    s = pd.Series(flag, index=g.index, name=FEATURE_CODE)
    return s