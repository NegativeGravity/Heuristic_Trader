# JUPYTER CELL — feature: liq_zone_touch_flag_50
FEATURE_CODE = "liq_zone_touch_flag_50"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Liquidity Zone Touch Flag (50)
    Description:
      Flags 1 if the close touches the liquidity zone (high-low range) over the last 50 bars.
      Proximity is determined within a small range (ε = 0.01).
    Formula / method (brief):
      liquidity_zone = high-low for the last 50 bars
      flag = 1 if close_t is within liquidity_zone (+ε) and (-ε)
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    high = g["high"]
    low = g["low"]

    liquidity_zone_high = high.rolling(50, min_periods=50).max()
    liquidity_zone_low = low.rolling(50, min_periods=50).min()

    epsilon = 0.01  # proximity range
    flag = ((g["close"] >= liquidity_zone_low - epsilon) & (g["close"] <= liquidity_zone_high + epsilon)).astype(int)

    s = pd.Series(flag, index=g.index, name=FEATURE_CODE)
    return s