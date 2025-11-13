# JUPYTER CELL — feature: pivot_classic_r1_dist_1d
FEATURE_CODE = "pivot_classic_r1_dist_1d"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Classic Pivot R1 distance (previous day)
    Description:
      Relative distance from close to prior day's Classic R1.
      R1_prev = 2*PP_prev - L_prev, where PP_prev = (H_prev+L_prev+C_prev)/3.
    Formula / method (brief):
      - Daily OHLC from intraday; shift by 1 to get prior day.
      - Compute PP_prev, then R1_prev.
      - dist = (close_t - R1_today)/close_t
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]
    days = g.index.normalize()

    daily = g.assign(__day=days).groupby("__day").agg(
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    )
    dprev = daily.shift(1)
    pp = (dprev["high"] + dprev["low"] + dprev["close"]) / 3.0
    r1_daily = 2.0 * pp - dprev["low"]

    r1_intraday = pd.Series(pd.Index(days).map(r1_daily), index=g.index, dtype=float)

    s = (g["close"].astype(float) - r1_intraday) / g["close"].astype(float)
    s.name = FEATURE_CODE
    return s