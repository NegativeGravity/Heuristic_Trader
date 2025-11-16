# JUPYTER CELL — feature: pivot_classic_s2_dist_1d
FEATURE_CODE = "pivot_classic_s2_dist_1d"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Classic Pivot S2 distance (previous day)
    Description:
      Relative distance from close to prior day's Classic S2.
      S2_prev = PP_prev - (H_prev - L_prev), where PP_prev = (H_prev+L_prev+C_prev)/3.
    Formula / method (brief):
      - Daily OHLC from intraday; shift by 1.
      - Compute PP_prev, then S2_prev.
      - dist = (close_t - S2_today)/close_t
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
    s2_daily = pp - (dprev["high"] - dprev["low"])

    s2_intraday = pd.Series(pd.Index(days).map(s2_daily), index=g.index, dtype=float)

    s = (g["close"].astype(float) - s2_intraday) / g["close"].astype(float)
    s.name = FEATURE_CODE
    return s