# JUPYTER CELL — feature: pivot_classic_r2_dist_1d
FEATURE_CODE = "pivot_classic_r2_dist_1d"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Classic Pivot R2 distance (previous day)
    Description:
      Relative distance from close to prior day's Classic R2.
      R2_prev = PP_prev + (H_prev - L_prev), where PP_prev = (H_prev+L_prev+C_prev)/3.
    Formula / method (brief):
      - Daily OHLC from intraday; shift by 1.
      - Compute PP_prev, then R2_prev.
      - dist = (close_t - R2_today)/close_t
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
    r2_daily = pp + (dprev["high"] - dprev["low"])

    r2_intraday = pd.Series(pd.Index(days).map(r2_daily), index=g.index, dtype=float)

    s = (g["close"].astype(float) - r2_intraday) / g["close"].astype(float)
    s.name = FEATURE_CODE
    return s