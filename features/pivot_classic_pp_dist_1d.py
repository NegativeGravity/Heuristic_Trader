# JUPYTER CELL — feature: pivot_classic_pp_dist_1d
FEATURE_CODE = "pivot_classic_pp_dist_1d"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Classic Pivot Point distance (previous day)
    Description:
      Relative distance from the current close to the prior day's Classic Pivot Point (PP).
      PP_prev_day = (H_prev + L_prev + C_prev) / 3, computed from the previous trading day.
    Formula / method (brief):
      - Aggregate intraday into daily H/L/C via groupby(normalized date).
      - Shift by 1 day to avoid look-ahead.
      - PP = (H_prev + L_prev + C_prev)/3
      - dist = (close_t - PP_for_today)/close_t
    Input:
      df: DataFrame with DatetimeIndex (ascending), columns:
           open, high, low, close, volume (case-insensitive)
    Output:
      pd.Series (float), same index as df.index, name == FEATURE_CODE.
      First day(s) will be NaN (no prior day).
    Constraints:
      - No look-ahead (uses prior day levels only).
      - Vectorized (groupby + map).
      - Uses numpy and pandas only.
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]  # normalize
    days = g.index.normalize()

    # Daily OHLC (based on intraday)
    daily = g.assign(__day=days).groupby("__day").agg(
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    )

    # Previous-day pivots
    daily_prev = daily.shift(1)
    pp_daily = (daily_prev["high"] + daily_prev["low"] + daily_prev["close"]) / 3.0

    # Map each intraday bar's day -> that day's PP (from previous day)
    pp_intraday = pd.Series(pd.Index(days).map(pp_daily), index=g.index, dtype=float)

    s = (g["close"].astype(float) - pp_intraday) / g["close"].astype(float)
    s.name = FEATURE_CODE
    return s