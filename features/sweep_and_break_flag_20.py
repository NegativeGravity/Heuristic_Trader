# JUPYTER CELL — feature: sweep_and_break_flag_20
FEATURE_CODE = "sweep_and_break_flag_20"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Sweep-and-Break Flag (20-bar lookback)

    Logic (single-bar proxy):

      1) Compute prior 20-bar extremes (excluding current bar):
           prior_high_20 = rolling_max(high, 20).shift(1)
           prior_low_20  = rolling_min(low,  20).shift(1)

      2) Define:
           up_sweep   = high > prior_high_20
           down_sweep = low  < prior_low_20

      3) Define "break" in the opposite direction using bar close vs open:
           break_down_after_up = (up_sweep   & (close < open))
           break_up_after_down = (down_sweep & (close > open))

      4) Flag:
           flag = 1 if either break_down_after_up or break_up_after_down, else 0.

      This is a compact, event-style SMC proxy:
        - It first takes liquidity beyond a prior local extreme (sweep),
        - Then closes with momentum in the opposite direction (break).
    """

    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    high  = g["high"].astype(float)
    low   = g["low"].astype(float)
    open_ = g["open"].astype(float)
    close = g["close"].astype(float)

    lookback = 20

    prior_high_20 = high.rolling(lookback).max().shift(1)
    prior_low_20  = low .rolling(lookback).min().shift(1)

    up_sweep   = (high > prior_high_20)
    down_sweep = (low  < prior_low_20)

    break_down_after_up = up_sweep & (close < open_)
    break_up_after_down = down_sweep & (close > open_)

    flag = (break_down_after_up | break_up_after_down).astype(int).fillna(0)

    return pd.Series(flag.values, index=g.index, name=FEATURE_CODE)