# JUPYTER CELL — feature: market_structure_break_count_50
FEATURE_CODE = "market_structure_break_count_50"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Market Structure Break Count (50-bar window)

    Logic:
      A simple, direction-agnostic count of structure breaks.

      1) Define prior 20-bar extremes (excluding current bar):
           prior_high_20 = rolling_max(high, 20).shift(1)
           prior_low_20  = rolling_min(low,  20).shift(1)

      2) Structure breaks:
           bull_break = close > prior_high_20
           bear_break = close < prior_low_20

      3) For each bar:
           break_flag = 1 if (bull_break or bear_break) else 0

      4) Feature:
           market_structure_break_count_50 =
               rolling_sum(break_flag over last 50 bars)

      Output:
        Non-negative float / int count.
    """

    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    high  = g["high"].astype(float)
    low   = g["low"].astype(float)
    close = g["close"].astype(float)

    lookback_ref = 20
    lookback_count = 50

    prior_high_20 = high.rolling(lookback_ref).max().shift(1)
    prior_low_20  = low .rolling(lookback_ref).min().shift(1)

    bull_break = (close > prior_high_20)
    bear_break = (close < prior_low_20)

    break_flag = (bull_break | bear_break).astype(float).fillna(0.0)

    break_count_50 = break_flag.rolling(lookback_count, min_periods=1).sum()

    return pd.Series(break_count_50.values, index=g.index, name=FEATURE_CODE)