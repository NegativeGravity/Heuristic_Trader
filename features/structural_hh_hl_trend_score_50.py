# JUPYTER CELL — feature: structural_hh_hl_trend_score_50
FEATURE_CODE = "structural_hh_hl_trend_score_50"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Structural HH/HL Trend Score (50-bar window)

    Logic:
      A simple bar-level structural proxy:

      For each bar:
        - up_struct   = 1 if high > prev_high and low > prev_low      (HH + HL)
        - down_struct = -1 if high < prev_high and low < prev_low     (LL + LH)
        - else 0

      Then:
        trend_score_50 = rolling mean of this structural sign over 50 bars.

      Output:
        Float in [-1, 1]:
          +1 ~ strongly HH/HL
          -1 ~ strongly LL/LH
           0 ~ mixed/choppy.
    """

    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    high = g["high"].astype(float)
    low  = g["low"].astype(float)

    prev_high = high.shift(1)
    prev_low  = low.shift(1)

    up_struct   = ((high > prev_high) & (low > prev_low)).astype(int)
    down_struct = ((high < prev_high) & (low < prev_low)).astype(int) * -1

    struct_sign = up_struct + down_struct
    struct_sign = struct_sign.astype(float).fillna(0.0)

    win = 50
    trend_score = struct_sign.rolling(win, min_periods=1).mean()
    trend_score = trend_score.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return pd.Series(trend_score.values, index=g.index, name=FEATURE_CODE)