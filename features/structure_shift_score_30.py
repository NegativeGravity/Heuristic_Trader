# JUPYTER CELL — feature: structure_shift_score_30
FEATURE_CODE = "structure_shift_score_30"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Structure Shift Score (30-bar comparison)

    Logic:
      1) Build a simple structural sign per bar:
           +1 if high > prev_high and low > prev_low    (HH + HL)
           -1 if high < prev_high and low < prev_low    (LL + LH)
            0 otherwise.

      2) Compute a 30-bar rolling mean of this structural sign:
           bias_30

      3) Compare the current bias_30 with bias_30 from 30 bars ago:
           shift_score = bias_30 - bias_30.shift(30)

      Interpretation:
        - Large positive values → structural regime has shifted toward bullish.
        - Large negative values → structural regime has shifted toward bearish.
        - Values near zero → little net structural change over that horizon.
    """

    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    high = g["high"].astype(float)
    low  = g["low"].astype(float)

    prev_high = high.shift(1)
    prev_low  = low.shift(1)

    up_struct   = ((high > prev_high) & (low > prev_low)).astype(int)
    down_struct = ((high < prev_high) & (low < prev_low)).astype(int) * -1

    struct_sign = (up_struct + down_struct).astype(float).fillna(0.0)

    win = 30

    bias_30 = struct_sign.rolling(win, min_periods=1).mean()
    shift_score = bias_30 - bias_30.shift(win)

    shift_score = shift_score.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return pd.Series(shift_score.values, index=g.index, name=FEATURE_CODE)