# JUPYTER CELL — feature: break_prev_low_flag_1
FEATURE_CODE = "break_prev_low_flag_1"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Break Previous Low Flag (1 bar)
    Description:
      Flags 1 if the current close is lower than the low of the previous bar, indicating a breakdown.
    Formula / method (brief):
      - Check if close_t < low_{t-1}.
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    prev_low = g["low"].shift(1)
    flag = (g["close"] < prev_low).astype(int)

    s = pd.Series(flag, index=g.index, name=FEATURE_CODE)
    return s