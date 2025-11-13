# JUPYTER CELL — feature: break_prev_high_flag_1
FEATURE_CODE = "break_prev_high_flag_1"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Break Previous High Flag (1 bar)
    Description:
      Flags 1 if the current close is higher than the high of the previous bar, indicating a breakout.
    Formula / method (brief):
      - Check if close_t > high_{t-1}.
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    prev_high = g["high"].shift(1)
    flag = (g["close"] > prev_high).astype(int)

    s = pd.Series(flag, index=g.index, name=FEATURE_CODE)
    return s