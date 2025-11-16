# JUPYTER CELL — feature: range_tagging_bias_50
FEATURE_CODE = "range_tagging_bias_50"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Range Tagging Bias (50-bar window)

    Logic:
      - Compute 50-bar high, low and range:
          high50, low50, range50
      - Define a tagging threshold:
          thresh = 0.10 * range50
      - For each bar:
          high_tag = 1 if (high50 - close) <= thresh
          low_tag  = 1 if (close - low50)  <= thresh
        (If both are true, they both count; rare except very narrow ranges.)
      - Over last 50 bars:
          high_count = sum(high_tag)
          low_count  = sum(low_tag)
          bias = (high_count - low_count) / 50

      Interpretation:
        +1 → mostly tagging upper part of range
        -1 → mostly tagging lower part
         0 → symmetric.
    """

    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    high = g["high"].astype(float)
    low = g["low"].astype(float)
    close = g["close"].astype(float)

    win = 50

    high50 = high.rolling(win).max()
    low50  = low .rolling(win).min()
    range50 = (high50 - low50).replace(0.0, np.nan)

    thresh = 0.10 * range50

    high_tag = ((high50 - close) <= thresh).astype(float)
    low_tag  = ((close - low50) <= thresh).astype(float)

    high_count = high_tag.rolling(win, min_periods=1).sum()
    low_count  = low_tag .rolling(win, min_periods=1).sum()

    bias = (high_count - low_count) / float(win)
    bias = bias.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return pd.Series(bias.values, index=g.index, name=FEATURE_CODE)