# JUPYTER CELL — feature: prior_range_overlap_ratio_50
FEATURE_CODE = "prior_range_overlap_ratio_50"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Prior Range Overlap Ratio (50-bar window)

    Logic:
      For each bar:
        - current range:  [low50, high50]   from last 50 bars including current
        - prior range:    [low50_prev, high50_prev] from last 50 bars ending at previous bar
        - intersection = max(0, min(high50, high50_prev) - max(low50, low50_prev))
        - union        = max(0, max(high50, high50_prev) - min(low50, low50_prev))
        - ratio        = intersection / union

      Output is between 0 and 1 (0 = no overlap, 1 = identical ranges).
    """

    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    high = g["high"].astype(float)
    low = g["low"].astype(float)

    win = 50

    high50 = high.rolling(win).max()
    low50  = low .rolling(win).min()

    high50_prev = high50.shift(1)
    low50_prev  = low50.shift(1)

    # Intersection
    inter_low  = np.maximum(low50, low50_prev)
    inter_high = np.minimum(high50, high50_prev)
    intersection = (inter_high - inter_low).clip(lower=0.0)

    # Union
    union_low  = np.minimum(low50, low50_prev)
    union_high = np.maximum(high50, high50_prev)
    union = (union_high - union_low).clip(lower=0.0)

    ratio = intersection / union.replace(0.0, np.nan)
    ratio = ratio.clip(0.0, 1.0).fillna(0.0)

    return pd.Series(ratio.values, index=g.index, name=FEATURE_CODE)