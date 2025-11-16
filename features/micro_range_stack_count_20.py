# JUPYTER CELL — feature: micro_range_stack_count_20
FEATURE_CODE = "micro_range_stack_count_20"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Micro Range Stack Count (20-bar rolling)

    Logic:
      - Compute true range of each candle.
      - Determine micro-ranges by comparing TR to the rolling 25th percentile
        of the last 100 TR values.
      - A micro-range is TR <= threshold.
      - This feature returns the rolling 20-bar sum of micro-range flags.
    """

    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    high = g["high"].astype(float)
    low = g["low"].astype(float)

    # True range
    tr = (high - low).abs()

    # Micro-range threshold (25th percentile)
    threshold = tr.rolling(100, min_periods=30).quantile(0.25)

    micro_flag = (tr <= threshold).astype(float).fillna(0.0)

    # Count of micro ranges over last 20 bars
    count = micro_flag.rolling(20, min_periods=1).sum()

    return pd.Series(count, index=g.index, name=FEATURE_CODE)