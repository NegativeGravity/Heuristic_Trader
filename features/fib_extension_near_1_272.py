# JUPYTER CELL — feature: fib_extension_near_1_272
FEATURE_CODE = "fib_extension_near_1_272"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Fibonacci Extension Near 1.272
    Description:
      Flags if the close is near the Fibonacci extension level of 1.272.
      Proximity is determined within a small range (ε = 0.01).
    Formula / method (brief):
      fib_1_272 = (high - low) * 1.272 + low
      flag = 1 if abs(close - fib_1_272) / close <= ε else 0
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    high = g["high"].rolling(2, min_periods=2).max()
    low = g["low"].rolling(2, min_periods=2).min()

    fib_1_272 = (high - low) * 1.272 + low

    epsilon = 0.01  # proximity range, you can adjust this value
    flag = (abs(g["close"] - fib_1_272) / g["close"] <= epsilon).astype(int)

    s = pd.Series(flag, index=g.index, name=FEATURE_CODE)
    return s