# JUPYTER CELL — feature: fib_retracement_near_0_618
FEATURE_CODE = "fib_retracement_near_0_618"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Fibonacci Retracement Near 0.618
    Description:
      Flags if the close is near the Fibonacci retracement level of 61.8%.
      Proximity is determined within a small range (ε = 0.01).
    Formula / method (brief):
      fib_0_618 = (high - low) * 0.618 + low
      flag = 1 if abs(close - fib_0_618) / close <= ε else 0
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    high = g["high"].rolling(2, min_periods=2).max()
    low = g["low"].rolling(2, min_periods=2).min()

    fib_0_618 = (high - low) * 0.618 + low

    epsilon = 0.01  # proximity range, you can adjust this value
    flag = (abs(g["close"] - fib_0_618) / g["close"] <= epsilon).astype(int)

    s = pd.Series(flag, index=g.index, name=FEATURE_CODE)
    return s