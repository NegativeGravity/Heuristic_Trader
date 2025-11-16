# JUPYTER CELL — feature: premium_discount_balance_50
FEATURE_CODE = "premium_discount_balance_50"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Premium–Discount Balance (50-bar window)

    Logic:
      - Compute 50-bar rolling high and low.
      - Midpoint: mid50 = (high50 + low50) / 2
      - For each bar:
          premium_flag  = 1 if close > mid50
          discount_flag = 1 if close < mid50
      - Feature = (sum(premium_flag) - sum(discount_flag)) / 50
        over the last 50 bars.

      Interpretation:
        +1 → closes mostly in premium
        -1 → closes mostly in discount
         0 → balanced.
    """

    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    high = g["high"].astype(float)
    low = g["low"].astype(float)
    close = g["close"].astype(float)

    win = 50

    high50 = high.rolling(win).max()
    low50  = low .rolling(win).min()
    mid50  = (high50 + low50) / 2.0

    premium_flag  = (close > mid50).astype(float)
    discount_flag = (close < mid50).astype(float)

    premium_count  = premium_flag.rolling(win, min_periods=1).sum()
    discount_count = discount_flag.rolling(win, min_periods=1).sum()

    balance = (premium_count - discount_count) / float(win)
    balance = balance.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return pd.Series(balance.values, index=g.index, name=FEATURE_CODE)