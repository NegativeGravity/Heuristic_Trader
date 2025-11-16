# JUPYTER CELL — feature: reg_trending_flag_30
FEATURE_CODE = "reg_trending_flag_30"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Regime Trending Flag (30-bar window)

    Logic:
      Flags whether the market is in a trending regime over the last ~30 bars.

      Steps:
        - Compute an ATR-like volatility using a 14-bar mean of true range:
              tr  = |high - low|
              atr14 = mean(tr, 14)
        - Compute a 30-bar net slope of close:
              slope30 = (close - close.shift(30)) / 30
        - Normalize slope by volatility:
              norm_slope = |slope30| / (atr14 + eps)
        - Flag as trending if norm_slope > threshold.

      Output:
        Integer flag in {0, 1}.
    """

    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    high = g["high"].astype(float)
    low  = g["low"].astype(float)
    close = g["close"].astype(float)

    eps = 1e-9

    # True range and ATR-like volatility
    tr = (high - low).abs()
    atr14 = tr.rolling(14).mean()

    # 30-bar slope of close
    slope30 = (close - close.shift(30)) / 30.0

    norm_slope = slope30.abs() / (atr14 + eps)

    # Threshold can be tuned based on asset / timeframe
    threshold = 0.5
    flag = (norm_slope > threshold).astype(int).fillna(0)

    return pd.Series(flag.values, index=g.index, name=FEATURE_CODE)