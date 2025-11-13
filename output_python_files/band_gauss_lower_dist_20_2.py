# JUPYTER CELL — feature: band_gauss_lower_dist_20_2
FEATURE_CODE = "band_gauss_lower_dist_20_2"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Gaussian Lower Band Distance (20, 2σ)
    Description:
      Relative distance from close to the lower Gaussian band:
        lower = SMA(close,20) - 2 * std(close,20, ddof=0)
      (This is the common Bollinger lower band with 20 periods, 2 standard deviations.)
    Formula / method (brief):
      ma20 = rolling mean(close,20); sd20 = rolling std(close,20) with ddof=0
      lower = ma20 - 2 * sd20
      dist = (close - lower) / close
    Input/Output/Constraints:
      Standard; vectorized; no look-ahead.
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    ma20 = g["close"].rolling(20, min_periods=20).mean()
    sd20 = g["close"].rolling(20, min_periods=20).std(ddof=0)
    lower = ma20 - 2 * sd20

    s = (g["close"].astype(float) - lower) / g["close"].astype(float)
    s = s.astype(float); s.name = FEATURE_CODE
    return s