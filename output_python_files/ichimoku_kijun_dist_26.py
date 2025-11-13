# JUPYTER CELL — feature: ichimoku_kijun_dist_26
FEATURE_CODE = "ichimoku_kijun_dist_26"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Ichimoku Kijun Distance (26)
    Description:
      Relative distance from close to Kijun-sen over 26 bars.
      Kijun(26) = (rolling_high_26 + rolling_low_26) / 2.
    Formula:
      hi26 = rolling max(high,26); lo26 = rolling min(low,26)
      kijun = (hi26 + lo26)/2
      dist = (close - kijun)/close
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    hi26 = g["high"].rolling(26, min_periods=26).max()
    lo26 = g["low"].rolling(26, min_periods=26).min()
    kijun = (hi26 + lo26) / 2.0

    s = (g["close"].astype(float) - kijun) / g["close"].astype(float)
    s = s.astype(float); s.name = FEATURE_CODE
    return s