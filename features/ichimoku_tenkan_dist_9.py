# JUPYTER CELL — feature: ichimoku_tenkan_dist_9
FEATURE_CODE = "ichimoku_tenkan_dist_9"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Ichimoku Tenkan Distance (9)
    Description:
      Relative distance from close to Tenkan-sen over 9 bars.
      Tenkan(9) = (rolling_high_9 + rolling_low_9) / 2.
    Formula / method (brief):
      hi9 = rolling max(high, 9); lo9 = rolling min(low, 9)
      tenkan = (hi9 + lo9)/2
      dist = (close - tenkan)/close
    Input/Output/Constraints:
      Standard; no look-ahead; vectorized.
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    hi9 = g["high"].rolling(9, min_periods=9).max()
    lo9 = g["low"].rolling(9, min_periods=9).min()
    tenkan = (hi9 + lo9) / 2.0

    s = (g["close"].astype(float) - tenkan) / g["close"].astype(float)
    s = s.astype(float); s.name = FEATURE_CODE
    return s