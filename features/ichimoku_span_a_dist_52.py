# JUPYTER CELL — feature: ichimoku_span_a_dist_52
FEATURE_CODE = "ichimoku_span_a_dist_52"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Ichimoku Senkou Span A Distance (lag-aligned)
    Description:
      Relative distance from close to Senkou Span A, aligned to current bar without look-ahead.
      Standard Span A is (Tenkan+Kijun)/2 shifted 26 bars forward. To avoid look-ahead,
      we use the unshifted value (equivalently, the standard Span A shifted BACK by 26),
      which depends only on past data.
    Formula / method (brief):
      tenkan = (max(high,9) + min(low,9))/2
      kijun  = (max(high,26)+ min(low,26))/2
      spanA_unshifted = (tenkan + kijun)/2
      dist = (close - spanA_unshifted)/close
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    hi9  = g["high"].rolling(9,  min_periods=9 ).max()
    lo9  = g["low"] .rolling(9,  min_periods=9 ).min()
    tenk = (hi9 + lo9) / 2.0

    hi26 = g["high"].rolling(26, min_periods=26).max()
    lo26 = g["low"] .rolling(26, min_periods=26).min()
    kij  = (hi26 + lo26) / 2.0

    span_a = (tenk + kij) / 2.0  # lag-aligned (no forward shift)
    s = (g["close"].astype(float) - span_a) / g["close"].astype(float)
    s = s.astype(float); s.name = FEATURE_CODE
    return s