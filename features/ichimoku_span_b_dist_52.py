# JUPYTER CELL — feature: ichimoku_span_b_dist_52
FEATURE_CODE = "ichimoku_span_b_dist_52"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Ichimoku Senkou Span B Distance (lag-aligned, 52)
    Description:
      Relative distance from close to Senkou Span B over 52 bars, aligned to current bar
      without look-ahead. Standard Span B = (max(high,52)+min(low,52))/2 shifted 26 forward;
      here we use the unshifted value (equivalently, standard Span B shifted BACK 26),
      which uses only past data at each row.
    Formula:
      hi52 = rolling max(high,52); lo52 = rolling min(low,52)
      spanB_unshifted = (hi52 + lo52)/2
      dist = (close - spanB_unshifted)/close
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    hi52 = g["high"].rolling(52, min_periods=52).max()
    lo52 = g["low"] .rolling(52, min_periods=52).min()
    span_b = (hi52 + lo52) / 2.0

    s = (g["close"].astype(float) - span_b) / g["close"].astype(float)
    s = s.astype(float); s.name = FEATURE_CODE
    return s