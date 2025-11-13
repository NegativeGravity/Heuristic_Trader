# JUPYTER CELL — feature: ichimoku_cloud_thickness_52
FEATURE_CODE = "ichimoku_cloud_thickness_52"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Ichimoku Cloud Thickness (lag-aligned, 52)
    Description:
      Normalized thickness of the Ichimoku cloud at the current bar (no look-ahead),
      defined as |SpanA - SpanB| / close. Both spans are the lag-aligned versions:
        SpanA = (Tenkan + Kijun)/2  (unshifted)
        SpanB = (max(high,52) + min(low,52))/2  (unshifted)
    Formula:
      tenkan(9), kijun(26), spanA=(tenkan+kijun)/2; spanB=(hi52+lo52)/2
      thickness = abs(spanA - spanB)/close
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    hi9  = g["high"].rolling(9,  min_periods=9 ).max()
    lo9  = g["low"] .rolling(9,  min_periods=9 ).min()
    tenk = (hi9 + lo9) / 2.0

    hi26 = g["high"].rolling(26, min_periods=26).max()
    lo26 = g["low"] .rolling(26, min_periods=26).min()
    kij  = (hi26 + lo26) / 2.0

    span_a = (tenk + kij) / 2.0
    hi52 = g["high"].rolling(52, min_periods=52).max()
    lo52 = g["low"] .rolling(52, min_periods=52).min()
    span_b = (hi52 + lo52) / 2.0

    thickness = (span_a - span_b).abs() / g["close"].astype(float)
    thickness = thickness.astype(float)
    thickness.name = FEATURE_CODE
    return thickness