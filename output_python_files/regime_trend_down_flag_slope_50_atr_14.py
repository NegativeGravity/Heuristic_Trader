# JUPYTER CELL — feature: regime_trend_down_flag_slopeatr_50_14
FEATURE_CODE = "regime_trend_down_flag_slope_50_atr_14"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Regime: Downtrend Flag via Regression Slope normalized by ATR (W=50, ATR=14)
    Description:
      Flags 1 when rolling OLS slope(close~time,50) normalized by ATR(14) is below −k (k=0.05),
      implying a persistent downward drift vs. volatility. Else 0.
    Method:
      Same slope/ATR as the uptrend version.
      z = slope50 / atr14
      flag = 1 if z <= -k else 0
    """
    g = df.copy(); g.columns = [str(c).lower() for c in g.columns]
    y = g["close"].astype(float)
    n = len(g); W = 50

    t = pd.Series(np.arange(n, dtype=float), index=g.index)
    t_mean = t.rolling(W, min_periods=W).mean()
    y_mean = y.rolling(W, min_periods=W).mean()
    cov = (t*y).rolling(W, min_periods=W).mean() - t_mean*y_mean
    var = (t*t).rolling(W, min_periods=W).mean() - t_mean*t_mean
    slope = cov / var.replace(0.0, np.nan)

    h, l, c = g["high"].astype(float), g["low"].astype(float), g["close"].astype(float)
    tr = pd.concat([
        (h - l),
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()

    z = slope / atr14.replace(0.0, np.nan)
    k = 0.05
    flag = (z <= -k).astype(int).fillna(0)
    s = pd.Series(flag.values, index=g.index, name=FEATURE_CODE)
    return s