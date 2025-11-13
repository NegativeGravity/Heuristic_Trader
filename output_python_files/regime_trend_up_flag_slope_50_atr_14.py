# JUPYTER CELL — feature: regime_trend_up_flag_slopeatr_50_14
FEATURE_CODE = "regime_trend_up_flag_slope_50_atr_14"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Regime: Uptrend Flag via Regression Slope normalized by ATR (W=50, ATR=14)
    Description:
      Flags 1 when rolling OLS slope(close~time,50) normalized by ATR(14) is above +k (k=0.05 by default),
      implying a persistent upward drift vs. recent volatility. Else 0.
    Method:
      slope50 = cov(t,y)/var(t) using rolling means (y=close, t=0..N-1)
      atr14 = Wilder ATR(14) on high/low/close
      z = slope50 / atr14
      flag = 1 if z >= k else 0   (k = 0.05)
    Notes:
      - Units: slope is price/bar; dividing by ATR (price units) yields per-bar in ATR units.
    """
    g = df.copy(); g.columns = [str(c).lower() for c in g.columns]
    y = g["close"].astype(float)
    n = len(g); W = 50

    # Rolling OLS slope
    t = pd.Series(np.arange(n, dtype=float), index=g.index)
    t_mean = t.rolling(W, min_periods=W).mean()
    y_mean = y.rolling(W, min_periods=W).mean()
    cov = (t*y).rolling(W, min_periods=W).mean() - t_mean*y_mean
    var = (t*t).rolling(W, min_periods=W).mean() - t_mean*t_mean
    slope = cov / var.replace(0.0, np.nan)

    # ATR(14) — Wilder
    h, l, c = g["high"].astype(float), g["low"].astype(float), g["close"].astype(float)
    tr = pd.concat([
        (h - l),
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()

    z = slope / atr14.replace(0.0, np.nan)
    k = 0.05
    flag = (z >= k).astype(int).fillna(0)
    s = pd.Series(flag.values, index=g.index, name=FEATURE_CODE)
    return s