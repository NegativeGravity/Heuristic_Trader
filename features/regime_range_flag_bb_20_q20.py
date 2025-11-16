# JUPYTER CELL — feature: regime_range_flag_bb_20_q20
FEATURE_CODE = "regime_range_flag_bb_20_q20"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Regime: Range Flag via Bollinger Bandwidth (BB(20), below 20th percentile over 120 bars)
    Description:
      Flags 1 when the 20-bar Bollinger Bandwidth is unusually low (<= rolling 20% quantile over 120 bars),
      which often corresponds to ranging/sideways regimes. Otherwise 0.
    Method:
      ma20 = SMA(close,20); sd20 = STD(close,20, ddof=0)
      bbw20 = (upper - lower) / ma20 = (2*sd20 + 2*sd20) / ma20 = 4*sd20/ma20
      thresh = rolling_quantile(bbw20, window=120, q=0.20) (using pandas.Series.quantile on rolling)
      flag = 1 if bbw20 <= thresh else 0
    Constraints:
      - No look-ahead (threshold is from rolling past+current window).
      - Vectorized; numpy & pandas only.
    """
    g = df.copy(); g.columns = [str(c).lower() for c in g.columns]
    c = g["close"].astype(float)

    ma20 = c.rolling(20, min_periods=20).mean()
    sd20 = c.rolling(20, min_periods=20).std(ddof=0)
    bbw20 = (4.0 * sd20) / ma20.replace(0.0, np.nan)  # normalized width

    # Rolling 20th percentile over 120 bars
    # Note: rolling(...).quantile(q) is vectorized and avoids look-ahead.
    thresh = bbw20.rolling(120, min_periods=120).quantile(0.20, interpolation="linear")

    flag = (bbw20 <= thresh).astype(int).fillna(0)
    s = pd.Series(flag.values, index=g.index, name=FEATURE_CODE)
    return s