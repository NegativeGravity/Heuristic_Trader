# JUPYTER CELL — feature: filt_dema_20
FEATURE_CODE = "filt_dema_20"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Double EMA Filter (20)
    DEMA(20) = 2 * EMA(20) - EMA(EMA(20))
    Causal, no look-ahead smoothing of close.
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    c = g["close"].astype(float)

    ema1 = c.ewm(span=20, adjust=False).mean()
    ema2 = ema1.ewm(span=20, adjust=False).mean()
    dema = 2.0 * ema1 - ema2

    s = dema.astype(float)
    s.name = FEATURE_CODE
    return s