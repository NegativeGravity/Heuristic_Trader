# JUPYTER CELL — feature: filt_gauss_close_20
FEATURE_CODE = "filt_gauss_close_20"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Gaussian Weighted Moving Average (20, causal)
    Uses a backward-looking Gaussian kernel of length 20 on closes.
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]
    c = g["close"].astype(float)

    window = 20
    idx = np.arange(window)
    # center at current bar (0) and decay into the past
    sigma = window / 4.0
    weights = np.exp(-0.5 * (idx / sigma) ** 2)
    weights = weights[::-1]  # bigger weight on most recent
    weights /= weights.sum()

    def gauss(x: np.ndarray) -> float:
        if len(x) < window:
            w = weights[-len(x):]
        else:
            w = weights
        return float(np.sum(x * w))

    s = c.rolling(window, min_periods=3).apply(gauss, raw=True)
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s