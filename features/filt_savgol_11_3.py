# JUPYTER CELL — feature: filt_savgol_11_3
FEATURE_CODE = "filt_savgol_11_3"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Savitzky-Golay-like Filter (window=11, poly=3, causal)
    Approximates a SG(11,3) on the last 11 closes via polynomial regression.
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]
    c = g["close"].astype(float)

    def sg_causal(x: np.ndarray) -> float:
        n = len(x)
        if n < 5:
            return float(x[-1])
        # Fit poly of degree 3 on indices [0..n-1], return fitted value at last index
        xs = np.arange(n, dtype=float)
        coeffs = np.polyfit(xs, x, deg=3)
        val = np.polyval(coeffs, xs[-1])
        return float(val)

    s = c.rolling(11, min_periods=5).apply(sg_causal, raw=True)
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s