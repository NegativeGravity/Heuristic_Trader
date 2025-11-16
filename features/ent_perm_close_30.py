# JUPYTER CELL — feature: ent_perm_close_30
FEATURE_CODE = "ent_perm_close_30"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Permutation-like Entropy of Close (30)
    Approximation:
      Uses Shannon entropy of rank-discretized closes over a 30-bar window.
      Normalized to [0,1].
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]
    c = g["close"].astype(float)

    def window_entropy(x: np.ndarray) -> float:
        if len(x) < 3:
            return np.nan
        # Rank discretization
        ranks = pd.Series(x).rank(method="average").values
        # Bin into 5 quantile-buckets
        qs = np.quantile(ranks, [0.2, 0.4, 0.6, 0.8])
        bins = np.digitize(ranks, qs)
        counts = np.bincount(bins, minlength=5).astype(float)
        p = counts / counts.sum() if counts.sum() > 0 else counts
        p = p[p > 0]
        if len(p) == 0:
            return np.nan
        ent = -np.sum(p * np.log(p))
        # Max entropy with 5 bins
        ent_norm = ent / np.log(5.0)
        return float(ent_norm)

    s = c.rolling(30, min_periods=10).apply(window_entropy, raw=True)
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s