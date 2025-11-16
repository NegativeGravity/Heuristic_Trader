# JUPYTER CELL — feature: volprof_val_dist_100 (robust)
FEATURE_CODE = "volprof_val_dist_100"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Volume Profile VAL distance (100)
    Description:
      Relative distance from close to Value Area Low (VAL) ≈ weighted 15% quantile
      of close over a 100-bar window (weights=volume).
    Formula / method (brief):
      VAL := weighted_quantile(close, weights=volume, q=0.15)
      dist := (close_t - VAL) / close_t
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]
    if not {"close","volume"}.issubset(g.columns):
        raise ValueError("DataFrame must contain 'close' and 'volume'.")

    def _wq(px: np.ndarray, w: np.ndarray, q: float) -> float:
        m = np.isfinite(px) & np.isfinite(w) & (w >= 0)
        px = px[m]; w = w[m]
        if px.size < 3 or w.sum() <= 0:
            return np.nan
        order = np.argsort(px)
        px = px[order]; w = w[order]
        csum = np.cumsum(w)
        thr = q * csum[-1]
        i = int(np.searchsorted(csum, thr, side="left"))
        i = min(max(i, 0), len(px)-1)
        return float(px[i])

    c = g["close"].to_numpy(float)
    v = g["volume"].to_numpy(float)
    n = len(g); W = 100

    level = np.full(n, np.nan, float)
    for i in range(W-1, n):
        cs = c[i-W+1:i+1]
        vs = v[i-W+1:i+1]
        level[i] = _wq(cs, vs, 0.15)

    s = (c - level) / c
    s = pd.Series(s, index=g.index, dtype=float, name=FEATURE_CODE)
    return s