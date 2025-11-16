# JUPYTER CELL — feature: volprof_poc_dist_100 (robust)
FEATURE_CODE = "volprof_poc_dist_100"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Volume Profile POC distance (100)
    Description:
      Relative distance from the current close to the Point of Control (POC)
      over a 100-bar window; POC via weighted histogram mode (weights=volume).
    Formula / method (brief):
      For each window (size=100):
        - Weighted histogram of close (bins=50 over [min,max]).
        - POC := center of the max-weight bin.
        - dist := (close_t - POC) / close_t
    Input:
      df with columns open, high, low, close, volume (case-insensitive).
    Output:
      pd.Series (float), same index, name == FEATURE_CODE; initial NaNs allowed.
    Constraints:
      - No look-ahead. Uses only current/past data.
      - Numpy/Pandas only (loop over windows; reliable across pandas versions).
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]
    if not {"close","volume"}.issubset(g.columns):
        raise ValueError("DataFrame must contain 'close' and 'volume'.")

    c = g["close"].to_numpy(float)
    v = g["volume"].to_numpy(float)
    n = len(g); W = 100; BINS = 50

    level = np.full(n, np.nan, float)

    for i in range(W-1, n):
        cs = c[i-W+1:i+1]
        vs = v[i-W+1:i+1]
        m = np.isfinite(cs) & np.isfinite(vs)
        if m.sum() < 3:
            continue
        cs = cs[m]; vs = vs[m]
        pmin, pmax = cs.min(), cs.max()
        if not np.isfinite(pmin) or not np.isfinite(pmax) or pmax <= pmin:
            continue
        hist, edges = np.histogram(cs, bins=BINS, range=(pmin, pmax), weights=vs)
        if hist.size == 0 or np.all(hist <= 0):
            continue
        j = int(np.argmax(hist))
        level[i] = 0.5 * (edges[j] + edges[j+1])

    s = (c - level) / c
    s = pd.Series(s, index=g.index, dtype=float, name=FEATURE_CODE)
    return s