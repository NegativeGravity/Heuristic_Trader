# JUPYTER CELL — feature: range_rotation_index_20
FEATURE_CODE = "range_rotation_index_20"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Range Rotation Index (20-bar window)

    Logic:
      - Compute 20-bar rolling high and low.
      - mid20 = (high20 + low20) / 2
      - For each bar:
          delta_mid = mid20 - mid20.shift(1)
          sign_rot  = sign(delta_mid) in {-1, 0, +1}
      - Feature = rolling mean of sign_rot over 20 bars.

      Interpretation:
        +1 → midpoint mostly rotating upward
        -1 → midpoint mostly rotating downward
         0 → balanced / choppy.
    """

    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    high = g["high"].astype(float)
    low = g["low"].astype(float)

    win = 20

    high20 = high.rolling(win).max()
    low20  = low .rolling(win).min()
    mid20  = (high20 + low20) / 2.0

    delta_mid = mid20 - mid20.shift(1)

    sign_rot = np.sign(delta_mid).fillna(0.0)

    rotation_index = sign_rot.rolling(win, min_periods=1).mean()
    rotation_index = rotation_index.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return pd.Series(rotation_index.values, index=g.index, name=FEATURE_CODE)