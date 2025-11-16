# JUPYTER CELL — feature: liquidity_rebuild_speed_20
FEATURE_CODE = "liquidity_rebuild_speed_20"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Liquidity Rebuild Speed (20-bar window)

    Idea:
      Measures how fast price returns toward the center of its 20-bar range.

      Steps:
        1. Compute high20, low20, and mid20 of the last 20 bars.
        2. Normalize distance:
             dist_norm = |close - mid20| / (high20 - low20)
        3. Speed is the reduction in this distance over 5 bars:
             speed = dist_norm.shift(5) - dist_norm

      Positive values → price is reverting back toward liquidity zones.
    """

    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    high = g["high"].astype(float)
    low = g["low"].astype(float)
    close = g["close"].astype(float)

    win = 20
    lag = 5

    high20 = high.rolling(win).max()
    low20  = low .rolling(win).min()
    mid20  = (high20 + low20) / 2
    range20 = (high20 - low20).replace(0.0, np.nan)

    dist_norm = (close - mid20).abs() / range20
    dist_norm = dist_norm.clip(0.0, 1.0)

    speed = dist_norm.shift(lag) - dist_norm
    speed = speed.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return pd.Series(speed, index=g.index, name=FEATURE_CODE)