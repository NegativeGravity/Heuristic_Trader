# JUPYTER CELL — feature: orderblock_freshness_score_50
FEATURE_CODE = "orderblock_freshness_score_50"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Orderblock Freshness Score (50-bar proxy)

    Idea:
      A simple proxy for "freshness" of orderblock-like zones.

      Steps:
        - Compute 50-bar high and 50-bar low.
        - Compute the candle’s distance to the nearest extreme (high50 or low50).
        - Normalize the distance by the 50-bar range.
        - Score = 1 - normalized_distance (clamped to [0,1])

      The closer price is to a recent extreme, the "fresher" the zone.
    """

    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    high = g["high"].astype(float)
    low = g["low"].astype(float)
    close = g["close"].astype(float)

    win = 50

    high50 = high.rolling(win).max()
    low50  = low .rolling(win).min()
    range50 = (high50 - low50).replace(0.0, np.nan)

    dist_high = (close - high50).abs()
    dist_low  = (close - low50).abs()

    nearest_dist = pd.concat([dist_high, dist_low], axis=1).min(axis=1)

    base = nearest_dist / range50
    score = (1.0 - base).clip(0.0, 1.0).fillna(0.0)

    return pd.Series(score, index=g.index, name=FEATURE_CODE)