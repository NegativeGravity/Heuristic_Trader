# JUPYTER CELL — feature: candle_engulf_strength_5
FEATURE_CODE = "candle_engulf_strength_5"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Candle Engulf Strength (5)
    Measures strength of engulfing patterns over a 5-bar context:
      - True engulf if body direction flips and current body fully contains previous body.
      - Strength = current body / max body in last 5 bars (0..1).
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    o = g["open"].astype(float)
    c = g["close"].astype(float)
    h = g["high"].astype(float)
    l = g["low"].astype(float)

    body = (c - o).abs()
    prev_body = body.shift(1)
    dir_curr = np.sign(c - o)
    dir_prev = np.sign(c.shift(1) - o.shift(1))

    engulf_range = (h >= h.shift(1)) & (l <= l.shift(1))
    opposite_dir = (dir_curr * dir_prev) < 0
    bigger_body = body > prev_body

    engulf_flag = (engulf_range & opposite_dir & bigger_body).astype(int)
    max_body_5 = body.rolling(5, min_periods=1).max()

    strength = np.where(engulf_flag == 1, body / max_body_5.replace(0.0, np.nan), 0.0)
    s = pd.Series(strength, index=g.index, name=FEATURE_CODE).astype(float)
    return s