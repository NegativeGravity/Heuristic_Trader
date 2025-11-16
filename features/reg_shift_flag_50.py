# JUPYTER CELL — feature: reg_shift_flag_50
FEATURE_CODE = "reg_shift_flag_50"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Regime Shift Flag (50-bar comparison)

    Logic:
      - Compute 50-bar range:    range50 = max(high,50) - min(low,50)
      - Trend component:         trend50 = (close - close.shift(50)) / (range50 + eps)
      - Volatility component:    vol50   = mean(true_range,50) / (mean(close,50) + eps)
      - Regime score:            regime  = trend50 * vol50

      For each bar, compare regime_score with regime_score 50 bars ago:
        - Both magnitudes must be larger than a small threshold.
        - Signs must be different (trend direction flip).
        - Absolute difference must exceed another threshold.

      If all conditions are met → flag = 1, else 0.
    """

    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    high = g["high"].astype(float)
    low  = g["low"].astype(float)
    close = g["close"].astype(float)

    win = 50
    eps = 1e-9

    # 50-bar range
    high50 = high.rolling(win).max()
    low50  = low .rolling(win).min()
    range50 = (high50 - low50).replace(0.0, np.nan)

    # True range approximation
    tr = (high - low).abs()
    tr_mean50 = tr.rolling(win).mean()

    close_mean50 = close.rolling(win).mean()

    trend50 = (close - close.shift(win)) / (range50 + eps)
    vol50   = tr_mean50 / (close_mean50.abs() + eps)

    regime = trend50 * vol50

    # Compare with regime 50 bars ago
    regime_prev = regime.shift(win)

    # Conditions for a regime shift
    mag_thresh = 0.02
    diff_thresh = 0.05

    cond_mag = (regime.abs() > mag_thresh) & (regime_prev.abs() > mag_thresh)
    cond_sign = (np.sign(regime) != np.sign(regime_prev))
    cond_jump = (regime - regime_prev).abs() > diff_thresh

    flag = (cond_mag & cond_sign & cond_jump).astype(int).fillna(0)

    return pd.Series(flag.values, index=g.index, name=FEATURE_CODE)