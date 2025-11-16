# JUPYTER CELL — feature: session_high_low_shift_dir_3d
FEATURE_CODE = "session_high_low_shift_dir_3d"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Session High–Low Shift Direction (3-day window)

    Requirements:
      - Index must be a DatetimeIndex.
      - Data is assumed intraday with multiple bars per calendar day.

    Logic:
      For each calendar day:
        - high_d = max(high) of the day
        - low_d  = min(low) of the day

      Then:
        - 1-day shift score:
            shift_score_d = ((high_d - high_d.shift(1)) +
                             (low_d  - low_d.shift(1))) / 2
        - 3-day smoothed shift:
            shift_mean3 = rolling_mean(shift_score_d, 3)
        - Direction:
            dir_d = sign(shift_mean3)   in {-1, 0, +1}

      That daily direction is broadcast to all intraday bars of the same day.
    """

    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    if not isinstance(g.index, pd.DatetimeIndex):
        raise ValueError("session_high_low_shift_dir_3d requires a DatetimeIndex.")

    high = g["high"].astype(float)
    low  = g["low"].astype(float)

    dates = g.index.normalize()

    temp = pd.DataFrame({"high": high, "low": low, "date": dates})

    grouped = temp.groupby("date", sort=False)
    high_d = grouped["high"].max()
    low_d  = grouped["low"].min()

    # 1-day shift in highs and lows
    shift_score = ((high_d - high_d.shift(1)) +
                   (low_d  - low_d.shift(1))) / 2.0

    # 3-day smoothed direction
    shift_mean3 = shift_score.rolling(3, min_periods=1).mean()
    dir_d = np.sign(shift_mean3).fillna(0.0)

    dir_series = pd.Series(dir_d.reindex(dates).values, index=g.index)

    return dir_series.rename(FEATURE_CODE)