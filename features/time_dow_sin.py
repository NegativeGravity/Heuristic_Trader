# JUPYTER CELL — feature: time_dow_sin
FEATURE_CODE = "time_dow_sin"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Time-of-Week Sine Encoding (Day of Week)

    Requirements:
      - Index must be a DatetimeIndex.

    Logic:
      dow  = day_of_week in {0..6}
      feat = sin(2π * dow / 7)

      This gives a smooth cyclical representation of day-of-week
      (useful for models that like continuous features).
    """

    g = df.copy()

    if not isinstance(g.index, pd.DatetimeIndex):
        raise ValueError("time_dow_sin requires a DatetimeIndex.")

    dow = g.index.dayofweek.astype(float)  # Monday=0, Sunday=6

    values = np.sin(2.0 * np.pi * dow / 7.0)

    return pd.Series(values, index=g.index, name=FEATURE_CODE)