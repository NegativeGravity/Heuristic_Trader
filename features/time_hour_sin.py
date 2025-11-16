# JUPYTER CELL — feature: time_hour_sin
FEATURE_CODE = "time_hour_sin"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Time-of-Day Sine Encoding (Hour of Day)

    Requirements:
      - Index must be a DatetimeIndex.
      - Data is intraday (or at least has hour information).

    Logic:
      hour = index.hour in {0..23}
      feat = sin(2π * hour / 24)

      This encodes time-of-day as a smooth cyclic feature.
    """

    g = df.copy()

    if not isinstance(g.index, pd.DatetimeIndex):
        raise ValueError("time_hour_sin requires a DatetimeIndex.")

    hour = g.index.hour.astype(float)
    values = np.sin(2.0 * np.pi * hour / 24.0)

    return pd.Series(values, index=g.index, name=FEATURE_CODE)