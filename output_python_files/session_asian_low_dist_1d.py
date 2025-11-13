# JUPYTER CELL — feature: session_asian_low_dist_1d
FEATURE_CODE = "session_asian_low_dist_1d"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Session Asian Low Distance (1d)
    Description:
      Flags 1 if the close is close to the low of the previous day's Asian session.
      Proximity is determined within a small range (ε = 0.01).
    Formula / method (brief):
      asian_low = low of previous day's Asian session
      flag = 1 if abs(close_t - asian_low) / close_t <= ε else 0
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    # Assuming Asian session low is the low of the first few hours of the day,
    # we'll calculate it from the first 4 hours for this example.
    asian_session = g.between_time('00:00', '04:00')  # Adjust based on Asian session time
    asian_low = asian_session["low"].min()

    flag = (abs(g["close"] - asian_low) / g["close"] <= 0.01).astype(int)

    s = pd.Series(flag, index=g.index, name=FEATURE_CODE)
    return s