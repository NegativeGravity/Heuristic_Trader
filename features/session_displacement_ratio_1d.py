# JUPYTER CELL — feature: session_displacement_ratio_1d
FEATURE_CODE = "session_displacement_ratio_1d"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Session Displacement Ratio (1-day)

    Requirements:
      - Index must be a DatetimeIndex.
      - Data is assumed intraday, with multiple bars per calendar day.

    Logic:
      For each calendar day:
        - open_d  = first close of the day (or open if you replace it)
        - close_d = last close of the day
        - high_d  = max high of the day
        - low_d   = min low of the day
        - displacement = close_d - open_d
        - range       = high_d - low_d

      Session-level metric:
        ratio_d = displacement / (range + eps)

      That daily ratio is then mapped back to all bars of the same day.
    """

    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    if not isinstance(g.index, pd.DatetimeIndex):
        raise ValueError("session_displacement_ratio_1d requires a DatetimeIndex.")

    high = g["high"].astype(float)
    low  = g["low"].astype(float)
    close = g["close"].astype(float)

    dates = g.index.normalize()

    temp = pd.DataFrame({
        "high": high,
        "low": low,
        "close": close,
        "date": dates,
    })

    grouped = temp.groupby("date", sort=False)

    high_d  = grouped["high"].max()
    low_d   = grouped["low"].min()
    close_d = grouped["close"].last()
    open_d  = grouped["close"].first()  # if you have 'open', swap to grouped["open"].first()

    displacement = close_d - open_d
    session_range = (high_d - low_d).replace(0.0, np.nan)

    ratio_d = displacement / (session_range + 1e-9)
    ratio_d = ratio_d.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Broadcast daily values to intraday index
    session_ratio = pd.Series(ratio_d.reindex(dates).values, index=g.index)

    return session_ratio.rename(FEATURE_CODE)