# JUPYTER CELL — feature: session_killzone_activity_index
FEATURE_CODE = "session_killzone_activity_index"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Session Killzone Activity Index (1-day)

    Clean, vectorized, and warning-free version.
    """

    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    if not isinstance(g.index, pd.DatetimeIndex):
        raise ValueError("session_killzone_activity_index requires a DatetimeIndex.")

    high = g["high"].astype(float)
    low  = g["low"].astype(float)

    tr = (high - low).abs()

    dates = g.index.normalize()
    hours = g.index.hour

    killzone_mask = (
        ((hours >= 7) & (hours < 10)) |   # London
        ((hours >= 13) & (hours < 16))    # New York
    )

    temp = pd.DataFrame({
        "tr": tr,
        "date": dates,
        "killzone": killzone_mask.astype(bool),
    })

    grouped = temp.groupby("date", sort=False)

    # Daily TR sum (vectorized)
    daily_tr_sum = grouped["tr"].sum()

    # Killzone TR sum (vectorized & no warning)
    kill_tr_sum = temp.loc[temp["killzone"], :].groupby("date")["tr"].sum()
    kill_tr_sum = kill_tr_sum.reindex(daily_tr_sum.index).fillna(0.0)

    eps = 1e-9
    activity_d = kill_tr_sum / (daily_tr_sum + eps)
    activity_d = activity_d.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Broadcast to full index
    activity_series = pd.Series(activity_d.reindex(dates).values, index=g.index)

    return activity_series.rename(FEATURE_CODE)