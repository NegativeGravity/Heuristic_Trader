# JUPYTER CELL — feature: wick_rejection_intensity_10
FEATURE_CODE = "wick_rejection_intensity_10"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Wick Rejection Intensity (10-bar z-score)

    Logic:
      Step 1: For each bar, compute wick size relative to its range.

        upper_body = max(open, close)
        lower_body = min(open, close)

        upper_wick = max(0, high - upper_body)
        lower_wick = max(0, lower_body - low)

        wick_size  = max(upper_wick, lower_wick)
        tr         = high - low

        raw_intensity = wick_size / (tr + eps)

      Step 2: Normalize vs recent history (10-bar window):

        mean_10 = rolling_mean(raw_intensity, 10)
        std_10  = rolling_std(raw_intensity, 10)

        z_score = (raw_intensity - mean_10) / (std_10 + eps)

      Output:
        wick_rejection_intensity_10 = z_score
        Positive values = unusually large wick relative to the last 10 bars.
    """

    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    high  = g["high"].astype(float)
    low   = g["low"].astype(float)
    open_ = g["open"].astype(float)
    close = g["close"].astype(float)

    eps = 1e-9

    upper_body = np.maximum(open_, close)
    lower_body = np.minimum(open_, close)

    upper_wick = (high - upper_body).clip(lower=0.0)
    lower_wick = (lower_body - low).clip(lower=0.0)

    wick_size = np.maximum(upper_wick, lower_wick)
    tr = (high - low).abs()

    raw_intensity = wick_size / (tr + eps)
    raw_intensity = raw_intensity.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    win = 10
    mean_10 = raw_intensity.rolling(win, min_periods=1).mean()
    std_10  = raw_intensity.rolling(win, min_periods=1).std(ddof=0)

    z_score = (raw_intensity - mean_10) / (std_10 + eps)
    z_score = z_score.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return pd.Series(z_score.values, index=g.index, name=FEATURE_CODE)