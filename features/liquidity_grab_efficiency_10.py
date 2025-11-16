# JUPYTER CELL — feature: liquidity_grab_efficiency_10
FEATURE_CODE = "liquidity_grab_efficiency_10"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Liquidity Grab Efficiency (10-bar lookback)

    Logic:
      Detects liquidity grabs (sweeps) above/below prior highs/lows.
        - Upward grab: current high > max(high[1..10]) AND close < previous max
        - Downward grab: current low < min(low[1..10]) AND close > previous min

      Efficiency is measured as:
          wick_outside_range / full_candle_range

      The output is between 0 and 1.
      Zero means no liquidity grab or no efficiency.
    """

    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    high = g["high"].astype(float)
    low = g["low"].astype(float)
    close = g["close"].astype(float)
    open_ = g["open"].astype(float)

    lookback = 10

    # Prior N-bar extremes (shifted to avoid using current bar)
    prior_high = high.rolling(lookback, min_periods=lookback).max().shift(1)
    prior_low  = low .rolling(lookback, min_periods=lookback).min().shift(1)

    # Conditions for sweeps
    up_grab   = (high > prior_high) & (close < prior_high)
    down_grab = (low  < prior_low ) & (close > prior_low)

    # Candle range
    tr = (high - low).replace(0.0, np.nan)

    upper_body = np.maximum(open_, close)
    lower_body = np.minimum(open_, close)

    wick_above = (high - upper_body).clip(lower=0.0)
    wick_below = (lower_body - low).clip(lower=0.0)

    eff_up   = np.where(up_grab,   wick_above / tr, 0.0)
    eff_down = np.where(down_grab, wick_below / tr, 0.0)

    eff = np.nan_to_num(eff_up + eff_down)

    return pd.Series(eff, index=g.index, name=FEATURE_CODE)