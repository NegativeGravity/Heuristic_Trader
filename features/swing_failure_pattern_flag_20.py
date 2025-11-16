# JUPYTER CELL — feature: swing_failure_pattern_flag_20
FEATURE_CODE = "swing_failure_pattern_flag_20"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Swing Failure Pattern Flag (20-bar lookback)

    Logic:
      Uses prior 20-bar extremes as liquidity reference.

      - prior_high_20 = rolling max(high, 20).shift(1)
      - prior_low_20  = rolling min(low,  20).shift(1)

      Bearish SFP (of a high):
        - high > prior_high_20       (sweep the prior high)
        - close < prior_high_20      (close back below the level)
        - optional: close < open     (bearish close)

      Bullish SFP (of a low):
        - low < prior_low_20         (sweep the prior low)
        - close > prior_low_20       (close back above the level)
        - optional: close > open     (bullish close)

      If either bullish or bearish SFP occurs → flag = 1, else 0.
    """

    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    high  = g["high"].astype(float)
    low   = g["low"].astype(float)
    open_ = g["open"].astype(float)
    close = g["close"].astype(float)

    lookback = 20

    prior_high_20 = high.rolling(lookback).max().shift(1)
    prior_low_20  = low .rolling(lookback).min().shift(1)

    # Bearish SFP: sweep above prior high, close back below it, bearish candle
    bearish_sfp = (
        (high > prior_high_20) &
        (close < prior_high_20) &
        (close < open_)
    )

    # Bullish SFP: sweep below prior low, close back above it, bullish candle
    bullish_sfp = (
        (low < prior_low_20) &
        (close > prior_low_20) &
        (close > open_)
    )

    flag = (bearish_sfp | bullish_sfp).astype(int).fillna(0)

    return pd.Series(flag.values, index=g.index, name=FEATURE_CODE)