# JUPYTER CELL — feature: liquidity_sweep_wick_ratio_20
FEATURE_CODE = "liquidity_sweep_wick_ratio_20"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Liquidity Sweep Wick Ratio (20-bar lookback)

    Logic:
      Measures what fraction of the wick that sweeps liquidity lies
      outside the prior 20-bar high/low.

      Steps:
        - prior_high_20 = rolling max(high, 20).shift(1)
        - prior_low_20  = rolling min(low,  20).shift(1)

        For each bar:
          up_sweep   = high > prior_high_20
          down_sweep = low  < prior_low_20

          upper_body = max(open, close)
          lower_body = min(open, close)

          wick_above = max(0, high - upper_body)
          wick_below = max(0, lower_body - low)

          outside_up   = max(0, high - prior_high_20)
          outside_down = max(0, prior_low_20 - low)

          ratio_up   = outside_up   / wick_above   if wick_above   > 0
          ratio_down = outside_down / wick_below   if wick_below   > 0

      Feature:
        liquidity_sweep_wick_ratio_20 = clip(ratio_up + ratio_down, 0, 1)
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

    upper_body = np.maximum(open_, close)
    lower_body = np.minimum(open_, close)

    wick_above = (high - upper_body).clip(lower=0.0)
    wick_below = (lower_body - low).clip(lower=0.0)

    up_sweep   = (high > prior_high_20)
    down_sweep = (low  < prior_low_20)

    outside_up   = (high - prior_high_20).where(up_sweep, 0.0).clip(lower=0.0)
    outside_down = (prior_low_20 - low).where(down_sweep, 0.0).clip(lower=0.0)

    ratio_up = np.where(
        wick_above > 0,
        outside_up / wick_above,
        0.0,
    )

    ratio_down = np.where(
        wick_below > 0,
        outside_down / wick_below,
        0.0,
    )

    ratio = ratio_up + ratio_down
    ratio = np.clip(ratio, 0.0, 1.0)

    s = pd.Series(ratio, index=g.index, name=FEATURE_CODE)
    return s