# JUPYTER CELL — feature: pivot_confluence_score_1d
FEATURE_CODE = "pivot_confluence_score_1d"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Pivot Confluence Score (classic pivots from previous day)

    Requirements:
        Index must be a DatetimeIndex.

    Logic:
        - Compute previous day's OHLC.
        - Compute classic pivot levels:
             PP, R1, S1, R2, S2
        - For each intraday bar, compute normalized distance:
             dist_norm = |close - level| / (prev_day_range)
        - Confluence score = Σ exp(-alpha * dist_norm)
          (Higher score = stronger confluence around pivots)

    Output is continuous, usually between 0 and ~5.
    """

    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    if not isinstance(g.index, pd.DatetimeIndex):
        raise ValueError("pivot_confluence_score_1d requires a DatetimeIndex.")

    high = g["high"].astype(float)
    low = g["low"].astype(float)
    close = g["close"].astype(float)

    # Extract date for grouping
    dates = g.index.normalize()

    # Daily OHLC
    daily = pd.DataFrame({"high": high, "low": low, "close": close})
    daily_ohlc = daily.groupby(dates).agg({"high": "max", "low": "min", "close": "last"})

    prev = daily_ohlc.shift(1)

    prev_high  = prev["high"].reindex(dates).values
    prev_low   = prev["low"] .reindex(dates).values
    prev_close = prev["close"].reindex(dates).values

    prev_high  = pd.Series(prev_high,  index=g.index)
    prev_low   = pd.Series(prev_low,   index=g.index)
    prev_close = pd.Series(prev_close, index=g.index)

    prev_range = (prev_high - prev_low).replace(0.0, np.nan)

    PP = (prev_high + prev_low + prev_close) / 3
    R1 = 2*PP - prev_low
    S1 = 2*PP - prev_high
    R2 = PP + (prev_high - prev_low)
    S2 = PP - (prev_high - prev_low)

    def norm_dist(level):
        d = (close - level).abs() / prev_range
        return d.replace([np.inf, -np.inf], np.nan)

    dist_levels = [
        norm_dist(PP),
        norm_dist(R1),
        norm_dist(S1),
        norm_dist(R2),
        norm_dist(S2),
    ]

    alpha = 5.0

    score = sum(np.exp(-alpha * d.fillna(99)) for d in dist_levels)
    score = pd.Series(score, index=g.index).fillna(0.0)

    return score.rename(FEATURE_CODE)