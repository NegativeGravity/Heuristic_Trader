# JUPYTER CELL — feature: fvg_fill_ratio_30
FEATURE_CODE = "fvg_fill_ratio_30"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    FVG Fill Ratio (30-bar lifetime, single active gap)

    Logic:
      - Detect 3-bar Fair Value Gaps (FVG) using:
          Bullish FVG at bar n:
            low_n > high_{n-2}
            gap range: [gap_low, gap_high] = [high_{n-2}, low_n]

          Bearish FVG at bar n:
            high_n < low_{n-2}
            gap range: [gap_low, gap_high] = [high_n, low_{n-2}]

      - Track at most one active FVG at a time:
          - gap_low, gap_high, gap_start_idx
          - covered_low, covered_high = union of portions of gap that have traded

      - For each bar t:
          1) If an active gap exists:
               - Compute overlap between [low_t, high_t] and [gap_low, gap_high].
               - Expand (covered_low, covered_high) by this overlap.
               - Compute:
                   gap_size      = gap_high - gap_low
                   covered_size  = max(0, covered_high - covered_low)
                   fill_ratio_t  = covered_size / gap_size
               - If fully filled OR t - gap_start_idx >= 30 → expire gap.
          2) If no active gap exists, check if a new FVG starts at t (using bars t, t-1, t-2).

      Output:
        fill_ratio_t in [0, 1] for the currently tracked FVG up to time t.
    """

    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    high = g["high"].astype(float).values
    low  = g["low"].astype(float).values

    n = len(g)

    fill_ratio = np.zeros(n, dtype=float)

    current_gap_low = np.nan
    current_gap_high = np.nan
    covered_low = np.nan
    covered_high = np.nan
    gap_start_idx = None
    max_lifetime = 30

    for i in range(n):
        h = high[i]
        l = low[i]

        # 1) If there is an active gap, update fill
        if not np.isnan(current_gap_low):
            # Overlap between current candle and gap
            overlap_low = max(l, current_gap_low)
            overlap_high = min(h, current_gap_high)

            if overlap_high > overlap_low:
                if np.isnan(covered_low):
                    covered_low = overlap_low
                    covered_high = overlap_high
                else:
                    covered_low = min(covered_low, overlap_low)
                    covered_high = max(covered_high, overlap_high)

            gap_size = current_gap_high - current_gap_low
            if gap_size > 0:
                if np.isnan(covered_low):
                    covered_size = 0.0
                else:
                    covered_size = max(0.0, covered_high - covered_low)
                fill_ratio[i] = covered_size / gap_size
            else:
                fill_ratio[i] = 0.0

            # Expire if fully filled or too old
            if gap_start_idx is not None:
                if (gap_size <= 0) or (covered_size >= gap_size) or (i - gap_start_idx >= max_lifetime):
                    current_gap_low = np.nan
                    current_gap_high = np.nan
                    covered_low = np.nan
                    covered_high = np.nan
                    gap_start_idx = None

        # 2) If no active gap, check for a new FVG at this bar
        if np.isnan(current_gap_low) and i >= 2:
            h_2 = high[i - 2]
            l_2 = low[i - 2]

            # Bullish FVG (gap above bar n-2)
            if l > h_2:
                current_gap_low = h_2
                current_gap_high = l
                gap_start_idx = i
                covered_low = np.nan
                covered_high = np.nan

            # Bearish FVG (gap below bar n-2)
            elif h < l_2:
                current_gap_low = h
                current_gap_high = l_2
                gap_start_idx = i
                covered_low = np.nan
                covered_high = np.nan

        # If no gap, fill_ratio[i] stays as previous (default 0.0)

    s = pd.Series(fill_ratio, index=g.index, name=FEATURE_CODE)
    return s