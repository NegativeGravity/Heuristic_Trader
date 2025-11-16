# JUPYTER CELL — feature: swing_leg_efficiency_ratio_30
FEATURE_CODE = "swing_leg_efficiency_ratio_30"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Swing Leg Efficiency Ratio (30-bar window)

    Logic:
      For each bar t:

        net_move   = |close_t - close_{t-30}|
        path_sum   = sum_{i=t-29..t} |close_i - close_{i-1}|
        efficiency = net_move / (path_sum + eps)

      Interpretation:
        - Values near 1 → very directional leg (trend-like).
        - Values near 0 → highly choppy, mean-reverting movement.
    """

    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    close = g["close"].astype(float)

    win = 30
    eps = 1e-9

    # Net move over 30 bars
    net_move = (close - close.shift(win)).abs()

    # Step-wise absolute changes
    step_change = close.diff().abs()

    # Rolling sum of absolute changes over 30 bars
    path_sum = step_change.rolling(win, min_periods=1).sum()

    efficiency = net_move / (path_sum + eps)
    efficiency = efficiency.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return pd.Series(efficiency.values, index=g.index, name=FEATURE_CODE)