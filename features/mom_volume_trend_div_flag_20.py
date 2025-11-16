# JUPYTER CELL — feature: mom_volume_trend_div_flag_20
FEATURE_CODE = "mom_volume_trend_div_flag_20"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Volume and Trend Divergence Flag (20)
    Description:
      Flags if there is a divergence between price trend and volume trend.
      Price goes up/down while volume behaves oppositely, indicating a divergence.
    Formula / method (brief):
      - Calculate rolling mean of close and volume over 20 periods.
      - Flag 1 if price and volume trends diverge (one goes up, the other down).
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    price_rolling_mean = g["close"].rolling(20).mean()
    volume_rolling_mean = g["volume"].rolling(20).mean()

    # Identify divergence
    price_up = g["close"] > price_rolling_mean
    volume_up = g["volume"] > volume_rolling_mean

    divergence = (price_up != volume_up).astype(int)

    s = pd.Series(divergence, index=g.index, name=FEATURE_CODE)
    return s