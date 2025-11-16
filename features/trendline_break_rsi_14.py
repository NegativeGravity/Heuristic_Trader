# JUPYTER CELL — feature: trendline_break_rsi_14
FEATURE_CODE = "trendline_break_rsi_14"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    RSI Trendline Breakout Flag (14)
    Description:
      Flags 1 if RSI(14) breaks through its trendline, indicating a breakout.
    Formula / method (brief):
      - Calculate the 14-period RSI.
      - Detect trendline breakout (RSI crosses its rolling mean).
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    rsi_14 = g["close"].rolling(14).apply(lambda x: 100 - (100 / (1 + (x.diff().clip(0).mean() / x.diff().clip(None).mean()))))

    rsi_trendline = rsi_14.rolling(14).mean()

    breakout = (rsi_14 > rsi_trendline).astype(int)

    s = pd.Series(breakout, index=g.index, name=FEATURE_CODE)
    return s