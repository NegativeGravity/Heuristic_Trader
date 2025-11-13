# JUPYTER CELL — feature: liq_zone_strength_50
FEATURE_CODE = "liq_zone_strength_50"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Liquidity Zone Strength (50)
    Description:
      Measures the relative liquidity strength (volume density) within a 50-bar window.
      Higher liquidity strength is associated with higher trading volumes within a given range.
    Formula / method (brief):
      - liquidity_strength = sum(volume within range) / (high - low) over the last 50 bars
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    high = g["high"]
    low = g["low"]
    volume = g["volume"]

    liquidity_strength = (volume * (high - low)).rolling(50, min_periods=50).sum() / (high - low).rolling(50, min_periods=50).sum()

    s = liquidity_strength.astype(float)
    s.name = FEATURE_CODE
    return s