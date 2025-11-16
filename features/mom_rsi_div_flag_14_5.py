# JUPYTER CELL — feature: mom_rsi_div_flag_14_5
FEATURE_CODE = "mom_rsi_div_flag_14_5"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    RSI Divergence Flag (RSI-14, lookback 5)
    Description:
      Flags simple bullish/bearish divergences between price and RSI:
        +1 (bullish): price makes a lower low vs prior 5 bars while RSI does NOT make a lower low.
        -1 (bearish): price makes a higher high vs prior 5 bars while RSI does NOT make a higher high.
         0 otherwise.
      (This is a lightweight, vectorized approximation of swing-based divergences.)
    Formula / method (brief, cite if needed):
      - RSI(14) via Wilder smoothing:
          gain = max(diff(close), 0), loss = max(-diff(close), 0)
          avg_gain = ewm(gain, alpha=1/14), avg_loss = ewm(loss, alpha=1/14)
          RS = avg_gain/avg_loss; RSI = 100 - 100/(1+RS)
      - With lookback L=5:
          prev_high  = rolling_max(close.shift(1), L)
          prev_low   = rolling_min(close.shift(1), L)
          prev_rsi_hi = rolling_max(RSI.shift(1), L)
          prev_rsi_lo = rolling_min(RSI.shift(1), L)
        bearish_div = (close > prev_high) & (RSI <= prev_rsi_hi - eps)
        bullish_div = (close < prev_low)  & (RSI >= prev_rsi_lo + eps)
      - flag = +1 if bullish_div, -1 if bearish_div, else 0
    Input:
      df: DataFrame with DatetimeIndex (ascending), columns:
           open, high, low, close, volume (case-insensitive)
    Output:
      pd.Series (int), values in {-1,0,+1}, same index as df.index, name == FEATURE_CODE.
      Initial NaNs from rolling windows are OK (mapped to 0 by comparisons).
    Constraints:
      - No look-ahead (all comparisons use shifted/rolling past data).
      - Vectorized; numpy and pandas only.
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    close = g["close"].astype(float)
    delta = close.diff()

    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    alpha = 1.0 / 14.0
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=14).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.clip(0.0, 100.0)

    L = 5
    prev_high   = g["close"].shift(1).rolling(L, min_periods=L).max()
    prev_low    = g["close"].shift(1).rolling(L, min_periods=L).min()
    prev_rsi_hi = rsi.shift(1).rolling(L, min_periods=L).max()
    prev_rsi_lo = rsi.shift(1).rolling(L, min_periods=L).min()

    eps = 0.1  # tiny tolerance
    bearish = (close > prev_high) & (rsi <= (prev_rsi_hi - eps))
    bullish = (close < prev_low)  & (rsi >= (prev_rsi_lo + eps))

    flag = np.where(bullish, 1, np.where(bearish, -1, 0)).astype(int)
    s = pd.Series(flag, index=g.index, name=FEATURE_CODE)
    return s