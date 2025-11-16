# JUPYTER CELL — feature: regime_trend_up_flag_adx_14
FEATURE_CODE = "regime_trend_up_flag_adx_14"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Regime: Uptrend Flag via ADX (14)
    Description:
      Flags 1 when ADX(14) >= 20 and +DI > -DI (trend strength + positive direction). Else 0.
    Method (Wilder approximations using EWM as proxy):
      +DM = max(high_t - high_{t-1}, 0) if > (low_{t-1} - low_t) else 0
      -DM = max(low_{t-1} - low_t, 0)  if > (high_t - high_{t-1}) else 0
      TR  = max(high-low, |high-close_{t-1}|, |low-close_{t-1}|)
      Smooth with EWM(alpha=1/14, adjust=False, min_periods=14)
      +DI = 100 * (+DM_sm / TR_sm); -DI = 100 * (-DM_sm / TR_sm)
      DX  = 100 * |(+DI - -DI)| / (+DI + -DI)
      ADX = EWM(DX, alpha=1/14, adjust=False, min_periods=14)
      flag = 1 if (ADX>=20) & (+DI > -DI) else 0
    """
    g = df.copy(); g.columns = [str(c).lower() for c in g.columns]
    h, l, c = g["high"].astype(float), g["low"].astype(float), g["close"].astype(float)

    up_move   = h.diff()
    down_move = -l.diff()

    plus_dm  = np.where((up_move > down_move) & (up_move > 0),  up_move,  0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm  = pd.Series(plus_dm, index=g.index)
    minus_dm = pd.Series(minus_dm, index=g.index)

    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)

    alpha = 1/14
    tr_sm     = tr.ewm(alpha=alpha, adjust=False, min_periods=14).mean()
    plus_sm   = plus_dm.ewm(alpha=alpha, adjust=False, min_periods=14).mean()
    minus_sm  = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=14).mean()

    plus_di  = 100.0 * (plus_sm  / tr_sm.replace(0.0, np.nan))
    minus_di = 100.0 * (minus_sm / tr_sm.replace(0.0, np.nan))

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False, min_periods=14).mean()

    flag = ((adx >= 20.0) & (plus_di > minus_di)).astype(int).fillna(0)
    s = pd.Series(flag.values, index=g.index, name=FEATURE_CODE)
    return s