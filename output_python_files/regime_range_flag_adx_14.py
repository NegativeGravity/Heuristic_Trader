# JUPYTER CELL — feature: regime_range_flag_adx_14
FEATURE_CODE = "regime_range_flag_adx_14"

import numpy as np
import pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Regime: Range Flag via ADX (14)
    Description:
      Flags 1 when ADX(14) < 20 (weak trend), suggesting a ranging/sideways regime. Else 0.
    Method:
      Same ADX pipeline as above; only final condition changes to (ADX < 20).
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

    flag = (adx < 20.0).astype(int).fillna(0)
    s = pd.Series(flag.values, index=g.index, name=FEATURE_CODE)
    return s