# JUPYTER CELL — Stage 1: Generate consecutive OHLCV data
import numpy as np
import pandas as pd

def make_consecutive_ohlcv(
    periods=5000,
    start="2022-01-03 09:30",
    freq="5min",
    start_price=100000.0,
    drift_per_bar=0.005,   # mean log-return per bar (upward drift)
    vol_per_bar=0.02,       # std of log-returns per bar (volatility)
    wick_frac=0.5,          # wick size factor relative to body/volatility
    vol_min=100, vol_max=5000,
    seed=2025
) -> pd.DataFrame:
    """Generate a consecutive OHLCV DataFrame where open[t] == close[t-1]."""
    rng = np.random.default_rng(seed)

    # Time index
    idx = pd.date_range(start, periods=periods, freq=freq)

    # Price path via log-returns ~ N(drift, vol)
    rets = rng.normal(loc=drift_per_bar, scale=vol_per_bar, size=periods)
    close = start_price * np.exp(np.cumsum(rets))

    # Consecutive opens
    open_ = np.empty(periods, dtype=float)
    open_[0] = start_price
    open_[1:] = close[:-1]

    # Wicks based on body and volatility
    body = np.abs(close - open_)
    wick_scale = wick_frac * (body + (vol_per_bar * 0.5 * close))
    up_wick = rng.random(periods) * wick_scale
    dn_wick = rng.random(periods) * wick_scale

    high = np.maximum(open_, close) + up_wick
    low  = np.minimum(open_, close) - dn_wick
    low = np.clip(low, 1e-9, None)  # keep prices positive

    # Integer volumes
    volume = rng.integers(low=vol_min, high=vol_max, size=periods)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx
    ).rename(columns=str.lower).sort_index()

    # Sanity: ensure consecutiveness
    assert np.allclose(df["open"].iloc[1:].values, df["close"].iloc[:-1].values), \
        "Consecutiveness failed: open[t] must equal close[t-1]."
    return df

# --- Build your OHLCV dataset (as requested) ---
df = make_consecutive_ohlcv(
    periods=5000,
    start="2022-01-03 09:30",
    freq="5min",
    start_price=100000.0,
    drift_per_bar=0.0005,
    vol_per_bar=0.02,
    wick_frac=0.5,
    vol_min=100, vol_max=5000,
    seed=2025,
)

df.head()