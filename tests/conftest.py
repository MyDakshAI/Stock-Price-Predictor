"""
Shared fixtures.

Tests run on synthetic price data, never on live yfinance downloads, so
the suite is fast, deterministic, and works offline in CI.
"""

import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(n_days: int = 500, seed: int = 0, drift: float = 0.0005) -> pd.DataFrame:
    """Generates a plausible random-walk OHLCV series."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, 0.015, n_days)
    close = 100 * np.exp(np.cumsum(returns))

    high = close * (1 + np.abs(rng.normal(0, 0.006, n_days)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, n_days)))
    open_ = close * (1 + rng.normal(0, 0.004, n_days))
    volume = rng.integers(1_000_000, 5_000_000, n_days).astype(float)

    index = pd.bdate_range("2018-01-01", periods=n_days)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )


@pytest.fixture
def price_df():
    return _make_ohlcv()


@pytest.fixture
def market_df():
    return _make_ohlcv(seed=42)


@pytest.fixture
def feats(price_df):
    from features import build_features
    return build_features(price_df)
