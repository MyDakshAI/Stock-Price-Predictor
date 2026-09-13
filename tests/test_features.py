"""
Tests for feature engineering.

The most important test in this file is test_no_lookahead_in_features:
a feature that accidentally peeks at the future is the single most
common way these projects produce impressive, meaningless results.
"""

import numpy as np
import pandas as pd
import pytest

from features import build_features, load_price_data


FEATURE_PREFIXES = (
    "sma_", "ema_", "rsi_", "macd", "bb_", "atr_", "stoch_", "obv_",
    "volatility_", "volume_change", "return_lag_",
)


def test_build_features_adds_expected_columns(price_df):
    out = build_features(price_df)
    for col in ["sma_10", "sma_50", "ema_10", "rsi_14", "macd", "macd_signal",
                "bb_pct", "bb_width", "atr_14", "stoch_k", "obv_change",
                "volatility_10", "volume_change", "return_lag_1",
                "target_return", "target_direction"]:
        assert col in out.columns, f"missing feature column: {col}"


def test_no_nans_after_build(price_df):
    out = build_features(price_df)
    assert not out.isna().any().any(), "features should be fully dropna'd"


def test_target_direction_matches_target_return(price_df):
    out = build_features(price_df)
    expected = (out["target_return"] > 0).astype(int)
    pd.testing.assert_series_equal(out["target_direction"], expected, check_names=False)


def test_target_return_is_next_day_not_same_day(price_df):
    """target_return must describe TOMORROW's move. If it described today's,
    the model would be predicting something it can already see."""
    out = build_features(price_df)
    close = price_df["Close"]
    row_date = out.index[10]
    pos = close.index.get_loc(row_date)
    expected = close.iloc[pos + 1] / close.iloc[pos] - 1
    assert out["target_return"].iloc[10] == pytest.approx(expected, rel=1e-9)


def test_no_lookahead_in_features(price_df):
    """
    Truncating the price history must not change the feature values
    computed for earlier dates. If it does, some feature is reaching
    forward in time, which is the classic silent bug that makes a
    backtest look great and be worthless.
    """
    full = build_features(price_df)
    truncated = build_features(price_df.iloc[:-50])

    shared_dates = truncated.index.intersection(full.index)
    assert len(shared_dates) > 100, "not enough overlap to make the test meaningful"

    feature_cols = [c for c in truncated.columns
                    if c.startswith(FEATURE_PREFIXES)]

    for col in feature_cols:
        np.testing.assert_allclose(
            full.loc[shared_dates, col].values,
            truncated.loc[shared_dates, col].values,
            rtol=1e-9, atol=1e-12,
            err_msg=f"feature '{col}' changed when future data was removed (look-ahead leak)",
        )


def test_market_features_added_only_when_requested(price_df, market_df):
    without = build_features(price_df)
    with_market = build_features(price_df, market_df=market_df)

    assert "relative_strength" not in without.columns
    assert "relative_strength" in with_market.columns
    assert "mkt_return_1" in with_market.columns


def test_load_price_data_drops_junk_rows(tmp_path, price_df):
    """yfinance sometimes writes a stray ticker-name row under the header;
    load_price_data should quietly drop rows like that."""
    csv = tmp_path / "TEST.csv"
    price_df.to_csv(csv)

    raw = csv.read_text().splitlines()
    raw.insert(1, "Ticker,TEST,TEST,TEST,TEST,TEST")
    csv.write_text("\n".join(raw))

    out = load_price_data(str(csv))
    assert len(out) == len(price_df)
    assert out.index.is_monotonic_increasing
    assert out["Close"].dtype == float
