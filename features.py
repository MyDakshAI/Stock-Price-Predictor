"""
features.py

Turns raw OHLCV price data into a feature set for the model:
technical indicators (moving averages, RSI, MACD, volatility, Bollinger
Bands, ATR, stochastic oscillator, on-balance volume), an optional
market-relative feature (how the stock is doing vs. a benchmark index
like SPY), and the target variable (next day's return / direction).
"""

import pandas as pd
import numpy as np
import ta  # technical analysis indicators library


def load_price_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0)

    # Force the index to real dates and drop any row where that fails
    # (older/newer yfinance versions sometimes leave stray header-like
    # rows in the CSV, e.g. a second row naming the ticker).
    df.index = pd.to_datetime(df.index, errors="coerce", format="mixed")
    df = df[df.index.notna()]

    # Drop any remaining non-numeric rows (same defensive reasoning)
    df = df[pd.to_numeric(df["Close"], errors="coerce").notnull()]
    df = df.astype(float)
    df = df.sort_index()
    return df


def build_features(df: pd.DataFrame, market_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Adds technical indicator columns and a target column to the dataframe.

    Args:
        df: OHLCV dataframe for the ticker being modeled.
        market_df: optional OHLCV dataframe for a benchmark index (e.g.
            SPY), used to add a market-relative feature. A stock's
            next-day direction is heavily influenced by what the broader
            market does, so this is often the single highest-value
            feature to add. Pass None to skip it.

    Target definitions:
        target_return    -> next day's pct change in Close (regression target)
        target_direction  -> 1 if next day's Close is higher, else 0 (classification target)
    """
    out = df.copy()

    # --- Trend / momentum indicators ---
    out["sma_10"] = ta.trend.sma_indicator(out["Close"], window=10)
    out["sma_50"] = ta.trend.sma_indicator(out["Close"], window=50)
    out["ema_10"] = ta.trend.ema_indicator(out["Close"], window=10)
    out["rsi_14"] = ta.momentum.rsi(out["Close"], window=14)

    macd = ta.trend.MACD(out["Close"])
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()

    # --- Bollinger Bands (mean-reversion / volatility context) ---
    bb = ta.volatility.BollingerBands(out["Close"], window=20, window_dev=2)
    out["bb_pct"] = bb.bollinger_pband()  # position within the band, 0-1
    out["bb_width"] = bb.bollinger_wband()  # band width, a volatility proxy

    # --- Average True Range (volatility, accounts for gaps) ---
    out["atr_14"] = ta.volatility.average_true_range(
        out["High"], out["Low"], out["Close"], window=14
    )

    # --- Stochastic oscillator (momentum vs. recent range) ---
    out["stoch_k"] = ta.momentum.stoch(
        out["High"], out["Low"], out["Close"], window=14, smooth_window=3
    )

    # --- On-balance volume (volume-weighted momentum) ---
    obv = ta.volume.OnBalanceVolumeIndicator(out["Close"], out["Volume"])
    out["obv_change"] = obv.on_balance_volume().pct_change()

    # --- Volatility (simple) ---
    out["volatility_10"] = out["Close"].pct_change().rolling(10).std()

    # --- Volume-based ---
    out["volume_change"] = out["Volume"].pct_change()

    # --- Lagged returns (recent momentum) ---
    for lag in [1, 2, 3, 5]:
        out[f"return_lag_{lag}"] = out["Close"].pct_change(lag)

    # --- Market-relative feature ---
    if market_df is not None:
        mkt = market_df[["Close"]].rename(columns={"Close": "mkt_close"})
        out = out.join(mkt, how="left")
        out["mkt_close"] = out["mkt_close"].ffill()
        out["mkt_return_1"] = out["mkt_close"].pct_change(1)
        out["mkt_return_5"] = out["mkt_close"].pct_change(5)
        # how much the stock outperformed/underperformed the market that day
        out["relative_strength"] = out["return_lag_1"] - out["mkt_return_1"]
        out = out.drop(columns=["mkt_close"])

    # --- Targets ---
    out["target_return"] = out["Close"].shift(-1) / out["Close"] - 1
    out["target_direction"] = (out["target_return"] > 0).astype(int)

    out = out.dropna()
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python features.py <path_to_csv>")
        sys.exit(1)

    df = load_price_data(sys.argv[1])
    feats = build_features(df)
    print(feats.tail())
    print(f"\nShape: {feats.shape}")
