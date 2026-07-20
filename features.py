"""
features.py

Turns raw OHLCV price data into a feature set for the model:
technical indicators (moving averages, RSI, MACD, volatility) plus
the target variable (next day's return / direction).
"""

import pandas as pd
import numpy as np
import ta  # technical analysis indicators library


def load_price_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    # yfinance sometimes writes a duplicate header row when saved/reloaded; guard against it
    df = df[pd.to_numeric(df["Close"], errors="coerce").notnull()]
    df = df.astype(float)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicator columns and a target column to the dataframe.

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

    # --- Volatility ---
    out["volatility_10"] = out["Close"].pct_change().rolling(10).std()

    # --- Volume-based ---
    out["volume_change"] = out["Volume"].pct_change()

    # --- Lagged returns (recent momentum) ---
    for lag in [1, 2, 3, 5]:
        out[f"return_lag_{lag}"] = out["Close"].pct_change(lag)

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
