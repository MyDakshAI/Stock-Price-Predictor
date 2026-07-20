"""
fetch_data.py

Downloads historical daily price data for a given ticker using yfinance
and saves it to data/{ticker}.csv.

Usage:
    python fetch_data.py AAPL
    python fetch_data.py AAPL --years 5
"""

import argparse
import os
import yfinance as yf


def fetch(ticker: str, years: int = 5) -> str:
    period = f"{years}y"
    print(f"[*] Downloading {ticker} data for the last {years} year(s)...")

    df = yf.download(ticker, period=period, interval="1d", progress=False)

    if df.empty:
        raise ValueError(
            f"No data returned for '{ticker}'. Check the ticker symbol is correct."
        )

    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", f"{ticker.upper()}.csv")
    df.to_csv(out_path)

    print(f"[+] Saved {len(df)} rows to {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download historical stock data.")
    parser.add_argument("ticker", help="Stock ticker symbol, e.g. AAPL, MSFT, TSLA")
    parser.add_argument("--years", type=int, default=5, help="Years of history to fetch")
    args = parser.parse_args()

    fetch(args.ticker, args.years)
