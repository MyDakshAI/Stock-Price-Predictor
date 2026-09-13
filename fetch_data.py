"""
fetch_data.py

Downloads historical daily price data for a given ticker using yfinance
and saves it to data/{ticker}.csv. Caches by default: if a CSV for that
ticker was already downloaded today, reuses it instead of re-hitting
yfinance (pass force=True to bypass).

Usage:
    python fetch_data.py AAPL
    python fetch_data.py AAPL --years 5
"""

import argparse
import os
import time
import yfinance as yf

CACHE_MAX_AGE_SECONDS = 24 * 60 * 60  # 1 day


def fetch(ticker: str, years: int = 5, data_dir: str = "data", force: bool = False) -> str:
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, f"{ticker.upper()}.csv")

    if not force and os.path.exists(out_path):
        age = time.time() - os.path.getmtime(out_path)
        if age < CACHE_MAX_AGE_SECONDS:
            print(f"[*] Using cached data for {ticker} ({out_path}, "
                  f"{age/3600:.1f}h old)")
            return out_path

    period = f"{years}y"
    print(f"[*] Downloading {ticker} data for the last {years} year(s)...")

    df = yf.download(ticker, period=period, interval="1d", progress=False)

    if df.empty:
        raise ValueError(
            f"No data returned for '{ticker}'. Check the ticker symbol is correct."
        )

    df.to_csv(out_path)

    print(f"[+] Saved {len(df)} rows to {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download historical stock data.")
    parser.add_argument("ticker", help="Stock ticker symbol, e.g. AAPL, MSFT, TSLA")
    parser.add_argument("--years", type=int, default=5, help="Years of history to fetch")
    parser.add_argument("--force", action="store_true", help="Bypass the cache and re-download")
    args = parser.parse_args()

    fetch(args.ticker, args.years, force=args.force)
