"""
main.py

End-to-end pipeline:
    1. Download historical price data for a ticker
    2. Build technical indicator features
    3. Split chronologically into train/test
    4. Train a model to predict next-day direction
    5. Backtest the model's predictions on the held-out test set
    6. Print results and save charts

Usage:
    python main.py AAPL
    python main.py TSLA --years 8 --test-frac 0.2 --threshold 0.55
"""

import argparse

from fetch_data import fetch
from features import load_price_data, build_features
from model import (
    train_test_split_chronological, train_model, predict,
    save_model, feature_importance_report
)
from backtest import run_backtest, summarize_backtest, print_summary
from plot_results import plot_backtest, plot_feature_importance


def run(ticker: str, years: int = 5, test_frac: float = 0.2, threshold: float = 0.5):
    ticker = ticker.upper()

    # 1. Data
    csv_path = fetch(ticker, years)

    # 2. Features
    df = load_price_data(csv_path)
    feats = build_features(df)
    print(f"[*] Built {feats.shape[1]} features across {feats.shape[0]} trading days")

    # 3. Chronological split (never shuffle time series data!)
    train_df, test_df = train_test_split_chronological(feats, test_frac)
    print(f"[*] Train: {len(train_df)} days ({train_df.index[0].date()} to {train_df.index[-1].date()})")
    print(f"[*] Test:  {len(test_df)} days ({test_df.index[0].date()} to {test_df.index[-1].date()})")

    # 4. Train
    print("[*] Training model...")
    model, scaler = train_model(train_df)
    save_model(model, scaler, ticker)

    # 5. Predict + backtest on test set only (out-of-sample)
    predicted_up_prob = predict(model, scaler, test_df)
    results = run_backtest(test_df, predicted_up_prob, threshold=threshold)
    summary = summarize_backtest(results)
    print_summary(summary, ticker)

    # 6. Charts
    plot_backtest(results, ticker)
    importance = feature_importance_report(model)
    plot_feature_importance(importance, ticker)

    print(f"\n[+] Top features driving predictions for {ticker}:")
    print(importance.head(5).to_string(index=False))

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and backtest a stock direction predictor.")
    parser.add_argument("ticker", help="Stock ticker symbol, e.g. AAPL, MSFT, TSLA")
    parser.add_argument("--years", type=int, default=5, help="Years of history to use")
    parser.add_argument("--test-frac", type=float, default=0.2,
                         help="Fraction of most recent data held out for testing")
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Probability threshold above which the strategy 'buys'")
    args = parser.parse_args()

    run(args.ticker, args.years, args.test_frac, args.threshold)
