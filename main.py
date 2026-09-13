"""
main.py

End-to-end pipeline:
    1. Download historical price data for a ticker (and optionally a
       benchmark index for market-relative features)
    2. Build technical indicator features
    3. Split chronologically into train/test (or run walk-forward
       validation across multiple chronological folds)
    4. Train a model to predict next-day direction
    5. Backtest the model's predictions on held-out data
    6. Print results and save charts

Usage:
    python main.py AAPL
    python main.py TSLA --years 8 --test-frac 0.2 --threshold 0.55
    python main.py AAPL --market SPY --walk-forward --folds 5
    python main.py AAPL --tune          # grid-search hyperparameters via TimeSeriesSplit CV
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
from walk_forward import run_walk_forward, summarize_walk_forward


def run(ticker: str, years: int = 5, test_frac: float = 0.2, threshold: float = 0.5,
        market_ticker: str = None, walk_forward: bool = False, folds: int = 5,
        tune: bool = False):
    ticker = ticker.upper()

    # 1. Data
    csv_path = fetch(ticker, years)
    market_df = None
    if market_ticker:
        market_csv = fetch(market_ticker.upper(), years)
        market_df = load_price_data(market_csv)

    # 2. Features
    df = load_price_data(csv_path)
    feats = build_features(df, market_df=market_df)
    print(f"[*] Built {feats.shape[1]} features across {feats.shape[0]} trading days"
          + (f" (including {market_ticker.upper()}-relative features)" if market_ticker else ""))

    if walk_forward:
        print(f"[*] Running walk-forward validation ({folds} folds)...")
        fold_results = run_walk_forward(feats, n_folds=folds, threshold=threshold, tune=tune)
        print(fold_results.to_string(index=False))
        summary = summarize_walk_forward(fold_results)
        print(f"\n[+] Mean accuracy across {summary['n_folds']} folds: "
              f"{summary['mean_accuracy']:.1%} (+/- {summary['std_accuracy']:.1%}), "
              f"vs. mean naive 'always up' baseline {summary['mean_naive_baseline']:.1%}")
        print(f"[+] Folds beating the naive baseline: "
              f"{summary['folds_beating_naive_baseline']}/{summary['n_folds']}")
        return summary

    # 3. Chronological split (never shuffle time series data!)
    train_df, test_df = train_test_split_chronological(feats, test_frac)
    print(f"[*] Train: {len(train_df)} days ({train_df.index[0].date()} to {train_df.index[-1].date()})")
    print(f"[*] Test:  {len(test_df)} days ({test_df.index[0].date()} to {test_df.index[-1].date()})")

    # 4. Train
    print("[*] Training model..." + (" (tuning hyperparameters via CV)" if tune else ""))
    model, scaler = train_model(train_df, tune=tune)
    save_model(model, scaler, ticker)

    # 5. Predict + backtest on test set only (out-of-sample)
    predicted_up_prob = predict(model, scaler, test_df)
    results = run_backtest(test_df, predicted_up_prob, threshold=threshold)
    summary = summarize_backtest(results, test_df)
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
    parser.add_argument("--market", type=str, default=None,
                         help="Benchmark ticker (e.g. SPY) to add market-relative features")
    parser.add_argument("--walk-forward", action="store_true",
                         help="Run walk-forward validation across multiple folds instead of one split")
    parser.add_argument("--folds", type=int, default=5, help="Number of walk-forward folds")
    parser.add_argument("--tune", action="store_true",
                         help="Grid-search hyperparameters with TimeSeriesSplit CV instead of fixed defaults")
    args = parser.parse_args()

    run(args.ticker, args.years, args.test_frac, args.threshold,
        market_ticker=args.market, walk_forward=args.walk_forward,
        folds=args.folds, tune=args.tune)
