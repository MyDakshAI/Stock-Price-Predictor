"""
walk_forward.py

Walk-forward validation: instead of one fixed 80/20 train/test split
(which gives you exactly one accuracy number that depends heavily on
which slice of history you happened to pick), this slides a training
window forward through time, retraining and testing on the next chunk
each time, and collects a whole distribution of out-of-sample results.

This is what turns "51-59% depending on the window" from an anecdote
into an actual measurement: the mean and spread of that distribution
tell you how stable any apparent edge really is.

Each fold is still strictly chronological (train only on the past,
test only on the immediate future of that fold) -- no shuffling, ever.
"""

import numpy as np
import pandas as pd

from model import train_model, predict, get_available_features
from backtest import run_backtest, summarize_backtest, naive_baseline_accuracy


def walk_forward_folds(n_rows: int, n_folds: int = 5, min_train_frac: float = 0.4):
    """
    Yields (train_start, train_end, test_start, test_end) index tuples
    for an expanding-window walk-forward split of n_rows rows.

    The training window expands each fold (uses all data up to that
    point); the test window is a fixed-size slice immediately after it.
    """
    min_train_size = int(n_rows * min_train_frac)
    remaining = n_rows - min_train_size
    test_size = max(remaining // n_folds, 1)

    folds = []
    train_end = min_train_size
    for _ in range(n_folds):
        test_start = train_end
        test_end = min(test_start + test_size, n_rows)
        if test_start >= n_rows:
            break
        folds.append((0, train_end, test_start, test_end))
        train_end = test_end
    return folds


def run_walk_forward(feats: pd.DataFrame, n_folds: int = 5, min_train_frac: float = 0.4,
                      threshold: float = 0.5, tune: bool = False) -> pd.DataFrame:
    """
    Runs walk-forward validation over feats (the full featurized
    dataframe, already chronologically sorted).

    Returns a DataFrame with one row per fold: accuracy, naive baseline
    accuracy, strategy return, and the date range tested, so you can see
    the whole distribution rather than a single number.
    """
    n_rows = len(feats)
    folds = walk_forward_folds(n_rows, n_folds=n_folds, min_train_frac=min_train_frac)

    if not folds:
        raise ValueError("Not enough data to run walk-forward validation with these settings.")

    rows = []
    for i, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
        train_df = feats.iloc[tr_start:tr_end]
        test_df = feats.iloc[te_start:te_end]
        if len(test_df) < 5 or len(train_df) < 30:
            continue

        model, scaler = train_model(train_df, tune=tune)
        predicted_up_prob = predict(model, scaler, test_df)
        results = run_backtest(test_df, predicted_up_prob, threshold=threshold)
        summary = summarize_backtest(results, test_df)

        rows.append({
            "fold": i + 1,
            "train_days": len(train_df),
            "test_days": len(test_df),
            "test_start": test_df.index[0].date(),
            "test_end": test_df.index[-1].date(),
            "accuracy": summary["directional_accuracy"],
            "naive_baseline": summary["naive_baseline_accuracy"],
            "roc_auc": summary["roc_auc"],
            "strategy_return": summary["strategy_total_return"],
            "buy_hold_return": summary["buy_hold_total_return"],
        })

    return pd.DataFrame(rows)


def summarize_walk_forward(fold_results: pd.DataFrame) -> dict:
    """Aggregate stats across folds: mean/std accuracy, how many folds
    beat the naive baseline, etc. This is the honest headline number --
    much more so than any single fold."""
    if fold_results.empty:
        return {}

    beat_naive = (fold_results["accuracy"] > fold_results["naive_baseline"]).sum()
    return {
        "n_folds": len(fold_results),
        "mean_accuracy": fold_results["accuracy"].mean(),
        "std_accuracy": fold_results["accuracy"].std(),
        "min_accuracy": fold_results["accuracy"].min(),
        "max_accuracy": fold_results["accuracy"].max(),
        "mean_naive_baseline": fold_results["naive_baseline"].mean(),
        "folds_beating_naive_baseline": int(beat_naive),
        "mean_strategy_return": fold_results["strategy_return"].mean(),
        "mean_buy_hold_return": fold_results["buy_hold_return"].mean(),
    }


if __name__ == "__main__":
    import sys
    from fetch_data import fetch
    from features import load_price_data, build_features

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    csv_path = fetch(ticker, years=8)
    df = load_price_data(csv_path)
    feats = build_features(df)

    fold_results = run_walk_forward(feats, n_folds=5)
    print(fold_results.to_string(index=False))

    summary = summarize_walk_forward(fold_results)
    print(f"\nMean accuracy across {summary['n_folds']} folds: "
          f"{summary['mean_accuracy']:.1%} (+/- {summary['std_accuracy']:.1%}), "
          f"vs. mean naive baseline {summary['mean_naive_baseline']:.1%}")
    print(f"Folds beating the naive baseline: "
          f"{summary['folds_beating_naive_baseline']}/{summary['n_folds']}")
