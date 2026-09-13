"""
backtest.py

Simulates trading using the model's predictions and compares
performance against two baselines: buy-and-hold, and a naive "always
predict up" classifier (stocks drift upward more often than not over
multi-year windows, so this is a much fairer bar than "50% = coin
flip" -- a model has to beat this, not just beat 50%, to have a real
edge).

Strategy simulated:
    Each day, if the model predicts "up" with probability > threshold,
    hold the stock for that day. Otherwise, hold cash (0% return).
    This is deliberately simple -- no leverage, no shorting, no fees
    modeled beyond an optional flat cost per trade.

IMPORTANT: this backtest tells you how the model WOULD have performed
on data it didn't train on (out-of-sample test set), which is the
honest way to evaluate it. Past performance, even out-of-sample,
still doesn't guarantee future performance -- markets change regime
over time. Treat results as a learning exercise, not a trading signal.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, brier_score_loss


def run_backtest(test_df: pd.DataFrame, predicted_up_prob: np.ndarray,
                  threshold: float = 0.5, transaction_cost: float = 0.0005):
    """
    Args:
        test_df: dataframe with at least 'target_return' column, indexed by date
        predicted_up_prob: model's predicted probability of "up" for each row
        threshold: probability above which we "buy" for that day
        transaction_cost: flat fractional cost applied whenever position changes
                           (default 0.05% -- a rough stand-in for spread/fees)

    Returns:
        DataFrame with daily strategy returns, buy-and-hold returns, and
        cumulative versions of both.
    """
    results = test_df[["target_return"]].copy()
    results["predicted_up_prob"] = predicted_up_prob
    results["position"] = (results["predicted_up_prob"] > threshold).astype(int)

    # Apply transaction cost whenever position changes (enter/exit)
    position_changes = results["position"].diff().abs().fillna(0)
    costs = position_changes * transaction_cost

    results["strategy_return"] = results["position"] * results["target_return"] - costs
    results["buy_hold_return"] = results["target_return"]

    results["strategy_cumulative"] = (1 + results["strategy_return"]).cumprod()
    results["buy_hold_cumulative"] = (1 + results["buy_hold_return"]).cumprod()

    return results


def naive_baseline_accuracy(test_df: pd.DataFrame) -> float:
    """
    Accuracy of the dumbest possible classifier: always predict "up".
    Stocks drift upward more often than not over most multi-year windows,
    so this is usually well above 50% -- the model needs to beat THIS,
    not 50%, to demonstrate a real edge.
    """
    actual_direction = (test_df["target_return"] > 0).astype(int)
    return float((actual_direction == 1).mean())


def significance_test(results: pd.DataFrame, baseline_accuracy: float = 0.5) -> dict:
    """
    One-sided binomial test: is the model's directional accuracy on the
    test set significantly better than a given baseline (default 50%,
    but pass naive_baseline_accuracy() for the fairer comparison)?

    A small p-value means the accuracy is unlikely to be explained by
    chance alone at this sample size. It does NOT mean the edge is
    tradeable or will persist -- markets can still change regime.
    """
    actual_direction = (results["target_return"] > 0).astype(int)
    predicted_direction = (results["predicted_up_prob"] > 0.5).astype(int)
    correct = (actual_direction == predicted_direction)
    n = len(correct)
    k = int(correct.sum())
    accuracy = k / n

    binom = stats.binomtest(k, n, p=baseline_accuracy, alternative="greater")
    return {
        "n": n,
        "correct": k,
        "accuracy": accuracy,
        "baseline": baseline_accuracy,
        "p_value": binom.pvalue,
        "significant_at_0.05": binom.pvalue < 0.05,
    }


def calibration_report(results: pd.DataFrame, n_bins: int = 5) -> dict:
    """
    Checks whether the model's predicted probabilities are trustworthy,
    not just its final up/down call. A model that says "70% confident"
    should be right about 70% of the time on those days -- if not, the
    threshold slider in the app is tuning against noise.

    Returns overall metrics plus a per-bin breakdown (predicted probability
    range -> actual fraction that went up).
    """
    actual_direction = (results["target_return"] > 0).astype(int).values
    predicted_prob = results["predicted_up_prob"].values

    auc = roc_auc_score(actual_direction, predicted_prob) if len(np.unique(actual_direction)) > 1 else float("nan")
    brier = brier_score_loss(actual_direction, predicted_prob)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(predicted_prob, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_range": f"{bins[b]:.2f}-{bins[b+1]:.2f}",
            "n": int(mask.sum()),
            "avg_predicted": float(predicted_prob[mask].mean()),
            "actual_up_rate": float(actual_direction[mask].mean()),
        })

    return {
        "roc_auc": float(auc),
        "brier_score": float(brier),
        "calibration_bins": rows,
    }


def summarize_backtest(results: pd.DataFrame, test_df: pd.DataFrame = None) -> dict:
    strategy_total_return = results["strategy_cumulative"].iloc[-1] - 1
    buy_hold_total_return = results["buy_hold_cumulative"].iloc[-1] - 1

    days = len(results)
    strategy_annualized = (1 + strategy_total_return) ** (252 / days) - 1
    buy_hold_annualized = (1 + buy_hold_total_return) ** (252 / days) - 1

    strategy_sharpe = _sharpe_ratio(results["strategy_return"])
    buy_hold_sharpe = _sharpe_ratio(results["buy_hold_return"])

    actual_direction = (results["target_return"] > 0).astype(int)
    predicted_direction = (results["predicted_up_prob"] > 0.5).astype(int)
    accuracy = (actual_direction == predicted_direction).mean()

    max_dd_strategy = _max_drawdown(results["strategy_cumulative"])
    max_dd_buy_hold = _max_drawdown(results["buy_hold_cumulative"])

    naive_acc = naive_baseline_accuracy(test_df if test_df is not None else results)
    sig_vs_naive = significance_test(results, baseline_accuracy=naive_acc)
    sig_vs_coinflip = significance_test(results, baseline_accuracy=0.5)
    calibration = calibration_report(results)

    return {
        "test_days": days,
        "directional_accuracy": accuracy,
        "naive_baseline_accuracy": naive_acc,
        "significance_vs_naive": sig_vs_naive,
        "significance_vs_coinflip": sig_vs_coinflip,
        "roc_auc": calibration["roc_auc"],
        "brier_score": calibration["brier_score"],
        "calibration_bins": calibration["calibration_bins"],
        "strategy_total_return": strategy_total_return,
        "buy_hold_total_return": buy_hold_total_return,
        "strategy_annualized_return": strategy_annualized,
        "buy_hold_annualized_return": buy_hold_annualized,
        "strategy_sharpe": strategy_sharpe,
        "buy_hold_sharpe": buy_hold_sharpe,
        "strategy_max_drawdown": max_dd_strategy,
        "buy_hold_max_drawdown": max_dd_buy_hold,
    }


def _sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Annualized Sharpe ratio.

    The near-zero tolerance matters: if the buy threshold is set high
    enough that the strategy never enters a position, every daily return
    is exactly 0 and the standard deviation comes out as floating-point
    noise (around 1e-19) rather than a clean 0. Dividing by that produced
    a nonsense Sharpe of ~1e16 on the dashboard, so anything below the
    tolerance is treated as "no variation, no risk-adjusted return".
    """
    excess = returns - risk_free_rate / 252
    std = excess.std()
    if not np.isfinite(std) or std < 1e-12:
        return 0.0
    return (excess.mean() / std) * np.sqrt(252)


def _max_drawdown(cumulative: pd.Series) -> float:
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def print_summary(summary: dict, ticker: str):
    print(f"\n{'='*50}")
    print(f"BACKTEST RESULTS: {ticker}")
    print(f"{'='*50}")
    print(f"Test period length:        {summary['test_days']} trading days")
    print(f"Directional accuracy:      {summary['directional_accuracy']:.1%}  "
          f"(naive 'always up' baseline: {summary['naive_baseline_accuracy']:.1%})")
    print(f"ROC-AUC:                   {summary['roc_auc']:.3f}  "
          f"(Brier score: {summary['brier_score']:.4f}, lower is better)")
    sig = summary["significance_vs_naive"]
    print(f"vs. naive baseline:        p={sig['p_value']:.3f}  "
          f"({'significant at 0.05' if sig['significant_at_0.05'] else 'not significant'})")
    print(f"\n{'Metric':<28}{'Strategy':>12}{'Buy & Hold':>12}")
    print(f"{'-'*52}")
    print(f"{'Total return':<28}{summary['strategy_total_return']:>11.1%} "
          f"{summary['buy_hold_total_return']:>11.1%}")
    print(f"{'Annualized return':<28}{summary['strategy_annualized_return']:>11.1%} "
          f"{summary['buy_hold_annualized_return']:>11.1%}")
    print(f"{'Sharpe ratio':<28}{summary['strategy_sharpe']:>12.2f}"
          f"{summary['buy_hold_sharpe']:>12.2f}")
    print(f"{'Max drawdown':<28}{summary['strategy_max_drawdown']:>11.1%} "
          f"{summary['buy_hold_max_drawdown']:>11.1%}")
    print(f"{'='*50}")

    if summary['directional_accuracy'] <= summary['naive_baseline_accuracy']:
        print("\n[!] Accuracy does not beat the naive 'always predict up' baseline.")
        print("    This model likely has little to no real predictive edge")
        print("    on this ticker/timeframe, which is the expected, honest")
        print("    result for short-term price direction. Treat this as a")
        print("    learning exercise, not a trading signal.")
