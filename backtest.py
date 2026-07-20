"""
backtest.py

Simulates trading using the model's predictions and compares
performance against a simple buy-and-hold baseline.

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


def summarize_backtest(results: pd.DataFrame) -> dict:
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

    return {
        "test_days": days,
        "directional_accuracy": accuracy,
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
    excess = returns - risk_free_rate / 252
    if excess.std() == 0:
        return 0.0
    return (excess.mean() / excess.std()) * np.sqrt(252)


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
          f"(50% = random guessing)")
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

    if summary['directional_accuracy'] < 0.53:
        print("\n[!] Directional accuracy is close to 50% (coin flip).")
        print("    This model likely has little to no real predictive edge")
        print("    on this ticker/timeframe -- which is the expected, honest")
        print("    result for short-term price direction. Treat this as a")
        print("    learning exercise, not a trading signal.")
