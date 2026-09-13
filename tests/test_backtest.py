"""Tests for the backtest, its baselines and its statistics."""

import numpy as np
import pandas as pd
import pytest

from backtest import (
    run_backtest, summarize_backtest, naive_baseline_accuracy,
    significance_test, calibration_report, _sharpe_ratio, _max_drawdown,
)


@pytest.fixture
def simple_test_df():
    """Five days of known returns, so expected values can be computed by hand."""
    idx = pd.bdate_range("2024-01-01", periods=5)
    return pd.DataFrame({"target_return": [0.01, -0.02, 0.03, -0.01, 0.02]}, index=idx)


def test_position_taken_only_above_threshold(simple_test_df):
    probs = np.array([0.6, 0.4, 0.7, 0.45, 0.55])
    results = run_backtest(simple_test_df, probs, threshold=0.5, transaction_cost=0.0)
    assert results["position"].tolist() == [1, 0, 1, 0, 1]


def test_strategy_earns_return_only_on_days_it_holds(simple_test_df):
    probs = np.array([0.6, 0.4, 0.7, 0.45, 0.55])
    results = run_backtest(simple_test_df, probs, threshold=0.5, transaction_cost=0.0)
    expected = [0.01, 0.0, 0.03, 0.0, 0.02]
    np.testing.assert_allclose(results["strategy_return"].values, expected)


def test_transaction_costs_reduce_returns(simple_test_df):
    probs = np.array([0.6, 0.4, 0.7, 0.45, 0.55])
    free = run_backtest(simple_test_df, probs, threshold=0.5, transaction_cost=0.0)
    costly = run_backtest(simple_test_df, probs, threshold=0.5, transaction_cost=0.01)
    assert costly["strategy_cumulative"].iloc[-1] < free["strategy_cumulative"].iloc[-1]


def test_buy_hold_is_unaffected_by_threshold(simple_test_df):
    probs = np.full(5, 0.9)
    a = run_backtest(simple_test_df, probs, threshold=0.1)
    b = run_backtest(simple_test_df, probs, threshold=0.8)
    pd.testing.assert_series_equal(a["buy_hold_return"], b["buy_hold_return"])


def test_naive_baseline_is_the_up_day_rate(simple_test_df):
    # 3 of 5 days are positive
    assert naive_baseline_accuracy(simple_test_df) == pytest.approx(0.6)


def test_perfect_predictions_are_significant():
    idx = pd.bdate_range("2024-01-01", periods=200)
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.01, 200)
    df = pd.DataFrame({"target_return": returns}, index=idx)
    perfect = (returns > 0).astype(float) * 0.9 + 0.05

    results = run_backtest(df, perfect, threshold=0.5)
    sig = significance_test(results, baseline_accuracy=0.5)

    assert sig["accuracy"] == pytest.approx(1.0)
    assert sig["p_value"] < 0.001
    assert sig["significant_at_0.05"]


def test_random_predictions_are_not_significant():
    idx = pd.bdate_range("2024-01-01", periods=300)
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"target_return": rng.normal(0, 0.01, 300)}, index=idx)
    noise = rng.uniform(0, 1, 300)

    results = run_backtest(df, noise, threshold=0.5)
    sig = significance_test(results, baseline_accuracy=0.5)
    assert not sig["significant_at_0.05"], "pure noise should not look significant"


def test_calibration_report_scores_a_perfect_model_well():
    idx = pd.bdate_range("2024-01-01", periods=200)
    rng = np.random.default_rng(1)
    returns = rng.normal(0, 0.01, 200)
    df = pd.DataFrame({"target_return": returns}, index=idx)
    confident = np.where(returns > 0, 0.99, 0.01)

    results = run_backtest(df, confident, threshold=0.5)
    cal = calibration_report(results)

    assert cal["roc_auc"] == pytest.approx(1.0)
    assert cal["brier_score"] < 0.01


def test_summarize_backtest_has_all_expected_keys(simple_test_df):
    probs = np.array([0.6, 0.4, 0.7, 0.45, 0.55])
    results = run_backtest(simple_test_df, probs)
    summary = summarize_backtest(results, simple_test_df)

    for key in ["test_days", "directional_accuracy", "naive_baseline_accuracy",
                "significance_vs_naive", "roc_auc", "brier_score",
                "strategy_total_return", "buy_hold_total_return",
                "strategy_sharpe", "strategy_max_drawdown"]:
        assert key in summary


def test_max_drawdown_is_negative_or_zero():
    rising = pd.Series([1.0, 1.1, 1.2, 1.3])
    assert _max_drawdown(rising) == pytest.approx(0.0)

    dipping = pd.Series([1.0, 1.2, 0.9, 1.1])
    assert _max_drawdown(dipping) == pytest.approx((0.9 - 1.2) / 1.2)


def test_sharpe_of_constant_returns_is_zero():
    assert _sharpe_ratio(pd.Series([0.01] * 10)) == 0.0
