"""Tests for walk-forward validation."""

import pytest

from walk_forward import walk_forward_folds, run_walk_forward, summarize_walk_forward


def test_folds_never_train_on_the_future():
    """Every fold's training window must end before its test window starts."""
    folds = walk_forward_folds(1000, n_folds=5)
    for tr_start, tr_end, te_start, te_end in folds:
        assert tr_start < tr_end <= te_start < te_end


def test_folds_do_not_overlap_each_other():
    folds = walk_forward_folds(1000, n_folds=5)
    test_windows = [(te_start, te_end) for _, _, te_start, te_end in folds]
    for (a_start, a_end), (b_start, b_end) in zip(test_windows, test_windows[1:]):
        assert a_end <= b_start, "test windows must be disjoint and in order"


def test_training_window_expands():
    folds = walk_forward_folds(1000, n_folds=5)
    train_sizes = [tr_end - tr_start for tr_start, tr_end, _, _ in folds]
    assert train_sizes == sorted(train_sizes), "training window should grow each fold"


def test_folds_stay_inside_the_data():
    folds = walk_forward_folds(500, n_folds=4)
    assert all(te_end <= 500 for _, _, _, te_end in folds)


def test_run_walk_forward_returns_one_row_per_fold(feats):
    results = run_walk_forward(feats, n_folds=3)
    assert len(results) == 3
    for col in ["fold", "accuracy", "naive_baseline", "roc_auc",
                "strategy_return", "buy_hold_return", "test_start", "test_end"]:
        assert col in results.columns


def test_walk_forward_test_periods_advance_through_time(feats):
    results = run_walk_forward(feats, n_folds=3)
    starts = list(results["test_start"])
    assert starts == sorted(starts)


def test_summarize_walk_forward_aggregates_correctly(feats):
    results = run_walk_forward(feats, n_folds=3)
    summary = summarize_walk_forward(results)

    assert summary["n_folds"] == 3
    assert summary["min_accuracy"] <= summary["mean_accuracy"] <= summary["max_accuracy"]
    assert 0 <= summary["folds_beating_naive_baseline"] <= 3


def test_raises_when_there_is_not_enough_data(feats):
    with pytest.raises(ValueError):
        run_walk_forward(feats.iloc[:5], n_folds=5, min_train_frac=1.0)
