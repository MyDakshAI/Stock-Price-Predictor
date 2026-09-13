"""Tests for the train/test split and the classifier wrapper."""

import numpy as np
import pytest

from model import (
    train_test_split_chronological, train_model, predict,
    get_available_features, feature_importance_report,
    save_model, load_model,
)


def test_split_is_chronological_not_shuffled(feats):
    train, test = train_test_split_chronological(feats, test_frac=0.2)

    assert len(train) + len(test) == len(feats)
    assert train.index.max() < test.index.min(), \
        "every training date must precede every test date"
    assert train.index.is_monotonic_increasing
    assert test.index.is_monotonic_increasing


def test_split_respects_test_fraction(feats):
    train, test = train_test_split_chronological(feats, test_frac=0.3)
    assert len(test) == pytest.approx(len(feats) * 0.3, abs=1)


def test_train_model_returns_usable_model(feats):
    train, test = train_test_split_chronological(feats)
    model, scaler = train_model(train)

    probs = predict(model, scaler, test)
    assert len(probs) == len(test)
    assert np.all((probs >= 0) & (probs <= 1)), "predict must return probabilities"


def test_model_remembers_its_feature_columns(feats):
    train, _ = train_test_split_chronological(feats)
    model, _ = train_model(train)
    assert model.feature_columns_ == get_available_features(train)


def test_scaler_is_fit_on_training_data_only(feats):
    """The scaler must never see test data during fit. If it did, test-set
    statistics would leak into training."""
    train, test = train_test_split_chronological(feats)
    _, scaler = train_model(train)

    cols = get_available_features(train)
    np.testing.assert_allclose(scaler.mean_, train[cols].values.mean(axis=0), rtol=1e-9)


def test_feature_importance_covers_all_features(feats):
    train, _ = train_test_split_chronological(feats)
    model, _ = train_model(train)
    report = feature_importance_report(model)

    assert len(report) == len(model.feature_columns_)
    assert report["importance"].sum() == pytest.approx(1.0, rel=1e-6)
    assert report["importance"].is_monotonic_decreasing


def test_save_and_load_roundtrip(feats, tmp_path):
    train, test = train_test_split_chronological(feats)
    model, scaler = train_model(train)
    save_model(model, scaler, "TEST", out_dir=str(tmp_path))

    loaded_model, loaded_scaler = load_model("TEST", model_dir=str(tmp_path))
    np.testing.assert_allclose(
        predict(model, scaler, test),
        predict(loaded_model, loaded_scaler, test),
    )


def test_training_is_deterministic(feats):
    """Same data in, same predictions out. Without a fixed random_state,
    'improvements' would be indistinguishable from seed luck."""
    train, test = train_test_split_chronological(feats)
    m1, s1 = train_model(train)
    m2, s2 = train_model(train)
    np.testing.assert_allclose(predict(m1, s1, test), predict(m2, s2, test))
