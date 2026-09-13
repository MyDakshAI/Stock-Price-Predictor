"""
model.py

Trains a model to predict next-day price DIRECTION (up/down) from
technical indicator features. Uses a Random Forest by default (fast,
hard to overfit badly, gives feature importances).

Why direction instead of exact price?
Predicting the exact next-day price is a much easier-looking but
much LESS meaningful task -- a model that just predicts "tomorrow ==
today" scores deceptively well on raw price because prices don't move
much day to day. Predicting direction (up/down) is the honest version
of "does this model have any edge at all."
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Full candidate feature list. build_features() may or may not produce
# every column here (e.g. the market-relative features only appear when
# a benchmark index was supplied) -- get_available_features() below
# intersects this list with whatever columns actually exist so the
# model quietly adapts instead of KeyError-ing.
ALL_FEATURE_COLUMNS = [
    "sma_10", "sma_50", "ema_10", "rsi_14", "macd", "macd_signal",
    "bb_pct", "bb_width", "atr_14", "stoch_k", "obv_change",
    "volatility_10", "volume_change",
    "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5",
    "mkt_return_1", "mkt_return_5", "relative_strength",
]

# Kept for backwards compatibility with any code importing the old name.
FEATURE_COLUMNS = ALL_FEATURE_COLUMNS


def get_available_features(df: pd.DataFrame) -> list:
    """Returns the subset of ALL_FEATURE_COLUMNS actually present in df,
    in a stable order. Lets the model work whether or not market-relative
    features were built."""
    return [c for c in ALL_FEATURE_COLUMNS if c in df.columns]


def train_test_split_chronological(df: pd.DataFrame, test_frac: float = 0.2):
    """
    Splits data by TIME, not randomly. Random splitting would leak
    future information into training (a classic mistake in finance ML
    that makes models look great and be useless).
    """
    split_idx = int(len(df) * (1 - test_frac))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test


def train_model(train_df: pd.DataFrame, feature_columns: list = None, tune: bool = False):
    """
    Trains a RandomForestClassifier on train_df.

    Args:
        train_df: training data with feature + target_direction columns.
        feature_columns: which columns to use as features. Defaults to
            whatever of ALL_FEATURE_COLUMNS is present in train_df.
        tune: if True, run a small grid search with TimeSeriesSplit
            cross-validation (never a random/shuffled split -- that
            would leak future data into validation folds) instead of
            using fixed hyperparameters. Slower, but avoids hand-picked
            hyperparameters that happen to work for one ticker/window.
    """
    if feature_columns is None:
        feature_columns = get_available_features(train_df)

    X_train = train_df[feature_columns]
    y_train = train_df["target_direction"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if tune:
        param_grid = {
            "n_estimators": [200, 300],
            "max_depth": [4, 6, 8],
            "min_samples_leaf": [10, 20, 40],
        }
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        tscv = TimeSeriesSplit(n_splits=4)
        search = GridSearchCV(base_model, param_grid, cv=tscv, scoring="accuracy", n_jobs=-1)
        search.fit(X_train_scaled, y_train)
        model = search.best_estimator_
    else:
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_scaled, y_train)

    model.feature_columns_ = feature_columns  # remember what it was trained on
    return model, scaler


def predict(model, scaler, df: pd.DataFrame) -> np.ndarray:
    feature_columns = getattr(model, "feature_columns_", None) or get_available_features(df)
    X = df[feature_columns]
    X_scaled = scaler.transform(X)
    return model.predict_proba(X_scaled)[:, 1]  # probability of "up"


def save_model(model, scaler, ticker: str, out_dir: str = "models"):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, f"{ticker}_model.pkl"))
    joblib.dump(scaler, os.path.join(out_dir, f"{ticker}_scaler.pkl"))


def load_model(ticker: str, model_dir: str = "models"):
    model = joblib.load(os.path.join(model_dir, f"{ticker}_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, f"{ticker}_scaler.pkl"))
    return model, scaler


def feature_importance_report(model, feature_names=None) -> pd.DataFrame:
    if feature_names is None:
        feature_names = getattr(model, "feature_columns_", ALL_FEATURE_COLUMNS[:len(model.feature_importances_)])
    importances = model.feature_importances_
    report = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)
    return report
