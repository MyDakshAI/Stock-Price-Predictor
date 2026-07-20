"""
model.py

Trains a model to predict next-day price DIRECTION (up/down) from
technical indicator features. Uses a Random Forest by default (fast,
hard to overfit badly, gives feature importances) with an option for
an LSTM if you want to go the deep-learning route.

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
from sklearn.preprocessing import StandardScaler
import joblib
import os

FEATURE_COLUMNS = [
    "sma_10", "sma_50", "ema_10", "rsi_14", "macd", "macd_signal",
    "volatility_10", "volume_change",
    "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5",
]


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


def train_model(train_df: pd.DataFrame):
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["target_direction"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    return model, scaler


def predict(model, scaler, df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURE_COLUMNS]
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


def feature_importance_report(model, feature_names=FEATURE_COLUMNS) -> pd.DataFrame:
    importances = model.feature_importances_
    report = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)
    return report
