"""
Shared analysis helpers for the web API (and any future UI).

Keeps fetch → features → train → backtest in one place so the dashboard
does not re-implement the CLI pipeline.
"""

from collections import OrderedDict
from datetime import date, datetime
import math

import numpy as np
import pandas as pd

from fetch_data import fetch
from features import load_price_data, build_features
from model import (
    train_test_split_chronological, train_model, predict,
    feature_importance_report,
)
from backtest import run_backtest, summarize_backtest
from walk_forward import run_walk_forward, summarize_walk_forward

FEATURE_LABELS = {
    "sma_10": "SMA (10d)", "sma_50": "SMA (50d)", "ema_10": "EMA (10d)",
    "rsi_14": "RSI (14d)", "macd": "MACD", "macd_signal": "MACD Signal",
    "bb_pct": "Bollinger %B", "bb_width": "Bollinger Width",
    "atr_14": "ATR (14d)", "stoch_k": "Stochastic %K", "obv_change": "OBV Δ",
    "volatility_10": "Volatility", "volume_change": "Volume Δ",
    "return_lag_1": "Return t-1", "return_lag_2": "Return t-2",
    "return_lag_3": "Return t-3", "return_lag_5": "Return t-5",
    "mkt_return_1": "Market Return t-1", "mkt_return_5": "Market Return t-5",
    "relative_strength": "Relative Strength",
}

_CACHE = OrderedDict()
_CACHE_MAX = 24


def json_safe(obj):
    """Convert numpy / pandas values into JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, pd.Series):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()[:10]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        if not math.isfinite(value):
            return None
        return value
    if obj is None:
        return None
    try:
        if pd.isna(obj):
            return None
    except (ValueError, TypeError):
        pass
    return obj


def _cache_get(key):
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    return None


def _cache_set(key, value):
    _CACHE[key] = value
    _CACHE.move_to_end(key)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)


def load_features(ticker: str, years: int, use_market: bool) -> pd.DataFrame:
    key = ("feats", ticker, years, use_market)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    csv_path = fetch(ticker, years)
    df = load_price_data(csv_path)
    market_df = None
    if use_market:
        market_csv = fetch("SPY", years)
        market_df = load_price_data(market_csv)
    feats = build_features(df, market_df=market_df)
    _cache_set(key, feats)
    return feats


def train_and_predict(ticker: str, years: int, use_market: bool):
    key = ("train", ticker, years, use_market)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    feats = load_features(ticker, years, use_market)
    train_df, test_df = train_test_split_chronological(feats, test_frac=0.2)
    model, scaler = train_model(train_df)
    predicted_up_prob = predict(model, scaler, test_df)
    importance = feature_importance_report(model)
    packed = (train_df, test_df, predicted_up_prob, importance)
    _cache_set(key, packed)
    return packed


def _series_payload(test_df: pd.DataFrame, predicted_up_prob, results: pd.DataFrame):
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in results.index],
        "target_return": [float(x) for x in results["target_return"].tolist()],
        "predicted_up_prob": [float(x) for x in predicted_up_prob],
        "strategy_cumulative": [float(x) for x in results["strategy_cumulative"].tolist()],
        "buy_hold_cumulative": [float(x) for x in results["buy_hold_cumulative"].tolist()],
        "test_start": test_df.index[0].strftime("%Y-%m-%d"),
        "test_end": test_df.index[-1].strftime("%Y-%m-%d"),
    }


def analyze_single(ticker: str, years: int, threshold: float, use_market: bool) -> dict:
    train_df, test_df, predicted_up_prob, importance = train_and_predict(
        ticker, years, use_market
    )
    results = run_backtest(test_df, predicted_up_prob, threshold=threshold)
    summary = summarize_backtest(results, test_df)
    importance_rows = [
        {
            "feature": row["feature"],
            "label": FEATURE_LABELS.get(row["feature"], row["feature"]),
            "importance": float(row["importance"]),
        }
        for _, row in importance.head(10).iterrows()
    ]
    last_prob = float(predicted_up_prob[-1])
    return json_safe({
        "mode": "single",
        "ticker": ticker,
        "use_market": use_market,
        "threshold": threshold,
        "train_days": len(train_df),
        "test_days": len(test_df),
        "summary": summary,
        "importance": importance_rows,
        "series": _series_payload(test_df, predicted_up_prob, results),
        "latest_signal": {
            "date": test_df.index[-1].strftime("%Y-%m-%d"),
            "up_probability": last_prob,
            "stance": "long" if last_prob > threshold else "cash",
        },
    })


def analyze_walk_forward(ticker: str, years: int, threshold: float,
                         use_market: bool, n_folds: int = 5) -> dict:
    feats = load_features(ticker, years, use_market)
    folds = run_walk_forward(feats, n_folds=n_folds, threshold=threshold)
    summary = summarize_walk_forward(folds)
    fold_rows = []
    for _, row in folds.iterrows():
        fold_rows.append({
            "fold": int(row["fold"]),
            "train_days": int(row["train_days"]),
            "test_days": int(row["test_days"]),
            "test_start": str(row["test_start"]),
            "test_end": str(row["test_end"]),
            "accuracy": float(row["accuracy"]),
            "naive_baseline": float(row["naive_baseline"]),
            "roc_auc": float(row["roc_auc"]),
            "strategy_return": float(row["strategy_return"]),
            "buy_hold_return": float(row["buy_hold_return"]),
        })
    return json_safe({
        "mode": "walk_forward",
        "ticker": ticker,
        "use_market": use_market,
        "threshold": threshold,
        "summary": summary,
        "folds": fold_rows,
    })


def analyze_compare(tickers: list, years: int, threshold: float,
                    use_market: bool, walk_forward: bool) -> dict:
    rows = []
    for ticker in tickers[:8]:
        if walk_forward:
            payload = analyze_walk_forward(ticker, years, threshold, use_market)
            wf = payload["summary"]
            rows.append({
                "ticker": ticker,
                "accuracy": wf["mean_accuracy"],
                "std_accuracy": wf.get("std_accuracy"),
                "naive_baseline": wf["mean_naive_baseline"],
                "beats_baseline": (
                    f"{wf['folds_beating_naive_baseline']}/{wf['n_folds']}"
                ),
                "strategy_return": wf["mean_strategy_return"],
                "buy_hold_return": wf["mean_buy_hold_return"],
            })
        else:
            payload = analyze_single(ticker, years, threshold, use_market)
            s = payload["summary"]
            rows.append({
                "ticker": ticker,
                "accuracy": s["directional_accuracy"],
                "naive_baseline": s["naive_baseline_accuracy"],
                "roc_auc": s["roc_auc"],
                "p_value": s["significance_vs_naive"]["p_value"],
                "strategy_return": s["strategy_total_return"],
                "buy_hold_return": s["buy_hold_total_return"],
                "sharpe": s["strategy_sharpe"],
            })
    return json_safe({
        "mode": "compare",
        "walk_forward": walk_forward,
        "use_market": use_market,
        "threshold": threshold,
        "rows": rows,
    })
