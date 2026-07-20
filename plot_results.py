"""
plot_results.py

Generates a chart comparing the strategy's cumulative return against
buy-and-hold over the test period, plus a feature importance chart.
"""

import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_backtest(results: pd.DataFrame, ticker: str, out_dir: str = "outputs"):
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(results.index, results["strategy_cumulative"], label="Model strategy")
    ax.plot(results.index, results["buy_hold_cumulative"], label="Buy & hold", linestyle="--")
    ax.set_title(f"{ticker}: Strategy vs Buy & Hold (out-of-sample test period)")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"{ticker}_backtest.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[+] Saved backtest chart to {out_path}")
    return out_path


def plot_feature_importance(importance_df: pd.DataFrame, ticker: str, out_dir: str = "outputs"):
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance_df["feature"], importance_df["importance"])
    ax.invert_yaxis()
    ax.set_title(f"{ticker}: Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"{ticker}_feature_importance.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[+] Saved feature importance chart to {out_path}")
    return out_path
