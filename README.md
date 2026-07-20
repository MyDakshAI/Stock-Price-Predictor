# Stock Price Direction Predictor (with Honest Backtesting)

An ML project that predicts next-day stock price **direction** (up/down)
using technical indicators, then backtests the resulting strategy
against simple buy-and-hold — the right way, with a chronological
train/test split so there's no lookahead bias.

**This is a portfolio/learning project, not a trading tool.** Read the
"Interpreting your results" section below before you get excited about
any number it spits out.

## Project structure

```
stock-predictor/
├── requirements.txt
├── fetch_data.py           # Downloads price history via yfinance
├── features.py             # Technical indicators + target labels
├── model.py                # RandomForest classifier (direction: up/down)
├── backtest.py             # Simulates the strategy vs buy & hold
├── plot_results.py         # Chart generation
├── main.py                 # Runs the full pipeline end to end
├── data/                   # Downloaded price CSVs land here
├── models/                 # Trained model + scaler saved here
└── outputs/                # Backtest + feature importance charts
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run it

```bash
python main.py AAPL
```

Optional flags:
```bash
python main.py TSLA --years 8 --test-frac 0.2 --threshold 0.55
```

- `--years` — how much history to pull (default 5)
- `--test-frac` — fraction of the most recent data held out as the
  out-of-sample test set (default 0.2 = last 20%)
- `--threshold` — probability above which the strategy "buys" for the
  day (default 0.5; raising it makes the strategy more selective)

This will:
1. Download the ticker's daily price history
2. Build ~12 technical-indicator features (moving averages, RSI, MACD, volatility, lagged returns)
3. Train a Random Forest on the **older** portion of the data
4. Test on the **newer**, unseen portion only
5. Print a comparison table (strategy vs buy-and-hold)
6. Save two charts to `outputs/`: cumulative returns and feature importance

## Interpreting your results (read this)

The most important number in the output is **directional accuracy**.

- **~50%** = the model is no better than a coin flip. This is the
  most common and most honest result for short-term stock direction —
  it means the model didn't find a real edge, which is expected
  because daily price moves are dominated by noise.
- **50–55%** = a small statistical edge might exist, but it's easily
  within the range of what you'd see from randomness or from the
  specific historical window chosen. Don't trust it without testing
  across many tickers and time periods.
- **55%+** = worth a second look, but be very suspicious — this is
  often a sign of subtle data leakage (e.g., a feature that
  accidentally "sees" future information) rather than a genuine edge.
  Double-check the feature engineering before believing it.

Also watch **max drawdown** — a strategy with a great total return but
a brutal drawdown isn't necessarily better than buy-and-hold; it just
took a rockier path to get there.

## Why this is built the way it is

A few choices that make this an honest backtest instead of a
misleading one, since a lot of tutorials online get these wrong:

- **Chronological split, not random.** Shuffling time-series data lets
  the model "peek" at the future during training. This project always
  trains on the older chunk and tests on the newer chunk only.
- **Predicting direction, not exact price.** A model that predicts
  "tomorrow's price ≈ today's price" scores deceptively well on raw
  price prediction because prices don't move much day-to-day, but it's
  useless for trading. Predicting direction is the honest version of
  "does this model know anything."
- **Transaction costs included.** Even a small per-trade cost
  (0.05% by default) can turn an apparently profitable strategy
  unprofitable once you account for realistic trading frictions.
- **Feature importance reported.** Lets you sanity-check *why* the
  model predicts what it does, rather than treating it as a black box.

## Extending this project

Ideas if you want to go further:
- Try different tickers/sectors and compare directional accuracy —
  does it vary by volatility or sector?
- Add sentiment features (news headlines, social media) as extra columns.
- Swap the Random Forest for an LSTM/Transformer and see if it actually
  helps (often it doesn't, for this kind of tabular technical data —
  which is itself an interesting finding to write up).
- Add walk-forward validation (retrain periodically through the test
  set) instead of a single train/test split, which is closer to how
  you'd actually deploy something like this.
