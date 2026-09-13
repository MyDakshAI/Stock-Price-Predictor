# Does the model know anything?

**Can machine learning predict which way a stock moves tomorrow? I built the honest version of that experiment to find out.**

[![Live demo](https://img.shields.io/badge/Live_demo-Open_the_dashboard-E8A33D?style=for-the-badge)](https://REPLACE-WITH-YOUR-STREAMLIT-URL.streamlit.app)
[![tests](https://github.com/MyDakshAI/Stock-Price-Predictor/actions/workflows/tests.yml/badge.svg)](https://github.com/MyDakshAI/Stock-Price-Predictor/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Random_Forest-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)

> Type a ticker, get a rigorous out-of-sample backtest with a verdict that tells you the truth, including when the truth is "this found nothing."

Built by [Daksh Goswami](https://www.linkedin.com/in/daksh-goswami-3077aa280).

<!-- Add a screenshot here: run the app, take a screenshot of the dashboard, save it as docs/dashboard.png -->
<!-- ![Dashboard screenshot](docs/dashboard.png) -->

---

## New to this? Start here

**What the project does, in plain English.** Every trading day, a stock closes either higher or lower than the day before. This trains a computer model to guess which one happens tomorrow, using only patterns in past prices and trading volume, and then measures how often that guess was right across a stretch of days the model never saw while learning.

**Why "beating 50%" is a trap.** A coin flip gets 50%, so that sounds like the target. It isn't. Stocks drift upward over time, so a rule as dumb as "always guess up, every day" already scores around 53% on most stocks while knowing nothing at all. That dumb rule is the real bar. A model scoring 52% has not found a signal, it has found a worse version of guessing, and a project that celebrates 52% as "beating the market" is fooling itself.

**Why a negative result is the point.** Short-term price movement is mostly noise. If a straightforward model could reliably call tomorrow's direction, firms with supercomputers and PhD teams would have traded that opportunity away long before it reached a laptop. So the interesting engineering question is not "can I get a big number", it is "can I build something rigorous enough that I would believe its answer either way". That is what this is.

**The vocabulary**, if you want to read the dashboard:

| Term | What it means |
|---|---|
| Directional accuracy | How often the model correctly guessed tomorrow's up or down |
| Naive baseline | The score from always guessing "up", the bar that actually matters |
| p-value | Chance of seeing an edge this big by luck. Below 0.05 is the usual "probably real" bar |
| ROC-AUC | How well it separates up-days from down-days. 0.50 is no signal, 1.00 is perfect |
| Buy and hold | What you'd have made just buying and doing nothing, the comparison that counts |
| Sharpe ratio | Return adjusted for how bumpy the ride was, higher is better |
| Max drawdown | The worst peak-to-trough fall along the way, closer to zero is better |
| Walk-forward validation | Testing across many different stretches of history instead of just one, so you see whether a result holds up or was luck |
| Data leakage | The classic bug where a model accidentally sees information from the future, making a useless model look brilliant |

---

## The short version

Most "AI predicts the stock market" projects quietly cheat, then report spectacular results. This one is built to catch itself cheating, and it reports what actually happened.

<!-- TODO: run `python main.py AAPL --market SPY --walk-forward` on a few tickers and
     paste your real measured range below, then delete this comment. -->

Running it across many tickers and many time windows, directional accuracy lands around **[FILL IN YOUR MEASURED RANGE]**, compared against the naive "always guess up" baseline rather than a 50% coin flip. Daily price moves are dominated by noise, and a result at or near that baseline is what efficient market theory predicts. The value of this project is the rigor that makes whatever conclusion you reach trustworthy, not a number that looks good on a slide.

## What it does

| | |
|---|---|
| **Predicts direction, not price** | Predicting tomorrow's exact price is a trap: a lazy model that guesses "same as today" scores beautifully and knows nothing. Up or down is the honest test. |
| **Tests on data it never saw** | Strictly chronological splits. The model trains only on the past and is judged only on its future. |
| **Compares against two baselines** | Buy-and-hold, plus the naive "always predict up" rule. Stocks rise on ~53% of days, so beating 50% proves nothing. This is the bar most projects skip. |
| **Walk-forward validation** | Retrains across many rolling time windows and reports the full distribution of accuracy, not one lucky split. |
| **Reports statistical significance** | A binomial test against the naive baseline, so you can tell a real edge from a small sample. |
| **Checks its own confidence** | ROC-AUC, Brier score and a calibration curve: when the model says 60%, is it actually right 60% of the time? |
| **Models transaction costs** | 0.05% per position change, because a strategy that only wins before fees is not a strategy. |

## Why the methodology is the point

When I started reading about ML and trading, I noticed most tutorials shuffle their training data randomly, which secretly lets the model peek at the future. Their backtests look incredible and are worthless in real life. Every design decision here pushes the other way:

- **No shuffling, ever.** Chronological splits in both the single backtest and every walk-forward fold.
- **The scaler is fit on training data only**, so test-set statistics never leak backwards. There is a unit test asserting this.
- **A dedicated look-ahead test.** `tests/test_features.py` truncates the price history, rebuilds the features, and asserts that no earlier feature value changed. If any indicator reaches forward in time, CI fails.
- **Suspicion built into the UI.** If accuracy comes back unusually high, the dashboard tells you to suspect data leakage rather than congratulating you.

That last one matters more than it sounds. A model that looks too good is nearly always broken, and a tool that says so is more useful than one that does not.

## Try it

**[Open the live dashboard](https://REPLACE-WITH-YOUR-STREAMLIT-URL.streamlit.app)**, no install required. Or run it locally:

```bash
git clone https://github.com/MyDakshAI/Stock-Price-Predictor.git
cd Stock-Price-Predictor
pip install -r requirements.txt
streamlit run app.py
```

The dashboard has three modes, all driven by the ticker box:

```
AAPL                    single out-of-sample backtest
AAPL + walk-forward     accuracy distribution across rolling time windows
AAPL, MSFT, TSLA, NVDA  side-by-side comparison across tickers
```

That last mode is the real test. An approach that finds genuine signal should work on more than one stock. If it beats the baseline on some tickers and not others with no pattern, that is what noise looks like.

### Command line

```bash
python main.py AAPL                                  # single backtest
python main.py AAPL --market SPY --walk-forward      # the rigorous version
python main.py TSLA --years 8 --threshold 0.55       # tune the strategy
python main.py AAPL --tune                           # grid-search via TimeSeriesSplit CV
```

## Tech stack

`scikit-learn` Random Forest classifier, `ta` for technical indicators (RSI, MACD, Bollinger Bands, ATR, stochastic oscillator, OBV), `yfinance` for market data, `scipy` for significance testing, `pandas` and `numpy` for the pipeline, `Streamlit` and `Plotly` for the dashboard, `pytest` and GitHub Actions for the test suite.

## Project structure

```
├── app.py              Streamlit dashboard (3 modes, cached pipeline)
├── main.py             CLI pipeline end to end
├── fetch_data.py       Price data download with daily caching
├── features.py         Technical indicators + market-relative features
├── model.py            Random Forest, chronological split, optional CV tuning
├── backtest.py         Strategy simulation, baselines, significance, calibration
├── walk_forward.py     Rolling-window validation across time
├── plot_results.py     Static chart generation
└── tests/              34 tests including a look-ahead leakage check
```

## How to read your own results

- **At or below the naive baseline** is the normal outcome. It means no edge was found, and that is a real finding, not a failure.
- **Slightly above, p > 0.05** means the gap is within what randomness produces. Run walk-forward before believing it.
- **Well above the baseline** should make you suspicious first and excited second. Check the features for leakage.

## Disclaimer

I built this to learn, and it is not financial advice. Please do not put real money behind it. The interesting part is not whether it makes money, it is understanding exactly why it does not.

## License

MIT. See [LICENSE](LICENSE).

## Contact

Built by Daksh Goswami. Find me on [LinkedIn](https://www.linkedin.com/in/daksh-goswami-3077aa280) or open an issue on this repo.

## What I would do next

Sentiment features from news headlines, an LSTM comparison to test whether sequence models beat a Random Forest on tabular market data (my hunch is no), and scaling the multi-ticker comparison to a few hundred stocks to measure how often an apparent edge is just the multiple-comparisons problem.
