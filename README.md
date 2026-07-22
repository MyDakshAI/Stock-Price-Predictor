# 📈 Stock Price Direction Predictor

Hey! Welcome to my project — this is something I built to dig into a
question that honestly kept bugging me: **can machine learning actually
predict the stock market, or is that mostly hype?**

Spoiler: the answer is nuanced, and figuring that out was way more
interesting than just slapping a model on some data and calling it a
day. This project trains an ML model to predict whether a stock will
go **up or down** tomorrow, then rigorously backtests it against just...
buying and holding. No cherry-picked results, no fake "I beat the
market" claims — just an honest look at what the model actually learned
(or didn't).

I also built a full interactive dashboard on top of it because reading
numbers in a terminal is fine, but watching it run live is way cooler. 🖥️

## 🚀 Try it live

```bash
streamlit run app.py
```

This spins up a dashboard where you can type in any ticker, tweak some
settings, and watch the whole pipeline run in real time — data
download, feature engineering, training, backtest, charts, all of it.
I'm honestly pretty proud of how this turned out.

## 🤔 Why I built this this way

When I started reading about ML + trading, I noticed a LOT of tutorials
online do this sketchy thing where they shuffle their training data
randomly, which secretly lets the model "peek" into the future. Their
backtests look amazing... and are completely useless in real life.

So I made sure to avoid that:
- **Chronological train/test split** — the model only ever trains on
  older data and gets tested on data it's never seen, like it would in
  the real world.
- **Predicting direction, not price** — predicting the exact next-day
  price is a trap; a lazy model that just guesses "same as today" scores
  great on paper. Predicting up/down is the honest test of whether the
  model knows *anything*.
- **Transaction costs included** — because a strategy that only wins
  before fees isn't actually a strategy.

## 📊 What I actually found

Running this on AAPL, my model landed around **51–59% directional
accuracy** depending on exactly which time window I tested on (50% =
literally a coin flip). That range itself was kind of a wake-up call —
it means the result isn't super stable, which is *exactly* the kind of
thing that should make you suspicious of any single backtest number
you see online. If someone shows you one great backtest and stops
there, ask what happens on a different date range. 👀

Basically: the model found a tiny bit of signal in some setups, but
nothing you'd bet real money on. Which, honestly, matches what
efficient market theory would predict! I think that's a more
interesting takeaway than pretending I built a money-printing machine.

## 🛠️ Tech stack

- `yfinance` — pulling real historical price data
- `pandas` / `numpy` — data wrangling
- `ta` — technical indicators (RSI, MACD, moving averages, etc.)
- `scikit-learn` — Random Forest classifier
- `matplotlib` / `plotly` — static + interactive charts
- `streamlit` — the live dashboard

## 📁 Project structure

```
stock-predictor/
├── app.py                # 🖥️ Interactive Streamlit dashboard
├── main.py                # Runs the full pipeline end-to-end (CLI version)
├── fetch_data.py           # Pulls price history
├── features.py             # Technical indicators + labels
├── model.py                # Random Forest classifier
├── backtest.py             # Simulates strategy vs. buy & hold
├── plot_results.py         # Static chart generation
├── requirements.txt
├── data/                   # Downloaded price CSVs
├── models/                 # Saved trained models
└── outputs/                # Generated charts
```

## ⚙️ Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Then either run the CLI version:
```bash
python main.py AAPL
```

...or fire up the dashboard:
```bash
streamlit run app.py
```

## 🧠 How to read your results

If you run this on your own ticker, here's the honest cheat sheet:

- **~50% accuracy** → totally normal! The model didn't find an edge.
  This is expected and doesn't mean you did anything wrong.
- **50–55%** → maybe a tiny statistical whisper of something, but
  easily just noise. Don't trust it on one ticker/window alone.
- **55%+** → cool, but be skeptical before celebrating — this is
  usually a sign of subtle data leakage somewhere, not a genuine edge.
  I'd double check the feature engineering before believing it.

## 🙋 Disclaimer

I'm a student building this to learn, not a financial advisor, and
this is NOT trading advice. Please don't put real money behind this
lol. If you want to actually learn from it, the real value is in
poking at *why* it works or doesn't — that's where I learned the most.

## 💡 Ideas for extending this

Stuff I want to try next if I keep working on this:
- Testing across way more tickers to see if accuracy is ticker-dependent
- Adding sentiment analysis from news headlines
- Walk-forward validation instead of one single train/test split
- Maybe trying an LSTM just to see if it actually beats the Random Forest
  (my hunch is it won't, for this kind of tabular data, but that'd be a
  cool thing to actually test and write up)

Thanks for checking this out! Feel free to fork it, break it, improve
it, or just poke around. 🚀
