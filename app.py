"""
app.py

Interactive web dashboard for the stock direction predictor.
Run with: streamlit run app.py

Three modes, driven by what you type in the ticker box:
    AAPL                 -> single out-of-sample backtest
    AAPL + walk-forward  -> accuracy distribution across many folds
    AAPL, MSFT, TSLA     -> side-by-side comparison across tickers
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from fetch_data import fetch
from features import load_price_data, build_features
from model import (
    train_test_split_chronological, train_model, predict,
    feature_importance_report
)
from backtest import run_backtest, summarize_backtest
from walk_forward import run_walk_forward, summarize_walk_forward

# ------------------------------------------------------------------
# Site identity
#
# Edit these three lines to change the byline and footer links. They are
# referenced from the header byline, the About tab and the site footer.
# ------------------------------------------------------------------
AUTHOR_NAME = "Daksh Goswami"
GITHUB_URL = "https://github.com/MyDakshAI/Stock-Price-Predictor"
LINKEDIN_URL = "https://www.linkedin.com/in/daksh-goswami-3077aa280"

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Does the model know anything?",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get Help": GITHUB_URL,
        "Report a bug": f"{GITHUB_URL}/issues",
        "About": f"A directional signal backtester built by {AUTHOR_NAME}. "
                  f"Research and learning project, not financial advice.",
    },
)

# ------------------------------------------------------------------
# Design tokens & CSS
# ------------------------------------------------------------------
CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500&display=swap" rel="stylesheet">

<style>
:root {
    --bg:        #0B0E11;
    --panel:     #14181D;
    --panel-2:   #1B2027;
    --border:    #262C34;
    --amber:     #E8A33D;
    --amber-dim: #8A6A2F;
    --green:     #3FB950;
    --red:       #F85149;
    --text:      #D6DBE1;
    --text-dim:  #6E7681;
}

/* Base */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Sans', sans-serif;
}
.stApp { background-color: var(--bg); }
#MainMenu, footer, header { visibility: hidden; }

/* Streamlit's default top padding wastes a lot of above-the-fold space,
   which matters most on phones arriving from a shared link. */
.block-container { padding-top: 2.5rem !important; }

@media (max-width: 640px) {
    .block-container { padding-top: 1.25rem !important; }
    .term-title { font-size: 2rem; }
    .term-sub { font-size: 0.92rem; margin-bottom: 1.4rem; }
    .byline { margin-top: -0.9rem; }
    .metric-card { min-width: 100%; }
    .site-footer { flex-direction: column; gap: 18px; }
    .feat-name { width: 110px; font-size: 0.72rem; }
}

/* Headline */
.term-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--amber);
    font-size: 0.78rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.term-title {
    font-family: 'Fraunces', serif;
    font-weight: 500;
    font-size: 2.6rem;
    color: #F2F4F7;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}
.term-sub {
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--text-dim);
    font-size: 0.98rem;
    max-width: 680px;
    margin-bottom: 1.8rem;
}

/* Terminal prompt input styling */
.term-prompt-label {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--amber-dim);
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}
div[data-testid="stTextInput"] input {
    background-color: var(--panel) !important;
    border: 1px solid var(--border) !important;
    color: var(--amber) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.2rem !important;
    padding: 0.7rem 0.9rem !important;
    border-radius: 4px !important;
    caret-color: var(--amber);
}
div[data-testid="stTextInput"] input:focus {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 1px var(--amber-dim) !important;
}

/* Sliders */
div[data-testid="stSlider"] label {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--text-dim);
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Checkboxes */
div[data-testid="stCheckbox"] label p {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    color: var(--text-dim) !important;
}

/* Buttons */
div[data-testid="stButton"] button {
    background-color: var(--amber) !important;
    color: #14100A !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.6rem 1.6rem !important;
    text-transform: uppercase;
    font-size: 0.85rem !important;
    transition: background-color 0.15s ease;
    width: 100%;
}
div[data-testid="stButton"] button:hover {
    background-color: #F0B454 !important;
}

/* Metric cards */
.metric-row { display: flex; gap: 14px; margin: 1.6rem 0; flex-wrap: wrap; }
.metric-card {
    background-color: var(--panel);
    border: 1px solid var(--border);
    border-left: 3px solid var(--amber-dim);
    border-radius: 4px;
    padding: 1.1rem 1.3rem;
    flex: 1;
    min-width: 165px;
}
.metric-card.accent { border-left-color: var(--amber); }
.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.7rem;
    font-weight: 500;
    color: #F2F4F7;
}
.metric-value.pos { color: var(--green); }
.metric-value.neg { color: var(--red); }
.metric-note {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.75rem;
    color: var(--text-dim);
    margin-top: 0.3rem;
}

/* Section headers */
.sec-head {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--amber-dim);
    font-size: 0.75rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin: 2rem 0 1rem 0;
}

/* Verdict callout */
.verdict-box {
    background-color: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1.2rem 1.4rem;
    margin-top: 1rem;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.92rem;
    line-height: 1.55;
    color: var(--text);
}
.verdict-box .verdict-tag {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border-radius: 3px;
    margin-bottom: 0.7rem;
}
.tag-neutral { background-color: rgba(232,163,61,0.15); color: var(--amber); }
.tag-caution { background-color: rgba(248,81,73,0.15); color: var(--red); }
.tag-good    { background-color: rgba(63,185,80,0.15); color: var(--green); }

/* Feature bars */
.feat-row { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
.feat-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-dim);
    width: 150px;
    flex-shrink: 0;
}
.feat-bar-track { flex: 1; background-color: var(--panel-2); border-radius: 2px; height: 8px; overflow: hidden; }
.feat-bar-fill { background-color: var(--amber-dim); height: 100%; }
.feat-val { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: var(--text-dim); width: 42px; text-align: right; }

/* Footer note */
.foot-note {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-dim);
    margin-top: 3rem;
    padding-top: 1.2rem;
    border-top: 1px solid var(--border);
    line-height: 1.6;
}

/* Dataframe restyle */
div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 4px; }

/* Plain-English explainer panel */
div[data-testid="stExpander"] {
    background-color: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    margin-bottom: 1.6rem;
}
div[data-testid="stExpander"] summary {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.06em;
}
div[data-testid="stExpander"] summary p {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    color: var(--amber) !important;
}
.explainer {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.92rem;
    line-height: 1.65;
    color: var(--text);
}
.explainer p { margin-bottom: 0.9rem; }
.explainer h4 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: var(--amber-dim);
    margin: 1.5rem 0 0.6rem 0;
}
.explainer h4:first-child { margin-top: 0; }
.gloss { display: flex; gap: 14px; margin-bottom: 9px; align-items: baseline; }
.gloss b {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--amber-dim);
    min-width: 178px;
    flex-shrink: 0;
    font-weight: 500;
}
.gloss span { color: var(--text-dim); font-size: 0.88rem; line-height: 1.5; }
@media (max-width: 640px) {
    .gloss { flex-direction: column; gap: 2px; }
    .gloss b { min-width: 0; }
}

/* Cursor blink accent next to title */
.blink { animation: blink 1.4s step-start infinite; color: var(--amber); }
@keyframes blink { 50% { opacity: 0; } }

/* Byline under the header */
.byline {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-dim);
    margin-top: -1.2rem;
    margin-bottom: 1.8rem;
    letter-spacing: 0.02em;
}
.byline a {
    color: var(--amber-dim);
    text-decoration: none;
    border-bottom: 1px solid rgba(138,106,47,0.35);
    transition: color 0.15s ease, border-color 0.15s ease;
}
.byline a:hover { color: var(--amber); border-bottom-color: var(--amber); }
.byline .sep { color: var(--border); margin: 0 0.55rem; }

/* Info tabs */
div[data-testid="stTabs"] { margin-top: 1rem; }
div[data-testid="stTabs"] button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.76rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-dim) !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] { color: var(--amber) !important; }
div[data-testid="stTabs"] div[data-baseweb="tab-highlight"] { background-color: var(--amber) !important; }

/* Site footer */
.site-footer {
    margin-top: 3.5rem;
    padding-top: 1.6rem;
    border-top: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    gap: 24px;
    flex-wrap: wrap;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-dim);
    line-height: 1.7;
}
.site-footer a {
    color: var(--amber-dim);
    text-decoration: none;
    border-bottom: 1px solid rgba(138,106,47,0.35);
}
.site-footer a:hover { color: var(--amber); border-bottom-color: var(--amber); }
.site-footer .col { max-width: 380px; }
.site-footer .col-title {
    color: var(--amber-dim);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-size: 0.68rem;
    margin-bottom: 0.5rem;
}
.site-footer .disclaim { color: #4E555E; font-size: 0.72rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

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

CHART_LAYOUT = dict(
    plot_bgcolor="#0B0E11", paper_bgcolor="#0B0E11",
    font=dict(family="IBM Plex Mono, monospace", color="#8B949E", size=12),
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(gridcolor="#1B2027", showline=False),
    yaxis=dict(gridcolor="#1B2027", showline=False),
    hovermode="x unified",
)


# ------------------------------------------------------------------
# Cached pipeline steps
#
# Everything expensive (downloading, feature building, training) is
# cached so that moving the threshold slider re-runs only the cheap
# backtest instead of retraining the whole model.
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def load_features(ticker: str, years: int, use_market: bool) -> pd.DataFrame:
    csv_path = fetch(ticker, years)
    df = load_price_data(csv_path)

    market_df = None
    if use_market:
        market_csv = fetch("SPY", years)
        market_df = load_price_data(market_csv)

    return build_features(df, market_df=market_df)


@st.cache_data(show_spinner=False, ttl=3600)
def train_and_predict(ticker: str, years: int, use_market: bool):
    """Returns everything that does NOT depend on the buy threshold,
    so threshold changes stay instant."""
    feats = load_features(ticker, years, use_market)
    train_df, test_df = train_test_split_chronological(feats, test_frac=0.2)
    model, scaler = train_model(train_df)
    predicted_up_prob = predict(model, scaler, test_df)
    importance = feature_importance_report(model)
    return train_df, test_df, predicted_up_prob, importance


@st.cache_data(show_spinner=False, ttl=3600)
def cached_walk_forward(ticker: str, years: int, use_market: bool,
                         n_folds: int, threshold: float) -> pd.DataFrame:
    feats = load_features(ticker, years, use_market)
    return run_walk_forward(feats, n_folds=n_folds, threshold=threshold)


def fmt_pct(x):
    return f"{x*100:.1f}%"


# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown('<div class="term-eyebrow">// directional signal backtester</div>', unsafe_allow_html=True)
st.markdown('<div class="term-title">Does the model know anything<span class="blink">_</span></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="term-sub">Trains a model to predict next-day price direction from technical '
    'indicators, then tests it on data it never saw against two baselines: buy-and-hold, and the '
    'naive "always guess up" rule that beats 50% on most stocks. '
    'This is a research tool, not trading advice; read the verdict below every run.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="byline">Built by {AUTHOR_NAME}'
    f'<span class="sep">/</span><a href="{GITHUB_URL}" target="_blank">Source on GitHub</a>'
    f'<span class="sep">/</span><a href="{LINKEDIN_URL}" target="_blank">LinkedIn</a>'
    f'</div>',
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Plain-English explainer
#
# Most people arriving from a shared link have no background in either
# machine learning or markets. Everything below is written for them, in
# a panel they can open rather than a wall of text they have to scroll past.
# ------------------------------------------------------------------
with st.expander("New here? What am I actually looking at?"):
    st.markdown("""
    <div class="explainer">

    <h4>The idea in one paragraph</h4>
    <p>Every trading day, a stock closes either higher or lower than the day before.
    This tool trains a computer model to guess which of those two things will happen
    tomorrow, using only patterns it can find in past prices and trading volume. Then it
    does the important part: it scores how often that guess was right across a stretch
    of days the model had never seen while it was learning.</p>

    <h4>Why beating 50% means nothing</h4>
    <p>A coin flip gets 50%, so that sounds like the number to beat. It isn't. Stocks
    drift upward over time, so a rule as dumb as <i>"always guess up, every single day"</i>
    already scores around 53% on most stocks without knowing anything at all. That dumb
    rule is the real bar, and it is what this dashboard measures against. A model that
    scores 52% has not found a signal, it has found a worse version of guessing.</p>

    <h4>What the numbers mean</h4>
    <div class="gloss"><b>Directional accuracy</b><span>How often the model correctly
    guessed tomorrow's direction.</span></div>
    <div class="gloss"><b>Edge vs. baseline</b><span>How far above or below the dumb
    "always guess up" rule it landed, in percentage points. A negative number means the
    model is worse than guessing.</span></div>
    <div class="gloss"><b>p-value</b><span>The probability of seeing an edge this large
    purely by luck. Below 0.05 is the usual bar for "this is probably not a coincidence".</span></div>
    <div class="gloss"><b>ROC-AUC</b><span>How well the model separates up-days from
    down-days overall. 0.50 means no signal whatsoever, 1.00 would be perfect.</span></div>
    <div class="gloss"><b>Strategy return</b><span>What your money would have done over the
    test period if you had followed the model's buy signals, after trading fees.</span></div>
    <div class="gloss"><b>Buy &amp; hold</b><span>What your money would have done if you had
    simply bought the stock at the start and done nothing. The comparison that matters.</span></div>
    <div class="gloss"><b>Sharpe ratio</b><span>Return adjusted for how bumpy the ride was.
    Higher is better. Two strategies can earn the same amount while one is far more stressful
    to hold.</span></div>
    <div class="gloss"><b>Max drawdown</b><span>The worst peak-to-trough drop along the way.
    Closer to zero is better.</span></div>

    <h4>What you should expect to see</h4>
    <p>Usually, no edge. That is the honest answer and not a malfunction. Short-term price
    moves are dominated by noise, and if a straightforward model could reliably predict
    tomorrow's direction, the firms with supercomputers and PhD teams would have traded that
    opportunity away long before it reached a laptop.</p>
    <p>So if a run comes back looking spectacular, the first suspect is a bug rather than a
    discovery, usually one where the model accidentally got a peek at information from the
    future. This tool is built to catch that, and it will tell you to be suspicious rather
    than congratulate you.</p>

    <h4>The two checkboxes</h4>
    <p><b>Add SPY market features</b> lets the model see what the S&amp;P 500 (the overall US
    market) did recently, not just this one stock. Most of what moves a single stock on a
    given day is the whole market moving.</p>
    <p><b>Walk-forward validation</b> is the rigorous mode. Instead of testing once, it
    retrains and retests across several different stretches of history and shows you the
    spread of results. This matters because any single test period can flatter or punish a
    model by luck, and seeing the spread is the fastest way to tell a real pattern from a
    coincidence.</p>

    <h4>Can I trade with this?</h4>
    <p>No. It is a learning project built to understand how markets and machine learning
    interact, and it is not financial advice.</p>

    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# Controls
# ------------------------------------------------------------------
col_input, col_years, col_thresh, col_btn = st.columns([2.4, 1, 1, 1])

with col_input:
    st.markdown('<div class="term-prompt-label">$ ticker (comma-separate to compare)</div>', unsafe_allow_html=True)
    ticker_raw = st.text_input("ticker", value="AAPL", label_visibility="collapsed")

with col_years:
    st.markdown('<div class="term-prompt-label">history (yrs)</div>', unsafe_allow_html=True)
    years = st.slider("years", 2, 10, 5, label_visibility="collapsed")

with col_thresh:
    st.markdown('<div class="term-prompt-label">buy threshold</div>', unsafe_allow_html=True)
    threshold = st.slider("threshold", 0.45, 0.65, 0.50, step=0.01, label_visibility="collapsed")

with col_btn:
    st.markdown('<div class="term-prompt-label">&nbsp;</div>', unsafe_allow_html=True)
    run_clicked = st.button("Run analysis")

opt1, opt2, opt3 = st.columns([1.2, 1.6, 3])
with opt1:
    use_market = st.checkbox("Add SPY market features", value=True,
                              help="Adds the S&P 500's recent return and this stock's strength "
                                   "relative to it. A single stock's direction is heavily driven "
                                   "by the whole market, so this is usually the highest-value feature.")
with opt2:
    walk_forward_mode = st.checkbox("Walk-forward validation", value=False,
                                     help="Instead of one train/test split, retrain across several "
                                          "chronological folds and report the distribution of accuracy. "
                                          "Slower, but far more honest than a single number.")

tickers = [t.strip().upper() for t in ticker_raw.split(",") if t.strip()]
compare_mode = len(tickers) > 1

# Run the default analysis automatically the first time someone opens the
# page, so a visitor arriving from a shared link lands on a finished result
# instead of an empty screen and a button they have to find.
first_load = not st.session_state.get("_has_auto_run", False)
if first_load:
    st.session_state["_has_auto_run"] = True

# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------
if (run_clicked or first_load) and tickers:
    st.session_state.pop("payload", None)
    try:
        if compare_mode:
            rows = []
            progress = st.progress(0.0, text="Running comparison...")
            for i, tkr in enumerate(tickers[:8]):  # cap at 8 to keep it responsive
                progress.progress(i / min(len(tickers), 8), text=f"Analyzing {tkr}...")
                if walk_forward_mode:
                    folds = cached_walk_forward(tkr, years, use_market, 5, threshold)
                    wf = summarize_walk_forward(folds)
                    rows.append({
                        "Ticker": tkr,
                        "Mean accuracy": wf["mean_accuracy"],
                        "Std dev": wf["std_accuracy"],
                        "Naive baseline": wf["mean_naive_baseline"],
                        "Beats baseline": f"{wf['folds_beating_naive_baseline']}/{wf['n_folds']} folds",
                        "Mean strategy return": wf["mean_strategy_return"],
                        "Mean buy & hold": wf["mean_buy_hold_return"],
                    })
                else:
                    train_df, test_df, prob, _ = train_and_predict(tkr, years, use_market)
                    results = run_backtest(test_df, prob, threshold=threshold)
                    s = summarize_backtest(results, test_df)
                    rows.append({
                        "Ticker": tkr,
                        "Accuracy": s["directional_accuracy"],
                        "Naive baseline": s["naive_baseline_accuracy"],
                        "ROC-AUC": s["roc_auc"],
                        "p-value": s["significance_vs_naive"]["p_value"],
                        "Strategy return": s["strategy_total_return"],
                        "Buy & hold": s["buy_hold_total_return"],
                        "Sharpe": s["strategy_sharpe"],
                    })
            progress.empty()
            st.session_state["payload"] = {"mode": "compare", "rows": pd.DataFrame(rows),
                                           "walk_forward": walk_forward_mode}

        elif walk_forward_mode:
            with st.spinner(f"Running walk-forward validation on {tickers[0]}..."):
                folds = cached_walk_forward(tickers[0], years, use_market, 5, threshold)
                st.session_state["payload"] = {
                    "mode": "walk_forward", "ticker": tickers[0],
                    "folds": folds, "summary": summarize_walk_forward(folds),
                }

        else:
            with st.spinner(f"Fetching {tickers[0]}, engineering features, training..."):
                train_df, test_df, prob, importance = train_and_predict(tickers[0], years, use_market)
                results = run_backtest(test_df, prob, threshold=threshold)
                summary = summarize_backtest(results, test_df)
                st.session_state["payload"] = {
                    "mode": "single", "ticker": tickers[0], "results": results,
                    "summary": summary, "importance": importance,
                    "train_df": train_df, "test_df": test_df,
                }

    except Exception as e:
        # Yahoo Finance rate-limits shared cloud IPs from time to time, and
        # a mistyped ticker is the other common cause. Both deserve a plain
        # explanation rather than a raw traceback.
        message = str(e)
        if "No data returned" in message:
            st.error(f"No price data came back for that ticker. Check the symbol is right "
                     f"(for example AAPL, not Apple), or try again in a moment if Yahoo "
                     f"Finance is rate-limiting requests.")
        else:
            st.error(f"Couldn't complete the run: {message}")
        st.session_state.pop("payload", None)

# ------------------------------------------------------------------
# Results
# ------------------------------------------------------------------
payload = st.session_state.get("payload")

if payload and payload["mode"] == "single":
    results = payload["results"]
    summary = payload["summary"]
    importance = payload["importance"]
    tkr = payload["ticker"]
    train_df, test_df = payload["train_df"], payload["test_df"]

    acc = summary["directional_accuracy"]
    naive = summary["naive_baseline_accuracy"]
    strat_ret = summary["strategy_total_return"]
    bh_ret = summary["buy_hold_total_return"]
    sig = summary["significance_vs_naive"]

    st.markdown(f'<div class="sec-head">// {tkr}, test period: {summary["test_days"]} trading days '
                f'({test_df.index[0].date()} to {test_df.index[-1].date()})</div>', unsafe_allow_html=True)

    edge = acc - naive
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card accent">
            <div class="metric-label">Directional Accuracy</div>
            <div class="metric-value">{acc*100:.1f}%</div>
            <div class="metric-note">naive baseline: {naive*100:.1f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Edge vs. Baseline</div>
            <div class="metric-value {'pos' if edge > 0 else 'neg'}">{edge*100:+.1f}pp</div>
            <div class="metric-note">p = {sig['p_value']:.3f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">ROC-AUC</div>
            <div class="metric-value">{summary['roc_auc']:.3f}</div>
            <div class="metric-note">0.50 = no signal</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Strategy Return</div>
            <div class="metric-value {'pos' if strat_ret >= 0 else 'neg'}">{fmt_pct(strat_ret)}</div>
            <div class="metric-note">buy &amp; hold: {fmt_pct(bh_ret)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Sharpe (Strategy)</div>
            <div class="metric-value">{summary['strategy_sharpe']:.2f}</div>
            <div class="metric-note">buy/hold: {summary['buy_hold_sharpe']:.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value neg">{fmt_pct(summary['strategy_max_drawdown'])}</div>
            <div class="metric-note">buy/hold: {fmt_pct(summary['buy_hold_max_drawdown'])}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Verdict, judged against the naive baseline, not 50% ---
    if acc <= naive:
        tag_class, tag_text = "tag-neutral", "no edge found"
        verdict = (
            f"Accuracy of {acc*100:.1f}% does not beat the naive 'always predict up' baseline of "
            f"{naive*100:.1f}% on this window. That baseline matters: {tkr} rose on "
            f"{naive*100:.1f}% of days in this test period, so a model has to clear that bar, not "
            f"just 50%, to have learned anything. This is the common and honest outcome for "
            f"short-term price direction, where daily moves are dominated by noise."
        )
    elif not sig["significant_at_0.05"]:
        tag_class, tag_text = "tag-neutral", "within noise"
        verdict = (
            f"Accuracy of {acc*100:.1f}% edges past the naive baseline of {naive*100:.1f}%, but a "
            f"binomial test gives p = {sig['p_value']:.3f}, so over {summary['test_days']} days this "
            f"gap is well within what randomness produces. Run walk-forward validation to see whether "
            f"it holds up across multiple time windows before reading anything into it."
        )
    elif acc - naive > 0.06:
        tag_class, tag_text = "tag-caution", "suspiciously strong"
        verdict = (
            f"Accuracy of {acc*100:.1f}% against a {naive*100:.1f}% baseline (p = {sig['p_value']:.3f}) "
            f"is strong enough that the first thing to suspect is data leakage, not skill. Results "
            f"this good for next-day direction usually mean a feature is inadvertently seeing the "
            f"future. Check the feature engineering before believing it, and confirm with walk-forward."
        )
    else:
        tag_class, tag_text = "tag-good", "statistically significant"
        verdict = (
            f"Accuracy of {acc*100:.1f}% beats the naive baseline of {naive*100:.1f}% with "
            f"p = {sig['p_value']:.3f}, so the gap is unlikely to be pure chance on this window. "
            f"That is a real result worth investigating, but one window is one data point: run "
            f"walk-forward validation and try other tickers before trusting it."
        )

    st.markdown(f"""
    <div class="verdict-box">
        <span class="verdict-tag {tag_class}">{tag_text}</span><br/>
        {verdict}
    </div>
    """, unsafe_allow_html=True)

    # --- Chart: cumulative returns ---
    st.markdown('<div class="sec-head">// cumulative growth of $1</div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results.index, y=results["strategy_cumulative"],
        mode="lines", name="Model strategy",
        line=dict(color="#E8A33D", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=results.index, y=results["buy_hold_cumulative"],
        mode="lines", name="Buy & hold",
        line=dict(color="#6E7681", width=1.5, dash="dot"),
    ))
    fig.update_layout(**CHART_LAYOUT, height=380)
    fig.update_yaxes(tickformat=".2f")
    st.plotly_chart(fig, width="stretch")

    # --- Calibration ---
    st.markdown('<div class="sec-head">// are the confidence scores trustworthy</div>', unsafe_allow_html=True)

    cal = pd.DataFrame(summary["calibration_bins"])
    if not cal.empty:
        cal_fig = go.Figure()
        cal_fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
            line=dict(color="#6E7681", width=1, dash="dot"),
        ))
        cal_fig.add_trace(go.Scatter(
            x=cal["avg_predicted"], y=cal["actual_up_rate"],
            mode="markers+lines", name="This model",
            marker=dict(color="#E8A33D", size=9),
            line=dict(color="#E8A33D", width=2),
        ))
        cal_fig.update_layout(**CHART_LAYOUT, height=300)
        cal_fig.update_xaxes(title_text="predicted probability of up", range=[0, 1])
        cal_fig.update_yaxes(title_text="actual fraction that went up", range=[0, 1])
        st.plotly_chart(cal_fig, width="stretch")

        st.markdown(
            f'<div class="verdict-box">Brier score {summary["brier_score"]:.4f} (lower is better). '
            f'If the amber line tracks the dotted line, days the model calls 60% really do go up '
            f'about 60% of the time, which is what makes the buy-threshold slider meaningful. '
            f'If it is flat, the model is expressing confidence it has not earned.</div>',
            unsafe_allow_html=True,
        )

    # --- Feature importance ---
    st.markdown('<div class="sec-head">// what the model is actually looking at</div>', unsafe_allow_html=True)

    max_imp = importance["importance"].max()
    rows_html = ""
    for _, row in importance.head(10).iterrows():
        pct = row["importance"] / max_imp * 100
        label = FEATURE_LABELS.get(row["feature"], row["feature"])
        rows_html += f"""
        <div class="feat-row">
            <div class="feat-name">{label}</div>
            <div class="feat-bar-track"><div class="feat-bar-fill" style="width:{pct}%"></div></div>
            <div class="feat-val">{row['importance']:.3f}</div>
        </div>
        """
    st.markdown(rows_html, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="foot-note">
        trained on {len(train_df)} days · tested on {len(test_df)} days (out-of-sample) ·
        chronological split, no shuffling · transaction cost modeled at 0.05% per position change ·
        {'SPY market features included' if use_market else 'ticker-only features'}
    </div>
    """, unsafe_allow_html=True)

elif payload and payload["mode"] == "walk_forward":
    folds = payload["folds"]
    wf = payload["summary"]
    tkr = payload["ticker"]

    st.markdown(f'<div class="sec-head">// {tkr}, walk-forward validation across '
                f'{wf["n_folds"]} chronological folds</div>', unsafe_allow_html=True)

    beat = wf["folds_beating_naive_baseline"]
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card accent">
            <div class="metric-label">Mean Accuracy</div>
            <div class="metric-value">{wf['mean_accuracy']*100:.1f}%</div>
            <div class="metric-note">± {wf['std_accuracy']*100:.1f}pp across folds</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Range</div>
            <div class="metric-value">{wf['min_accuracy']*100:.0f}-{wf['max_accuracy']*100:.0f}%</div>
            <div class="metric-note">worst to best fold</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Naive Baseline</div>
            <div class="metric-value">{wf['mean_naive_baseline']*100:.1f}%</div>
            <div class="metric-note">mean across folds</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Folds Beating Baseline</div>
            <div class="metric-value {'pos' if beat > wf['n_folds']/2 else 'neg'}">{beat}/{wf['n_folds']}</div>
            <div class="metric-note">coin flip would give ~50%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Mean Strategy Return</div>
            <div class="metric-value {'pos' if wf['mean_strategy_return'] >= 0 else 'neg'}">{fmt_pct(wf['mean_strategy_return'])}</div>
            <div class="metric-note">per fold, buy &amp; hold: {fmt_pct(wf['mean_buy_hold_return'])}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if beat <= wf["n_folds"] / 2:
        tag_class, tag_text = "tag-neutral", "no stable edge"
        wf_verdict = (
            f"Across {wf['n_folds']} independent time windows the model beat the naive baseline in "
            f"only {beat} of them, and accuracy swung {wf['min_accuracy']*100:.0f}% to "
            f"{wf['max_accuracy']*100:.0f}%. That spread is the whole point of this view: any single "
            f"backtest from this range could be cherry-picked to look like success or failure. "
            f"The honest read is that there is no stable edge here."
        )
    else:
        tag_class, tag_text = "tag-good", "consistent across folds"
        wf_verdict = (
            f"The model beat the naive baseline in {beat} of {wf['n_folds']} independent time windows, "
            f"with accuracy ranging {wf['min_accuracy']*100:.0f}% to {wf['max_accuracy']*100:.0f}%. "
            f"Consistency across folds is much stronger evidence than any single backtest, though a "
            f"± {wf['std_accuracy']*100:.1f}pp spread still means individual periods vary a lot. "
            f"Next step would be repeating this across many more tickers."
        )

    st.markdown(f"""
    <div class="verdict-box">
        <span class="verdict-tag {tag_class}">{tag_text}</span><br/>
        {wf_verdict}
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-head">// accuracy by fold vs. naive baseline</div>', unsafe_allow_html=True)

    wf_fig = go.Figure()
    wf_fig.add_trace(go.Bar(
        x=[f"Fold {int(f)}" for f in folds["fold"]], y=folds["accuracy"],
        name="Model accuracy", marker_color="#E8A33D",
    ))
    wf_fig.add_trace(go.Scatter(
        x=[f"Fold {int(f)}" for f in folds["fold"]], y=folds["naive_baseline"],
        mode="markers+lines", name="Naive baseline",
        line=dict(color="#F85149", width=1.5, dash="dot"),
        marker=dict(color="#F85149", size=8),
    ))
    wf_fig.update_layout(**CHART_LAYOUT, height=340)
    wf_fig.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(wf_fig, width="stretch")

    st.markdown('<div class="sec-head">// fold detail</div>', unsafe_allow_html=True)
    display = folds.copy()
    for col in ["accuracy", "naive_baseline", "strategy_return", "buy_hold_return"]:
        display[col] = (display[col] * 100).round(1).astype(str) + "%"
    display["roc_auc"] = display["roc_auc"].round(3)
    st.dataframe(display, width="stretch", hide_index=True)

    st.markdown(f"""
    <div class="foot-note">
        expanding training window, fixed-size test window, strictly chronological ·
        each fold retrains from scratch on only the data before its test period ·
        {'SPY market features included' if use_market else 'ticker-only features'}
    </div>
    """, unsafe_allow_html=True)

elif payload and payload["mode"] == "compare":
    rows = payload["rows"]
    st.markdown(f'<div class="sec-head">// comparison across {len(rows)} tickers'
                f'{" (walk-forward)" if payload["walk_forward"] else ""}</div>', unsafe_allow_html=True)

    acc_col = "Mean accuracy" if payload["walk_forward"] else "Accuracy"
    beat_count = (rows[acc_col] > rows["Naive baseline"]).sum()

    st.markdown(f"""
    <div class="verdict-box">
        <span class="verdict-tag {'tag-good' if beat_count > len(rows)/2 else 'tag-neutral'}">
            {beat_count}/{len(rows)} beat the naive baseline
        </span><br/>
        This is the test that matters most. An approach that finds real signal should work on more
        than one stock. If accuracy beats the baseline on some tickers and not others with no pattern,
        that is what noise looks like, not a strategy.
    </div>
    """, unsafe_allow_html=True)

    display = rows.copy()
    pct_cols = [c for c in display.columns if c in
                (acc_col, "Naive baseline", "Std dev", "Strategy return", "Buy & hold",
                 "Mean strategy return", "Mean buy & hold")]
    for col in pct_cols:
        display[col] = (display[col] * 100).round(1).astype(str) + "%"
    for col in ("ROC-AUC", "Sharpe", "p-value"):
        if col in display.columns:
            display[col] = display[col].round(3)

    st.dataframe(display, width="stretch", hide_index=True)

    chart_fig = go.Figure()
    chart_fig.add_trace(go.Bar(
        x=rows["Ticker"], y=rows[acc_col], name="Model accuracy", marker_color="#E8A33D",
    ))
    chart_fig.add_trace(go.Scatter(
        x=rows["Ticker"], y=rows["Naive baseline"], mode="markers", name="Naive baseline",
        marker=dict(color="#F85149", size=11, symbol="diamond"),
    ))
    chart_fig.update_layout(**CHART_LAYOUT, height=340)
    chart_fig.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(chart_fig, width="stretch")

    st.markdown(f"""
    <div class="foot-note">
        each ticker independently downloaded, featurized, trained and tested ·
        chronological splits only · transaction cost 0.05% per position change ·
        {'SPY market features included' if use_market else 'ticker-only features'}
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="verdict-box" style="margin-top: 2rem;">
        <span class="verdict-tag tag-neutral">standing by</span><br/>
        Enter a ticker and hit <b>Run analysis</b>. The model trains on the older 80% of the price
        history and is tested only on the most recent 20% it has never seen, the honest way to
        evaluate whether it actually learned anything.
        <br/><br/>
        Try <b>AAPL, MSFT, TSLA, NVDA</b> to compare several at once, or tick
        <b>Walk-forward validation</b> to see how much the result swings depending on which slice
        of history you test on. That swing is usually the most interesting thing here.
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------------
# Site sections
#
# Placed below the tool rather than above it: a visitor who arrived from a
# shared link should see a working result first and read about it second.
# ------------------------------------------------------------------
st.markdown('<div class="sec-head">// about this project</div>', unsafe_allow_html=True)

tab_how, tab_why, tab_about = st.tabs(["How it works", "Why it's built this way", "About"])

with tab_how:
    st.markdown(f"""
    <div class="explainer">
    <p>The pipeline runs end to end every time you hit <b>Run analysis</b>, in about five steps.</p>

    <div class="gloss"><b>1. Download</b><span>Daily price history for the ticker is pulled from
    Yahoo Finance, plus the S&amp;P 500 if market features are enabled. Results are cached for a
    day so repeat runs are instant.</span></div>
    <div class="gloss"><b>2. Build features</b><span>Around twenty technical indicators are computed
    from price and volume: moving averages, RSI, MACD, Bollinger Bands, ATR, a stochastic
    oscillator, on-balance volume, recent returns, and how the stock is moving relative to the
    wider market.</span></div>
    <div class="gloss"><b>3. Split by time</b><span>The oldest 80% becomes training data, the newest
    20% becomes the test set. Never shuffled, so the model only ever learns from the past and is
    judged on its future.</span></div>
    <div class="gloss"><b>4. Train</b><span>A Random Forest classifier learns to map those indicators
    to the next day's direction. It never touches the test period.</span></div>
    <div class="gloss"><b>5. Backtest and score</b><span>The model's predictions are turned into a
    simple strategy (hold the stock on predicted up-days, hold cash otherwise, minus trading fees),
    then scored against buy-and-hold and the naive baseline, with a significance test and a
    calibration check.</span></div>

    <p style="margin-top:1.2rem;">Tick <b>walk-forward validation</b> and steps 3 through 5 repeat
    across several rolling slices of history instead of once, which is the difference between one
    data point and an actual measurement.</p>
    </div>
    """, unsafe_allow_html=True)

with tab_why:
    st.markdown("""
    <div class="explainer">
    <p>Most public tutorials on machine learning for trading produce spectacular backtests that
    would lose money instantly. The usual reason is that the model was accidentally allowed to see
    the future. This project is built specifically to make that impossible, or at least to make it
    loud when it happens.</p>

    <div class="gloss"><b>No shuffling</b><span>Time series data shuffled before splitting lets a
    model train on next month to predict last month. Every split here is strictly chronological,
    in the single backtest and in every walk-forward fold.</span></div>
    <div class="gloss"><b>Scaler fit on training only</b><span>Normalizing before splitting leaks
    test-period statistics backwards into training. There is a unit test asserting this cannot
    happen.</span></div>
    <div class="gloss"><b>A look-ahead test in CI</b><span>The test suite truncates the price
    history, rebuilds every indicator, and fails if any earlier feature value changed. If an
    indicator ever reaches forward in time, the build breaks.</span></div>
    <div class="gloss"><b>The right baseline</b><span>Comparing against a 50% coin flip is the
    quiet mistake that makes noise look like skill. Everything here is measured against the naive
    "always guess up" rule, which already clears 50% on most stocks.</span></div>
    <div class="gloss"><b>Significance testing</b><span>A binomial test reports how likely an
    apparent edge is to be luck, so a small sample cannot masquerade as a discovery.</span></div>
    <div class="gloss"><b>Costs included</b><span>Trading fees are modeled at 0.05% per position
    change, because a strategy that only wins before fees is not a strategy.</span></div>

    <p style="margin-top:1.2rem;">The dashboard is also written to be skeptical of good news: if
    accuracy comes back unusually high, the verdict tells you to suspect a bug rather than
    congratulating you. A tool that only flatters you is not much of a tool.</p>
    </div>
    """, unsafe_allow_html=True)

with tab_about:
    st.markdown(f"""
    <div class="explainer">
    <p>I built this to answer a question that kept bugging me: can machine learning actually predict
    the stock market, or is that mostly hype? The honest answer turned out to be far more
    interesting than a good backtest would have been.</p>

    <p>Short-term price direction is dominated by noise. If a straightforward model running on a
    laptop could reliably call tomorrow's move, firms with supercomputers and research teams would
    have traded that opportunity away long ago. So the engineering challenge is not chasing a big
    number, it is building something rigorous enough that you would trust its answer in either
    direction, including when that answer is "this found nothing".</p>

    <p>That is what the methodology above is for, and it is the part I would want someone to look
    at. Negative results are still results when the process behind them is sound.</p>

    <p style="margin-top:1.2rem;">Built with Python, scikit-learn, pandas, Streamlit and Plotly.
    Tested with pytest and GitHub Actions.
    The full source is on <a href="{GITHUB_URL}" target="_blank">GitHub</a>, and you can find me on
    <a href="{LINKEDIN_URL}" target="_blank">LinkedIn</a>.</p>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.markdown(f"""
<div class="site-footer">
    <div class="col">
        <div class="col-title">Does the model know anything</div>
        A directional signal backtester.<br/>
        Built by {AUTHOR_NAME}.<br/>
        <a href="{GITHUB_URL}" target="_blank">Source on GitHub</a> &nbsp;
        <a href="{LINKEDIN_URL}" target="_blank">LinkedIn</a>
    </div>
    <div class="col">
        <div class="col-title">Built with</div>
        Python · scikit-learn · pandas · Streamlit · Plotly<br/>
        Tested with pytest, running in GitHub Actions<br/>
        Market data from Yahoo Finance via yfinance
    </div>
    <div class="col">
        <div class="col-title">Disclaimer</div>
        <span class="disclaim">This is a learning and research project, not financial advice and
        not a trading system. Nothing here is a recommendation to buy or sell any security. Past
        performance, including out-of-sample backtests, does not predict future results.</span>
    </div>
</div>
""", unsafe_allow_html=True)
