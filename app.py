"""
app.py

Interactive web dashboard for the stock direction predictor.
Run with: streamlit run app.py
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

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Signal / Terminal",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="collapsed",
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
    max-width: 620px;
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

/* Feature bars */
.feat-row { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
.feat-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-dim);
    width: 130px;
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

/* Cursor blink accent next to title */
.blink { animation: blink 1.4s step-start infinite; color: var(--amber); }
@keyframes blink { 50% { opacity: 0; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

FEATURE_LABELS = {
    "sma_10": "SMA (10d)", "sma_50": "SMA (50d)", "ema_10": "EMA (10d)",
    "rsi_14": "RSI (14d)", "macd": "MACD", "macd_signal": "MACD Signal",
    "volatility_10": "Volatility", "volume_change": "Volume Δ",
    "return_lag_1": "Return t-1", "return_lag_2": "Return t-2",
    "return_lag_3": "Return t-3", "return_lag_5": "Return t-5",
}

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown('<div class="term-eyebrow">// directional signal backtester</div>', unsafe_allow_html=True)
st.markdown('<div class="term-title">Does the model know anything<span class="blink">_</span></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="term-sub">Trains a model to predict next-day price direction from technical '
    'indicators, then backtests it against buy-and-hold on data it never saw. '
    'This is a research tool, not trading advice — read the verdict below every run.</div>',
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Controls
# ------------------------------------------------------------------
col_input, col_years, col_thresh, col_btn = st.columns([2, 1, 1, 1])

with col_input:
    st.markdown('<div class="term-prompt-label">$ ticker</div>', unsafe_allow_html=True)
    ticker = st.text_input("ticker", value="AAPL", label_visibility="collapsed").strip().upper()

with col_years:
    st.markdown('<div class="term-prompt-label">history (yrs)</div>', unsafe_allow_html=True)
    years = st.slider("years", 2, 10, 5, label_visibility="collapsed")

with col_thresh:
    st.markdown('<div class="term-prompt-label">buy threshold</div>', unsafe_allow_html=True)
    threshold = st.slider("threshold", 0.45, 0.65, 0.50, step=0.01, label_visibility="collapsed")

with col_btn:
    st.markdown('<div class="term-prompt-label">&nbsp;</div>', unsafe_allow_html=True)
    run_clicked = st.button("Run analysis")

# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------
if run_clicked and ticker:
    with st.spinner(f"Fetching {ticker}, engineering features, training..."):
        try:
            csv_path = fetch(ticker, years)
            df = load_price_data(csv_path)
            feats = build_features(df)

            train_df, test_df = train_test_split_chronological(feats, test_frac=0.2)
            model, scaler = train_model(train_df)

            predicted_up_prob = predict(model, scaler, test_df)
            results = run_backtest(test_df, predicted_up_prob, threshold=threshold)
            summary = summarize_backtest(results)
            importance = feature_importance_report(model)

            st.session_state["results"] = results
            st.session_state["summary"] = summary
            st.session_state["importance"] = importance
            st.session_state["ticker"] = ticker
            st.session_state["train_df"] = train_df
            st.session_state["test_df"] = test_df
        except Exception as e:
            st.error(f"Couldn't complete the run: {e}")
            st.session_state.pop("results", None)

# ------------------------------------------------------------------
# Results
# ------------------------------------------------------------------
if "results" in st.session_state:
    results = st.session_state["results"]
    summary = st.session_state["summary"]
    importance = st.session_state["importance"]
    tkr = st.session_state["ticker"]
    train_df = st.session_state["train_df"]
    test_df = st.session_state["test_df"]

    acc = summary["directional_accuracy"]
    strat_ret = summary["strategy_total_return"]
    bh_ret = summary["buy_hold_total_return"]

    st.markdown(f'<div class="sec-head">// {tkr} — test period: {summary["test_days"]} trading days '
                f'({test_df.index[0].date()} to {test_df.index[-1].date()})</div>', unsafe_allow_html=True)

    # --- Metric cards ---
    def fmt_pct(x):
        return f"{x*100:.1f}%"

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card accent">
            <div class="metric-label">Directional Accuracy</div>
            <div class="metric-value">{acc*100:.1f}%</div>
            <div class="metric-note">50% = coin flip</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Strategy Return</div>
            <div class="metric-value {'pos' if strat_ret >= 0 else 'neg'}">{fmt_pct(strat_ret)}</div>
            <div class="metric-note">out-of-sample</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Buy &amp; Hold Return</div>
            <div class="metric-value {'pos' if bh_ret >= 0 else 'neg'}">{fmt_pct(bh_ret)}</div>
            <div class="metric-note">same period</div>
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

    # --- Verdict ---
    if acc < 0.53:
        tag_class, tag_text = "tag-neutral", "expected result"
        verdict = (
            f"Directional accuracy of {acc*100:.1f}% is statistically close to a coin flip. "
            f"This is the common, honest outcome for short-term price direction — daily moves "
            f"are dominated by noise, and this model doesn't appear to have found a real edge "
            f"on {tkr} over this window. Treat this as a learning exercise, not a signal."
        )
    elif acc < 0.56:
        tag_class, tag_text = "tag-neutral", "marginal edge"
        verdict = (
            f"Accuracy of {acc*100:.1f}% is a little above coin-flip, but within the range you'd "
            f"expect from randomness or from the specific historical window chosen. Don't trust "
            f"this without testing across more tickers and time periods."
        )
    else:
        tag_class, tag_text = "tag-caution", "verify before trusting"
        verdict = (
            f"Accuracy of {acc*100:.1f}% is high enough to be worth a second look — but be "
            f"suspicious. Results this strong for short-term direction more often indicate subtle "
            f"data leakage (a feature that inadvertently 'sees' the future) than a genuine edge. "
            f"Double-check the feature engineering before believing it."
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
    fig.update_layout(
        plot_bgcolor="#0B0E11", paper_bgcolor="#0B0E11",
        font=dict(family="IBM Plex Mono, monospace", color="#8B949E", size=12),
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#1B2027", showline=False),
        yaxis=dict(gridcolor="#1B2027", showline=False, tickformat=".2f"),
        height=380,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Feature importance ---
    st.markdown('<div class="sec-head">// what the model is actually looking at</div>', unsafe_allow_html=True)

    max_imp = importance["importance"].max()
    rows_html = ""
    for _, row in importance.head(8).iterrows():
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
        chronological split, no shuffling · transaction cost modeled at 0.05% per position change
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="verdict-box" style="margin-top: 2rem;">
        <span class="verdict-tag tag-neutral">standing by</span><br/>
        Enter a ticker above and hit <b>Run analysis</b>. The model trains on the older 80% of
        the price history and is tested only on the most recent 20% it has never seen —
        the honest way to evaluate whether it actually learned anything.
    </div>
    """, unsafe_allow_html=True)
