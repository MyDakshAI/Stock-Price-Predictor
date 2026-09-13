const COST = 0.0005;
const chartTheme = {
  color: "#8b949e",
  grid: "rgba(27,32,39,1)",
  amber: "#e8a33d",
  muted: "#6e7681",
  red: "#f85149",
};

let lastPayload = null;
let equityChart = null;
let calChart = null;
let wfChart = null;
let cmpChart = null;

const $ = (id) => document.getElementById(id);

function fmtPct(x, digits = 1) {
  if (x == null || Number.isNaN(x)) return "—";
  return `${(x * 100).toFixed(digits)}%`;
}

function clsSign(x) {
  return x >= 0 ? "pos" : "neg";
}

function destroyCharts() {
  [equityChart, calChart, wfChart, cmpChart].forEach((c) => c && c.destroy());
  equityChart = calChart = wfChart = cmpChart = null;
}

function chartDefaults() {
  Chart.defaults.font.family = "IBM Plex Mono, monospace";
  Chart.defaults.color = chartTheme.color;
  Chart.defaults.borderColor = chartTheme.grid;
}

function mean(arr) {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function stdev(arr) {
  const m = mean(arr);
  const v = mean(arr.map((x) => (x - m) ** 2));
  return Math.sqrt(v);
}

function sharpe(returns) {
  const s = stdev(returns);
  if (!Number.isFinite(s) || s < 1e-12) return 0;
  return (mean(returns) / s) * Math.sqrt(252);
}

function maxDrawdown(cum) {
  let peak = cum[0];
  let dd = 0;
  for (const x of cum) {
    peak = Math.max(peak, x);
    dd = Math.min(dd, (x - peak) / peak);
  }
  return dd;
}

function recomputeBacktest(series, threshold) {
  const n = series.dates.length;
  const stratR = [];
  const bhR = series.target_return.slice();
  const stratCum = [];
  const bhCum = [];
  let prev = 0;
  let sc = 1;
  let bc = 1;
  let correct = 0;
  let upDays = 0;

  for (let i = 0; i < n; i++) {
    const pos = series.predicted_up_prob[i] > threshold ? 1 : 0;
    const cost = Math.abs(pos - prev) * COST;
    prev = pos;
    const sr = pos * series.target_return[i] - cost;
    stratR.push(sr);
    sc *= 1 + sr;
    bc *= 1 + series.target_return[i];
    stratCum.push(sc);
    bhCum.push(bc);
    const actualUp = series.target_return[i] > 0;
    if (actualUp) upDays += 1;
    const predUp = series.predicted_up_prob[i] > 0.5;
    if (actualUp === predUp) correct += 1;
  }

  const bins = [];
  for (let b = 0; b < 5; b++) {
    const lo = b / 5;
    const hi = (b + 1) / 5;
    const idx = [];
    for (let i = 0; i < n; i++) {
      const p = series.predicted_up_prob[i];
      if (p >= lo && (b === 4 ? p <= hi : p < hi)) idx.push(i);
    }
    if (!idx.length) continue;
    bins.push({
      avg_predicted: mean(idx.map((i) => series.predicted_up_prob[i])),
      actual_up_rate: mean(idx.map((i) => (series.target_return[i] > 0 ? 1 : 0))),
    });
  }

  return {
    strategy_total_return: sc - 1,
    buy_hold_total_return: bc - 1,
    strategy_sharpe: sharpe(stratR),
    buy_hold_sharpe: sharpe(bhR),
    strategy_max_drawdown: maxDrawdown(stratCum),
    buy_hold_max_drawdown: maxDrawdown(bhCum),
    directional_accuracy: correct / n,
    naive_baseline_accuracy: upDays / n,
    strategy_cumulative: stratCum,
    buy_hold_cumulative: bhCum,
    calibration_bins: bins,
  };
}

function verdictSingle(summary) {
  const acc = summary.directional_accuracy;
  const naive = summary.naive_baseline_accuracy;
  const sig = summary.significance_vs_naive;
  const tkr = lastPayload.ticker;
  if (acc <= naive) {
    return {
      tag: "no edge found",
      cls: "tag-neutral",
      text: `Accuracy of ${(acc * 100).toFixed(1)}% does not beat the naive “always predict up” baseline of ${(naive * 100).toFixed(1)}% on this window. ${tkr} rose on ${(naive * 100).toFixed(1)}% of test days, so the model has to clear that bar, not 50%. This is the common, honest outcome for short-term price direction.`,
    };
  }
  if (!sig["significant_at_0.05"]) {
    return {
      tag: "within noise",
      cls: "tag-neutral",
      text: `Accuracy of ${(acc * 100).toFixed(1)}% edges past the naive baseline of ${(naive * 100).toFixed(1)}%, but p = ${sig.p_value.toFixed(3)} over ${summary.test_days} days is well within randomness. Run walk-forward before reading anything into it.`,
    };
  }
  if (acc - naive > 0.06) {
    return {
      tag: "suspiciously strong",
      cls: "tag-caution",
      text: `Accuracy of ${(acc * 100).toFixed(1)}% against a ${(naive * 100).toFixed(1)}% baseline (p = ${sig.p_value.toFixed(3)}) is strong enough that the first suspect is data leakage, not skill. Confirm with walk-forward.`,
    };
  }
  return {
    tag: "statistically significant",
    cls: "tag-good",
    text: `Accuracy of ${(acc * 100).toFixed(1)}% beats the naive baseline of ${(naive * 100).toFixed(1)}% with p = ${sig.p_value.toFixed(3)}. One window is still one data point — try walk-forward and other tickers before trusting it.`,
  };
}

function metricCard(label, value, note, extraClass = "") {
  return `<div class="metric-card ${extraClass}">
    <div class="metric-label">${label}</div>
    <div class="metric-value">${value}</div>
    <div class="metric-note">${note}</div>
  </div>`;
}

function renderSingle(payload, live) {
  const s = payload.summary;
  const merged = { ...s, ...live };
  const edge = merged.directional_accuracy - merged.naive_baseline_accuracy;
  const v = verdictSingle({ ...s, directional_accuracy: merged.directional_accuracy, naive_baseline_accuracy: merged.naive_baseline_accuracy });
  const maxImp = Math.max(...payload.importance.map((r) => r.importance), 1e-9);
  const featHtml = payload.importance.map((row) => {
    const pct = (row.importance / maxImp) * 100;
    return `<div class="feat-row">
      <div class="feat-name">${row.label}</div>
      <div class="feat-bar-track"><div class="feat-bar-fill" style="width:${pct}%"></div></div>
      <div class="feat-val">${row.importance.toFixed(3)}</div>
    </div>`;
  }).join("");

  const sig = s.significance_vs_naive;
  const signal = payload.latest_signal;
  return `
    <div class="sec-head">// ${payload.ticker}, test period: ${s.test_days} trading days (${payload.series.test_start} to ${payload.series.test_end})</div>
    <div class="metric-row">
      ${metricCard("Directional Accuracy", fmtPct(merged.directional_accuracy), `naive baseline: ${fmtPct(merged.naive_baseline_accuracy)}`, "accent")}
      ${metricCard("Edge vs. Baseline", `<span class="${clsSign(edge)}">${(edge * 100).toFixed(1) >= 0 ? "+" : ""}${(edge * 100).toFixed(1)}pp</span>`, `p = ${sig.p_value.toFixed(3)}`)}
      ${metricCard("ROC-AUC", s.roc_auc.toFixed(3), "0.50 = no signal")}
      ${metricCard("Strategy Return", `<span class="${clsSign(merged.strategy_total_return)}">${fmtPct(merged.strategy_total_return)}</span>`, `buy & hold: ${fmtPct(merged.buy_hold_total_return)}`)}
      ${metricCard("Sharpe (Strategy)", merged.strategy_sharpe.toFixed(2), `buy/hold: ${merged.buy_hold_sharpe.toFixed(2)}`)}
      ${metricCard("Max Drawdown", `<span class="neg">${fmtPct(merged.strategy_max_drawdown)}</span>`, `buy/hold: ${fmtPct(merged.buy_hold_max_drawdown)}`)}
    </div>
    <div class="verdict"><span class="tag ${v.cls}">${v.tag}</span><br/>${v.text}</div>
    <div class="sec-head">// last test-day stance</div>
    <div class="verdict">On ${signal.date} the model assigned ${(signal.up_probability * 100).toFixed(1)}% probability of an up day.
      At the current threshold it would sit in <b>${signal.up_probability > Number($("threshold").value) ? "long" : "cash"}</b>.
      Directional accuracy still uses a 0.50 cutoff; the slider only changes when the simulated strategy holds the stock.</div>
    <div class="sec-head">// cumulative growth of $1</div>
    <div class="chart-wrap"><canvas id="equity-chart"></canvas></div>
    <div class="sec-head">// are the confidence scores trustworthy</div>
    <div class="chart-wrap short"><canvas id="cal-chart"></canvas></div>
    <div class="verdict">Brier score ${s.brier_score.toFixed(4)} (lower is better). If the amber line tracks the dotted diagonal, a 60% call really went up about 60% of the time — which is what makes the threshold slider meaningful.</div>
    <div class="sec-head">// what the model is actually looking at</div>
    ${featHtml}
    <div class="foot">trained on ${payload.train_days} days · tested on ${payload.test_days} days (out-of-sample) · chronological split · ${payload.use_market ? "SPY market features included" : "ticker-only features"}</div>
  `;
}

function paintEquity(series, live) {
  const ctx = document.getElementById("equity-chart");
  if (!ctx) return;
  equityChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: series.dates,
      datasets: [
        {
          label: "Model strategy",
          data: live.strategy_cumulative,
          borderColor: chartTheme.amber,
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.15,
        },
        {
          label: "Buy & hold",
          data: live.buy_hold_cumulative,
          borderColor: chartTheme.muted,
          borderDash: [4, 4],
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.15,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { display: true } },
      scales: {
        x: { ticks: { maxTicksLimit: 8 }, grid: { color: chartTheme.grid } },
        y: { grid: { color: chartTheme.grid } },
      },
    },
  });
}

function paintCal(bins) {
  const ctx = document.getElementById("cal-chart");
  if (!ctx) return;
  calChart = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Perfect calibration",
          data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
          showLine: true,
          borderColor: chartTheme.muted,
          borderDash: [4, 4],
          pointRadius: 0,
        },
        {
          label: "This model",
          data: bins.map((b) => ({ x: b.avg_predicted, y: b.actual_up_rate })),
          showLine: true,
          borderColor: chartTheme.amber,
          backgroundColor: chartTheme.amber,
          pointRadius: 5,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { min: 0, max: 1, title: { display: true, text: "predicted probability of up" }, grid: { color: chartTheme.grid } },
        y: { min: 0, max: 1, title: { display: true, text: "actual fraction that went up" }, grid: { color: chartTheme.grid } },
      },
    },
  });
}

function renderWalkForward(payload) {
  const wf = payload.summary;
  const beat = wf.folds_beating_naive_baseline;
  const good = beat > wf.n_folds / 2;
  const v = good
    ? { tag: "consistent across folds", cls: "tag-good", text: `The model beat the naive baseline in ${beat} of ${wf.n_folds} independent windows, with accuracy from ${fmtPct(wf.min_accuracy, 0)} to ${fmtPct(wf.max_accuracy, 0)}. Consistency is stronger evidence than any single backtest.` }
    : { tag: "no stable edge", cls: "tag-neutral", text: `Across ${wf.n_folds} windows the model beat the naive baseline in only ${beat}. Accuracy swung ${fmtPct(wf.min_accuracy, 0)} to ${fmtPct(wf.max_accuracy, 0)}. That spread is the point of this view: a single backtest from this range can be cherry-picked.` };

  const table = payload.folds.map((f) => `<tr>
    <td>${f.fold}</td><td>${f.test_start} → ${f.test_end}</td>
    <td>${fmtPct(f.accuracy)}</td><td>${fmtPct(f.naive_baseline)}</td>
    <td>${f.roc_auc.toFixed(3)}</td>
    <td>${fmtPct(f.strategy_return)}</td><td>${fmtPct(f.buy_hold_return)}</td>
  </tr>`).join("");

  return `
    <div class="sec-head">// ${payload.ticker}, walk-forward across ${wf.n_folds} chronological folds</div>
    <div class="metric-row">
      ${metricCard("Mean Accuracy", fmtPct(wf.mean_accuracy), `± ${(wf.std_accuracy * 100).toFixed(1)}pp across folds`, "accent")}
      ${metricCard("Range", `${(wf.min_accuracy * 100).toFixed(0)}–${(wf.max_accuracy * 100).toFixed(0)}%`, "worst to best fold")}
      ${metricCard("Naive Baseline", fmtPct(wf.mean_naive_baseline), "mean across folds")}
      ${metricCard("Folds Beating Baseline", `<span class="${good ? "pos" : "neg"}">${beat}/${wf.n_folds}</span>`, "coin flip would give ~50%")}
      ${metricCard("Mean Strategy Return", `<span class="${clsSign(wf.mean_strategy_return)}">${fmtPct(wf.mean_strategy_return)}</span>`, `buy & hold: ${fmtPct(wf.mean_buy_hold_return)}`)}
    </div>
    <div class="verdict"><span class="tag ${v.cls}">${v.tag}</span><br/>${v.text}</div>
    <div class="sec-head">// accuracy by fold vs. naive baseline</div>
    <div class="chart-wrap"><canvas id="wf-chart"></canvas></div>
    <div class="sec-head">// fold detail</div>
    <table><thead><tr><th>Fold</th><th>Test window</th><th>Accuracy</th><th>Naive</th><th>AUC</th><th>Strategy</th><th>Buy & hold</th></tr></thead><tbody>${table}</tbody></table>
  `;
}

function paintWalkForward(folds) {
  const ctx = document.getElementById("wf-chart");
  if (!ctx) return;
  wfChart = new Chart(ctx, {
    data: {
      labels: folds.map((f) => `Fold ${f.fold}`),
      datasets: [
        { type: "bar", label: "Model accuracy", data: folds.map((f) => f.accuracy), backgroundColor: chartTheme.amber },
        { type: "line", label: "Naive baseline", data: folds.map((f) => f.naive_baseline), borderColor: chartTheme.red, pointBackgroundColor: chartTheme.red, tension: 0 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { min: 0, max: 1, ticks: { callback: (v) => `${(v * 100).toFixed(0)}%` }, grid: { color: chartTheme.grid } },
        x: { grid: { display: false } },
      },
    },
  });
}

function renderCompare(payload) {
  const rows = payload.rows;
  const beat = rows.filter((r) => r.accuracy > r.naive_baseline).length;
  const extraHead = payload.walk_forward ? "<th>Std</th><th>Beats</th>" : "<th>AUC</th><th>p</th><th>Sharpe</th>";
  const body = rows.map((r) => {
    const extra = payload.walk_forward
      ? `<td>${r.std_accuracy != null ? fmtPct(r.std_accuracy) : "—"}</td><td>${r.beats_baseline}</td>`
      : `<td>${r.roc_auc.toFixed(3)}</td><td>${r.p_value.toFixed(3)}</td><td>${r.sharpe.toFixed(2)}</td>`;
    return `<tr>
      <td>${r.ticker}</td>
      <td>${fmtPct(r.accuracy)}</td>
      <td>${fmtPct(r.naive_baseline)}</td>
      ${extra}
      <td>${fmtPct(r.strategy_return)}</td>
      <td>${fmtPct(r.buy_hold_return)}</td>
    </tr>`;
  }).join("");

  return `
    <div class="sec-head">// comparison across ${rows.length} tickers${payload.walk_forward ? " (walk-forward)" : ""}</div>
    <div class="verdict"><span class="tag ${beat > rows.length / 2 ? "tag-good" : "tag-neutral"}">${beat}/${rows.length} beat the naive baseline</span><br/>
      An approach with real signal should work on more than one stock. If it beats the baseline on some names and not others with no pattern, that is what noise looks like.</div>
    <div class="chart-wrap"><canvas id="cmp-chart"></canvas></div>
    <table><thead><tr><th>Ticker</th><th>Accuracy</th><th>Naive</th>${extraHead}<th>Strategy</th><th>Buy & hold</th></tr></thead><tbody>${body}</tbody></table>
  `;
}

function paintCompare(rows) {
  const ctx = document.getElementById("cmp-chart");
  if (!ctx) return;
  cmpChart = new Chart(ctx, {
    data: {
      labels: rows.map((r) => r.ticker),
      datasets: [
        { type: "bar", label: "Model accuracy", data: rows.map((r) => r.accuracy), backgroundColor: chartTheme.amber },
        { type: "scatter", label: "Naive baseline", data: rows.map((r) => r.naive_baseline), backgroundColor: chartTheme.red, pointRadius: 6 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { min: 0, max: 1, ticks: { callback: (v) => `${(v * 100).toFixed(0)}%` }, grid: { color: chartTheme.grid } },
        x: { grid: { display: false } },
      },
    },
  });
}

function drawPayload(payload) {
  destroyCharts();
  const root = $("results");
  if (payload.mode === "single") {
    const live = recomputeBacktest(payload.series, Number($("threshold").value));
    root.innerHTML = renderSingle(payload, live);
    paintEquity(payload.series, live);
    paintCal(live.calibration_bins.length ? live.calibration_bins : payload.summary.calibration_bins);
  } else if (payload.mode === "walk_forward") {
    root.innerHTML = renderWalkForward(payload);
    paintWalkForward(payload.folds);
  } else {
    root.innerHTML = renderCompare(payload);
    paintCompare(payload.rows);
  }
}

async function runAnalysis(evt) {
  if (evt) evt.preventDefault();
  const btn = $("run-btn");
  const status = $("status");
  btn.disabled = true;
  status.hidden = false;
  status.textContent = "fetching prices, training, backtesting…";
  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ticker: $("ticker").value,
        years: Number($("years").value),
        threshold: Number($("threshold").value),
        use_market: $("use-market").checked,
        walk_forward: $("walk-forward").checked,
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      const detail = data.detail;
      const msg = Array.isArray(detail)
        ? detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
        : (detail || "Request failed");
      throw new Error(msg);
    }
    lastPayload = data;
    status.hidden = true;
    drawPayload(data);
  } catch (err) {
    $("results").innerHTML = `<div class="error">${err.message}</div>`;
    status.hidden = true;
  } finally {
    btn.disabled = false;
  }
}

function bind() {
  chartDefaults();
  $("years").addEventListener("input", () => {
    $("years-label").textContent = `${$("years").value}y`;
  });
  $("threshold").addEventListener("input", () => {
    $("thresh-label").textContent = Number($("threshold").value).toFixed(2);
    if (lastPayload && lastPayload.mode === "single") {
      drawPayload(lastPayload);
    }
  });
  $("controls").addEventListener("submit", runAnalysis);
  runAnalysis();
}

bind();
