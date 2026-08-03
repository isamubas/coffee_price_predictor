"""Generate a self-contained static dashboard (index.html).

Hugging Face gates Gradio Spaces behind PRO or ZeroGPU eligibility, but Static
Spaces are free for everyone with no conditions. Python cannot run on a static
Space — so instead it runs *here*, ahead of time, and emits a page whose charts
stay interactive client-side via Plotly.

Everything the Gradio app computed is precomputed per currency and per grade,
then toggled in the browser. The GitHub Action regenerates this daily, so the
page stays current without any server.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
OUT_PATH = ROOT / "index.html"

TARGET_COLS = ["bugisu_aa", "bugisu_a", "bugisu_b", "screen_18", "screen_15", "screen_12"]
FEATURE_COLS = [
    "arabica_usd_kg",
    "robusta_usd_kg",
    "usd_ugx_rate",
    "brent_usd_bbl",
    "fed_funds_rate",
]
GRADE_LABELS = {
    "bugisu_aa": "Bugisu AA (Arabica)",
    "bugisu_a": "Bugisu A (Arabica)",
    "bugisu_b": "Bugisu B (Arabica)",
    "screen_18": "Screen 18 (Robusta)",
    "screen_15": "Screen 15 (Robusta)",
    "screen_12": "Screen 12 (Robusta)",
}
CURRENCIES = {
    "UGX/kg": {"decimals": 0, "suffix": "UGX/kg"},
    "USD/kg": {"decimals": 2, "suffix": "$/kg"},
    "US cents/kg": {"decimals": 2, "suffix": "¢/kg"},
}
FARMGATE_GRADES = {"kiboko", "faq", "arabica_parchment"}

PLOT_LAYOUT = {
    "template": "plotly_white",
    "margin": {"l": 55, "r": 20, "t": 40, "b": 45},
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#666"},
}


def to_currency(values, source_unit, rate, currency):
    usd = values / 100 if source_unit == "USc/kg" else values / rate
    if currency == "USD/kg":
        return usd
    if currency == "US cents/kg":
        return usd * 100
    return usd * rate


def fmt(value, currency):
    spec = CURRENCIES[currency]
    return f"{value:,.{spec['decimals']}f} {spec['suffix']}"


def load_data():
    df = pd.read_csv(DATA_PROCESSED / "merged_monthly.csv", index_col=0, parse_dates=True)
    df.index.name = "date"
    snapshot = pd.read_csv(DATA_PROCESSED / "uganda_grades_snapshot.csv")
    return df, snapshot


def converted_grades(df, currency):
    return df[TARGET_COLS].apply(
        lambda col: to_currency(col, "USc/kg", df["usd_ugx_rate"], currency)
    )


def build_price_rows(snapshot, currency, fx_now):
    rows = []
    for row in snapshot.itertuples():
        unit = "UGX/kg" if row.grade_key in FARMGATE_GRADES else "USc/kg"
        rows.append(
            {
                "grade": row.grade,
                "level": "farmgate" if row.grade_key in FARMGATE_GRADES else "export (FOB)",
                "price": fmt(to_currency(row.price, unit, fx_now, currency), currency),
            }
        )
    rows.sort(key=lambda r: (r["level"] != "export (FOB)", r["grade"]))
    return rows


def score_model(df, target, currency):
    """Walk-forward evaluation plus a random-walk baseline."""
    model_df = df.dropna(subset=FEATURE_COLS + [target])
    X = model_df[FEATURE_COLS]
    y = converted_grades(df, currency).loc[model_df.index, target]

    actuals, predicted, naive = [], [], []
    for train_idx, test_idx in TimeSeriesSplit(n_splits=4).split(X):
        fold = LinearRegression().fit(X.iloc[train_idx], y.iloc[train_idx])
        predicted.extend(fold.predict(X.iloc[test_idx]))
        carried = y.iloc[train_idx].iloc[-1]
        for i in test_idx:
            naive.append(carried)
            carried = y.iloc[i]
        actuals.extend(y.iloc[test_idx])

    mae = mean_absolute_error(actuals, predicted)
    r2_oos = r2_score(actuals, predicted)
    naive_mae = mean_absolute_error(actuals, naive)
    naive_r2 = r2_score(actuals, naive)

    model = LinearRegression().fit(X, y)
    r2_in = r2_score(y, model.predict(X))

    beats_naive = r2_oos >= naive_r2
    if beats_naive:
        verdict = (
            f"<h3 class='ok'>✅ Regression beats the naive baseline</h3>"
            f"<p>Walk-forward R² <b>{r2_oos:.2f}</b>, MAE <b>{fmt(mae, currency)}</b>, "
            f"versus random-walk R² <b>{naive_r2:.2f}</b>, MAE <b>{fmt(naive_mae, currency)}</b>.</p>"
        )
    else:
        verdict = (
            f"<h3 class='bad'>❌ This regression does not work yet</h3>"
            f"<p>In-sample R² of <b>{r2_in:.2f}</b> looks strong, but walk-forward R² is "
            f"<b>{r2_oos:.2f}</b> — a negative value means it predicts held-out months "
            f"<i>worse than simply guessing the training mean</i>.</p>"
            f"<p>A random-walk baseline (carry last month's price forward) gets R² "
            f"<b>{naive_r2:.2f}</b> and MAE <b>{fmt(naive_mae, currency)}</b>, versus the "
            f"regression's MAE <b>{fmt(mae, currency)}</b>. "
            f"<b>The naive baseline beats the model.</b></p>"
            f"<p>This is the 30-month trending dataset overfitting, exactly as the "
            f"data-quality note warns. Real per-grade history is needed first.</p>"
        )

    metrics = [
        ("R² in-sample", f"{r2_in:.3f}"),
        ("R² walk-forward", f"{r2_oos:.3f}"),
        (f"MAE walk-forward", fmt(mae, currency)),
        (f"MAE random-walk baseline", fmt(naive_mae, currency)),
        ("Observations", str(len(model_df))),
    ]

    fitted = pd.DataFrame({"actual": y, "fitted": model.predict(X)}, index=model_df.index)
    fig = px.line(fitted.reset_index(), x="date", y=["actual", "fitted"])
    fig.update_layout(yaxis_title=currency, legend_title="", **PLOT_LAYOUT)

    return {"verdict": verdict, "metrics": metrics, "figure": json.loads(fig.to_json())}


def build_payload(df, snapshot):
    fx_now = float(snapshot["usd_ugx_rate"].iloc[0])

    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "upstream_updated": str(snapshot["updated_utc"].iloc[0]),
        "fx_now": fx_now,
        "grade_labels": GRADE_LABELS,
        "currencies": list(CURRENCIES),
        "prices": {},
        "history": {},
        "correlation": {},
        "models": {},
    }

    for currency in CURRENCIES:
        payload["prices"][currency] = build_price_rows(snapshot, currency, fx_now)

        grades = converted_grades(df, currency)
        hist = px.line(
            grades.rename(columns=GRADE_LABELS).reset_index(),
            x="date",
            y=list(GRADE_LABELS.values()),
        )
        hist.update_layout(yaxis_title=currency, legend_title="Grade", **PLOT_LAYOUT)
        payload["history"][currency] = json.loads(hist.to_json())

        corr = pd.DataFrame({t: df[FEATURE_COLS].corrwith(grades[t]) for t in TARGET_COLS})
        payload["correlation"][currency] = {
            "columns": [GRADE_LABELS[c] for c in corr.columns],
            "rows": [[idx] + [f"{v:.3f}" for v in row] for idx, row in corr.iterrows()],
        }

        payload["models"][currency] = {
            t: score_model(df, t, currency) for t in TARGET_COLS
        }

    drivers = px.line(df.reset_index(), x="date", y=FEATURE_COLS)
    drivers.update_layout(legend_title="Driver", **PLOT_LAYOUT)
    payload["drivers"] = json.loads(drivers.to_json())

    return payload


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Uganda Coffee Price Predictor</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root {
    --bg: #ffffff; --fg: #1a1a1a; --muted: #666; --border: #e2e2e2;
    --card: #fafafa; --accent: #6b4423; --bad: #c0392b; --ok: #27865a;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #12141a; --fg: #e8e8e8; --muted: #9a9a9a; --border: #2a2d36;
      --card: #1a1d24; --accent: #d4a843; --bad: #ff6b5b; --ok: #4ade80;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 1.5rem 1rem 4rem; background: var(--bg); color: var(--fg);
    font: 16px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  .wrap { max-width: 1100px; margin: 0 auto; }
  h1 { font-size: 1.9rem; margin: 0 0 .3rem; }
  h2 { font-size: 1.25rem; margin: 2.5rem 0 .75rem; }
  h3 { margin: 0 0 .5rem; font-size: 1.05rem; }
  h3.bad { color: var(--bad); } h3.ok { color: var(--ok); }
  .sub { color: var(--muted); margin: 0 0 1.5rem; }
  .controls { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; margin-bottom: 1rem; }
  button.pill, select {
    font: inherit; padding: .4rem .9rem; border-radius: 999px;
    border: 1px solid var(--border); background: var(--card); color: var(--fg); cursor: pointer;
  }
  button.pill.active { background: var(--accent); color: #fff; border-color: var(--accent); }
  select { border-radius: 8px; }
  .grid { display: grid; gap: .6rem; grid-template-columns: repeat(auto-fill, minmax(215px, 1fr)); }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: .8rem .9rem; }
  .card .g { font-size: .78rem; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
  .card .p { font-size: 1.3rem; font-weight: 600; margin-top: .15rem; }
  .tablewrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 10px; }
  table { border-collapse: collapse; width: 100%; font-size: .9rem; min-width: 520px; }
  th, td { padding: .55rem .8rem; text-align: left; border-bottom: 1px solid var(--border); white-space: nowrap; }
  th { background: var(--card); font-weight: 600; }
  tr:last-child td { border-bottom: none; }
  .plot { width: 100%; height: 420px; }
  details { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: .8rem 1rem; margin: 2rem 0; }
  summary { cursor: pointer; font-weight: 600; }
  details ul { margin: .8rem 0 0; padding-left: 1.2rem; }
  footer { margin-top: 3rem; color: var(--muted); font-size: .85rem; border-top: 1px solid var(--border); padding-top: 1rem; }
  a { color: var(--accent); }
  .note { color: var(--muted); font-size: .85rem; margin-top: .5rem; }
</style>
</head>
<body>
<div class="wrap">
  <h1>&#9749; Uganda Coffee Price Predictor</h1>
  <p class="sub">Predicting Ugandan coffee prices by grade from the forces that move them &mdash;
  world Arabica/Robusta prices, the USD/UGX rate, Brent crude, and US Fed policy.</p>

  <div class="controls">
    <span style="color:var(--muted);font-size:.9rem">Currency:</span>
    <span id="currencyBtns"></span>
  </div>

  <h2>Current prices</h2>
  <p class="note" id="priceNote"></p>
  <div class="grid" id="priceGrid"></div>

  <h2>Price history by grade</h2>
  <div class="plot" id="historyPlot"></div>

  <h2>Drivers</h2>
  <div class="plot" id="driversPlot"></div>

  <h2>Correlation of each driver with each grade</h2>
  <div class="tablewrap"><table id="corrTable"></table></div>
  <p class="note" id="corrNote"></p>

  <h2>Baseline model</h2>
  <div class="controls">
    <label for="gradeSelect" style="color:var(--muted);font-size:.9rem">Grade to predict:</label>
    <select id="gradeSelect"></select>
  </div>
  <div id="verdict"></div>
  <div class="tablewrap"><table id="metricsTable"></table></div>
  <div class="plot" id="fittedPlot"></div>
  <p class="note">The chart above is the <b>in-sample</b> fit &mdash; the model saw every point it
  is drawing. It looks good for the same reason in-sample R&sup2; looks good, and is not evidence
  the model forecasts.</p>

  <details>
    <summary>&#9888;&#65039; Data quality &mdash; read before trusting the model</summary>
    <ul>
      <li>The <b>current prices</b> come from the site's live JSON feed, refreshed daily.</li>
      <li>The <b>30-month history</b> behind the charts and model is the <i>static fallback series
      embedded in the source page</i>, not a live UCDA feed. Its recent values track the live
      snapshot closely, so it is plausible &mdash; but approximate, not official UCDA records.</li>
      <li>Within each family the grades are <b>0.99+ correlated</b> (the three Bugisu grades are
      one curve scaled, likewise the three Screen grades). So this is effectively
      <b>2 independent series, not 6</b>.</li>
      <li>With only 30 strongly-trending observations, correlations are inflated by shared trend.
      Treat them as suggestive, not causal.</li>
      <li><b>The fix is running:</b> a daily GitHub Action records the live snapshot, accumulating
      genuine per-grade history.</li>
    </ul>
  </details>

  <footer>
    Data: ugandacoffeeprices.com/UCDA (Uganda grades), World Bank Pink Sheet (world coffee, oil),
    FRED (Fed funds rate), Yahoo Finance (USD/UGX).
    <a href="https://github.com/isamubas/coffee_price_predictor">Source and methodology</a>.
    <br>Page generated __GENERATED__ &middot; upstream prices updated __UPSTREAM__.
  </footer>
</div>

<script>
const DATA = __PAYLOAD__;
let currency = "UGX/kg";
let grade = "bugisu_aa";

function renderCurrencyButtons() {
  document.getElementById("currencyBtns").innerHTML = DATA.currencies
    .map(c => `<button class="pill${c === currency ? " active" : ""}" data-c="${c}">${c}</button>`)
    .join(" ");
  document.querySelectorAll("#currencyBtns .pill").forEach(b =>
    b.onclick = () => { currency = b.dataset.c; renderAll(); });
}

function renderPrices() {
  document.getElementById("priceNote").textContent =
    `Shown in ${currency} · converted at USD/UGX ${DATA.fx_now.toLocaleString(undefined,
      {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
  document.getElementById("priceGrid").innerHTML = DATA.prices[currency]
    .map(r => `<div class="card"><div class="g">${r.grade} · ${r.level}</div>
               <div class="p">${r.price}</div></div>`).join("");
}

function renderCorr() {
  const c = DATA.correlation[currency];
  document.getElementById("corrTable").innerHTML =
    `<thead><tr><th>Driver</th>${c.columns.map(h => `<th>${h}</th>`).join("")}</tr></thead>` +
    `<tbody>${c.rows.map(r => `<tr>${r.map((v, i) =>
      i === 0 ? `<th>${v}</th>` : `<td>${v}</td>`).join("")}</tr>`).join("")}</tbody>`;
  document.getElementById("corrNote").innerHTML = currency === "UGX/kg"
    ? "\\u26a0\\ufe0f In UGX the grade series is itself a function of USD/UGX, so its correlation " +
      "with <code>usd_ugx_rate</code> is partly mechanical. Switch to USD/kg to see the " +
      "underlying relationship."
    : "";
}

function renderGradeSelect() {
  document.getElementById("gradeSelect").innerHTML = Object.entries(DATA.grade_labels)
    .map(([k, v]) => `<option value="${k}"${k === grade ? " selected" : ""}>${v}</option>`).join("");
  document.getElementById("gradeSelect").onchange = e => { grade = e.target.value; renderModel(); };
}

function renderModel() {
  const m = DATA.models[currency][grade];
  document.getElementById("verdict").innerHTML = m.verdict;
  document.getElementById("metricsTable").innerHTML =
    `<thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>` +
    m.metrics.map(([k, v]) => `<tr><th>${k}</th><td>${v}</td></tr>`).join("") + `</tbody>`;
  Plotly.newPlot("fittedPlot", m.figure.data, m.figure.layout, {responsive: true});
}

function renderAll() {
  renderCurrencyButtons();
  renderPrices();
  const h = DATA.history[currency];
  Plotly.newPlot("historyPlot", h.data, h.layout, {responsive: true});
  Plotly.newPlot("driversPlot", DATA.drivers.data, DATA.drivers.layout, {responsive: true});
  renderCorr();
  renderModel();
}

renderGradeSelect();
renderAll();
</script>
</body>
</html>
"""


def main() -> None:
    df, snapshot = load_data()
    payload = build_payload(df, snapshot)

    html = (
        HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload))
        .replace("__GENERATED__", payload["generated_utc"])
        .replace("__UPSTREAM__", payload["upstream_updated"])
    )
    OUT_PATH.write_text(html, encoding="utf-8")

    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"Wrote {OUT_PATH.name} ({size_kb:,.0f} KB)")
    print(f"  currencies: {len(payload['currencies'])}")
    print(f"  grades modelled: {len(TARGET_COLS)} per currency")
    print(f"  generated: {payload['generated_utc']}")


if __name__ == "__main__":
    main()
