"""Gradio dashboard for the Uganda Coffee Price Predictor.

Shows current Uganda coffee prices across every tracked grade, the drivers
that move them (world prices, USD/UGX, oil, Fed rate), and a baseline
regression scored honestly against a random-walk baseline.

Gradio rather than Streamlit because Hugging Face Spaces offers only gradio,
docker, and static SDKs, and Docker is gated behind a paid plan.

Run locally with `python app.py`, or deploy as a Gradio Space.
"""
import sys
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parent
DATA_PROCESSED = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "src"))

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
LABEL_TO_KEY = {v: k for k, v in GRADE_LABELS.items()}

CURRENCIES = {
    "UGX/kg": {"decimals": 0, "suffix": "UGX/kg"},
    "USD/kg": {"decimals": 2, "suffix": "$/kg"},
    "US cents/kg": {"decimals": 2, "suffix": "¢/kg"},
}

FARMGATE_GRADES = {"kiboko", "faq", "arabica_parchment"}


# --- Data loading ----------------------------------------------------------
def load_merged() -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / "merged_monthly.csv", index_col=0, parse_dates=True)
    df.index.name = "date"
    return df


def load_snapshot() -> tuple[pd.DataFrame, bool]:
    """Current prices — fetched live, falling back to the committed CSV.

    Live fetching is what keeps a deployed Space current without a redeploy.
    Any network failure degrades to the last committed snapshot rather than
    taking the page down.
    """
    try:
        from fetch_uganda_prices import fetch_market_snapshot, snapshot_to_frame

        return snapshot_to_frame(fetch_market_snapshot(attempts=2, timeout=10)), True
    except Exception:
        return pd.read_csv(DATA_PROCESSED / "uganda_grades_snapshot.csv"), False


DF = load_merged()
SNAPSHOT, IS_LIVE = load_snapshot()
FX_NOW = float(SNAPSHOT["usd_ugx_rate"].iloc[0])


# --- Currency conversion ---------------------------------------------------
def to_currency(values, source_unit: str, rate, currency: str):
    """Convert a price series/scalar from its source unit into `currency`."""
    usd = values / 100 if source_unit == "USc/kg" else values / rate
    if currency == "USD/kg":
        return usd
    if currency == "US cents/kg":
        return usd * 100
    return usd * rate


def fmt(value, currency: str) -> str:
    spec = CURRENCIES[currency]
    return f"{value:,.{spec['decimals']}f} {spec['suffix']}"


def converted_grades(currency: str) -> pd.DataFrame:
    """Grade history in the chosen currency, using each month's own FX rate."""
    return DF[TARGET_COLS].apply(
        lambda col: to_currency(col, "USc/kg", DF["usd_ugx_rate"], currency)
    )


# --- Views -----------------------------------------------------------------
def current_prices(currency: str) -> tuple[str, pd.DataFrame]:
    rows = []
    for row in SNAPSHOT.itertuples():
        unit = "UGX/kg" if row.grade_key in FARMGATE_GRADES else "USc/kg"
        rows.append(
            {
                "Grade": row.grade,
                "Level": "farmgate" if row.grade_key in FARMGATE_GRADES else "export (FOB)",
                f"Price ({currency})": fmt(
                    to_currency(row.price, unit, FX_NOW, currency), currency
                ),
            }
        )
    table = pd.DataFrame(rows).sort_values("Level", ascending=False)

    freshness = (
        "🟢 fetched live just now"
        if IS_LIVE
        else "🟠 live fetch unavailable — showing last committed snapshot"
    )
    note = (
        f"**Shown in {currency}** · converted at USD/UGX {FX_NOW:,.2f}  \n"
        f"Source: ugandacoffeeprices.com (UCDA) · upstream updated "
        f"{SNAPSHOT['updated_utc'].iloc[0]} · {freshness}"
    )
    return note, table


def history_plot(grade_labels: list[str], currency: str):
    if not grade_labels:
        return None
    keys = [LABEL_TO_KEY[label] for label in grade_labels]
    data = converted_grades(currency)[keys].rename(columns=GRADE_LABELS).reset_index()
    fig = px.line(data, x="date", y=[GRADE_LABELS[k] for k in keys])
    fig.update_layout(yaxis_title=currency, legend_title="Grade", template="plotly_white")
    return fig


def drivers_plot(drivers: list[str]):
    if not drivers:
        return None
    fig = px.line(DF.reset_index(), x="date", y=drivers)
    fig.update_layout(template="plotly_white", legend_title="Driver")
    return fig


def correlation_table(currency: str) -> tuple[pd.DataFrame, str]:
    grades = converted_grades(currency)
    corr = pd.DataFrame({t: DF[FEATURE_COLS].corrwith(grades[t]) for t in TARGET_COLS})
    corr = corr.rename(columns=GRADE_LABELS).round(3).reset_index(names="Driver")

    caveat = ""
    if currency == "UGX/kg":
        caveat = (
            "⚠️ In UGX the grade series is itself a function of USD/UGX, so its "
            "correlation with `usd_ugx_rate` is partly mechanical. Switch to USD/kg "
            "to see the underlying relationship."
        )
    return corr, caveat


def run_model(grade_label: str, currency: str):
    """Fit the baseline regression and score it against a random walk."""
    target = LABEL_TO_KEY[grade_label]
    model_df = DF.dropna(subset=FEATURE_COLS + [target])
    X = model_df[FEATURE_COLS]
    y = converted_grades(currency).loc[model_df.index, target]

    if len(model_df) < 12:
        return "Not enough overlapping data to train a model yet.", None, None

    # Walk-forward evaluation — a random split would let the model peek at the
    # future, which badly overstates accuracy on a trending series. The
    # random-walk baseline (carry last price forward) is the bar to clear.
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

    verdict = (
        f"### ❌ This regression does not work yet\n\n"
        f"In-sample R² of **{r2_in:.2f}** looks strong, but walk-forward R² is "
        f"**{r2_oos:.2f}** — a negative value means it predicts held-out months "
        f"*worse than simply guessing the training mean*.\n\n"
        f"A random-walk baseline (carry last month's price forward) gets R² "
        f"**{naive_r2:.2f}** and MAE **{fmt(naive_mae, currency)}**, versus the "
        f"regression's MAE **{fmt(mae, currency)}**. **The naive baseline beats "
        f"the model.**\n\n"
        f"This is the 30-month trending dataset overfitting, exactly as the "
        f"data-quality note warns. Real per-grade history is needed first."
        if r2_oos < naive_r2
        else (
            f"### ✅ Regression beats the naive baseline\n\n"
            f"Walk-forward R² **{r2_oos:.2f}**, MAE **{fmt(mae, currency)}** versus "
            f"random-walk R² **{naive_r2:.2f}**, MAE **{fmt(naive_mae, currency)}**."
        )
    )

    metrics = pd.DataFrame(
        {
            "Metric": [
                "R² in-sample",
                "R² walk-forward",
                f"MAE walk-forward ({currency})",
                f"MAE random-walk baseline ({currency})",
                "Observations",
            ],
            "Value": [
                f"{r2_in:.3f}",
                f"{r2_oos:.3f}",
                fmt(mae, currency),
                fmt(naive_mae, currency),
                str(len(model_df)),
            ],
        }
    )

    fitted = pd.DataFrame({"actual": y, "fitted": model.predict(X)}, index=model_df.index)
    fig = px.line(
        fitted.reset_index(),
        x="date",
        y=["actual", "fitted"],
        title=f"{grade_label}: actual vs fitted (in-sample — not forecast evidence)",
    )
    fig.update_layout(yaxis_title=currency, template="plotly_white")

    return verdict, metrics, fig


DATA_QUALITY_NOTE = """
- The **current prices are live**, pulled from the site's JSON feed.
- The **30-month history** driving the charts and model is the *static fallback
  series embedded in the source page*, not a live UCDA feed. Its recent values
  track the live snapshot closely, so it is plausible — but it is approximate,
  not official UCDA records.
- Within each family the grades are **0.99+ correlated** (the three Bugisu
  grades are one curve scaled, likewise the three Screen grades). So this is
  effectively **2 independent series, not 6**.
- With only 30 strongly-trending observations, correlations are inflated by
  shared trend. Treat them as suggestive, not causal.

**The fix is running:** `src/log_daily_prices.py` records the live snapshot
daily via GitHub Actions, accumulating genuine per-grade history.
"""


# --- UI --------------------------------------------------------------------
with gr.Blocks(title="Uganda Coffee Price Predictor") as demo:
    gr.Markdown(
        "# ☕ Uganda Coffee Price Predictor\n"
        "Predicting Ugandan coffee prices by grade from the forces that move them — "
        "world Arabica/Robusta prices, the USD/UGX rate, Brent crude, and US Fed policy."
    )

    currency_in = gr.Radio(
        choices=list(CURRENCIES),
        value="UGX/kg",
        label="Currency",
        info="Upstream quotes mix units (export in US cents/kg, farmgate in UGX/kg). "
        "History converts at each month's own USD/UGX rate.",
    )

    with gr.Tab("Current prices"):
        price_note = gr.Markdown()
        price_table = gr.Dataframe(interactive=False, wrap=True)

    with gr.Tab("History"):
        grade_select = gr.CheckboxGroup(
            choices=list(GRADE_LABELS.values()),
            value=["Bugisu AA (Arabica)", "Screen 18 (Robusta)"],
            label="Grades to plot",
        )
        history_out = gr.Plot()

    with gr.Tab("Drivers"):
        driver_select = gr.CheckboxGroup(
            choices=FEATURE_COLS,
            value=["arabica_usd_kg", "usd_ugx_rate"],
            label="Driver series",
        )
        drivers_out = gr.Plot()
        gr.Markdown("### Correlation of each driver with each grade")
        corr_out = gr.Dataframe(interactive=False, wrap=True)
        corr_caveat = gr.Markdown()

    with gr.Tab("Model"):
        model_grade = gr.Dropdown(
            choices=list(GRADE_LABELS.values()),
            value="Bugisu AA (Arabica)",
            label="Grade to predict",
        )
        verdict_out = gr.Markdown()
        metrics_out = gr.Dataframe(interactive=False, wrap=True)
        fitted_out = gr.Plot()

    with gr.Accordion("⚠️ Data quality — read before trusting the model", open=False):
        gr.Markdown(DATA_QUALITY_NOTE)

    gr.Markdown(
        "Data: ugandacoffeeprices.com/UCDA (Uganda grades), World Bank Pink Sheet "
        "(world coffee, oil), FRED (Fed funds rate), Yahoo Finance (USD/UGX). "
        "[Source and methodology](https://github.com/isamubas/coffee_price_predictor)"
    )

    # --- Wiring ---
    def refresh_all(currency, grades, drivers, model_grade_label):
        note, table = current_prices(currency)
        corr, caveat = correlation_table(currency)
        verdict, metrics, fitted = run_model(model_grade_label, currency)
        return (
            note,
            table,
            history_plot(grades, currency),
            drivers_plot(drivers),
            corr,
            caveat,
            verdict,
            metrics,
            fitted,
        )

    all_outputs = [
        price_note,
        price_table,
        history_out,
        drivers_out,
        corr_out,
        corr_caveat,
        verdict_out,
        metrics_out,
        fitted_out,
    ]
    all_inputs = [currency_in, grade_select, driver_select, model_grade]

    currency_in.change(refresh_all, all_inputs, all_outputs)
    grade_select.change(history_plot, [grade_select, currency_in], history_out)
    driver_select.change(drivers_plot, driver_select, drivers_out)
    model_grade.change(
        run_model, [model_grade, currency_in], [verdict_out, metrics_out, fitted_out]
    )
    demo.load(refresh_all, all_inputs, all_outputs)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
