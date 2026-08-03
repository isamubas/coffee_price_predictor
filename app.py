"""Streamlit dashboard for the Uganda Coffee Price Predictor.

Shows current Uganda coffee prices across every tracked grade, the drivers
that move them (world prices, USD/UGX, oil, Fed rate), and a baseline
regression that predicts a chosen grade from those drivers.

Runs locally (`streamlit run app.py`) or as a Hugging Face Space.
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

DATA_PROCESSED = Path(__file__).resolve().parent / "data" / "processed"

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

st.set_page_config(page_title="Uganda Coffee Price Predictor", page_icon="☕", layout="wide")

st.title("☕ Uganda Coffee Price Predictor")
st.caption(
    "Predicting Ugandan coffee prices by grade from the forces that move them — "
    "world Arabica/Robusta prices, the USD/UGX rate, oil, and US Fed policy."
)


@st.cache_data
def load_merged() -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / "merged_monthly.csv", index_col=0, parse_dates=True)
    df.index.name = "date"
    return df


@st.cache_data
def load_snapshot() -> pd.DataFrame:
    return pd.read_csv(DATA_PROCESSED / "uganda_grades_snapshot.csv")


df = load_merged()
snapshot = load_snapshot()

# --- Currency -------------------------------------------------------------
# Upstream quotes export grades in US cents/kg and farmgate in UGX/kg. We
# normalise everything to one unit so grades are actually comparable.
CURRENCIES = {
    "UGX/kg": {"decimals": 0, "suffix": "UGX/kg"},
    "USD/kg": {"decimals": 2, "suffix": "$/kg"},
    "US cents/kg": {"decimals": 2, "suffix": "¢/kg"},
}

currency = st.sidebar.radio("Currency", list(CURRENCIES), index=0)
st.sidebar.caption(
    "Uganda grades are quoted upstream in US cents/kg (export) and UGX/kg "
    "(farmgate). Historical values convert at each month's own USD/UGX rate."
)

fx_now = float(snapshot["usd_ugx_rate"].iloc[0])


def to_currency(values, source_unit: str, rate):
    """Convert a price series/scalar from its source unit into `currency`."""
    usd = values / 100 if source_unit == "USc/kg" else values / rate
    if currency == "USD/kg":
        return usd
    if currency == "US cents/kg":
        return usd * 100
    return usd * rate


def fmt(value) -> str:
    spec = CURRENCIES[currency]
    return f"{value:,.{spec['decimals']}f} {spec['suffix']}"


# --- Current prices across every grade -------------------------------------
st.subheader("Current Uganda coffee prices")
st.caption(f"Shown in {currency} · converted at USD/UGX {fx_now:,.2f}")

fob = snapshot[snapshot["level"] == "fob"]
farmgate = snapshot[snapshot["level"] == "farmgate"]

st.markdown("**Export grades** (FOB Kampala)")
fob_cols = st.columns(3)
for i, row in enumerate(fob.itertuples()):
    fob_cols[i % 3].metric(row.grade, fmt(to_currency(row.price, "USc/kg", fx_now)))

st.markdown("**Farmgate** (what farmers actually receive)")
fg_cols = st.columns(3)
for i, row in enumerate(farmgate.itertuples()):
    fg_cols[i % 3].metric(row.grade, fmt(to_currency(row.price, "UGX/kg", fx_now)))

st.caption(f"Snapshot from ugandacoffeeprices.com (UCDA), updated {snapshot['updated_utc'].iloc[0]}")

with st.expander("⚠️ Data quality — read this before trusting the model"):
    st.markdown(
        """
- The **current prices above are live**, pulled from the site's JSON feed.
- The **30-month history** driving the charts and model below is the *static
  fallback series embedded in the source page*, not a live UCDA feed. Its
  recent values track the live snapshot closely, so it is plausible — but it
  is approximate, not official UCDA records.
- Within each family the grades are **0.99+ correlated** (the three Bugisu
  grades are one curve scaled, likewise the three Screen grades). So this is
  effectively **2 independent series, not 6**.
- With only 30 strongly-trending observations, correlations here are inflated
  by shared trend. Treat them as suggestive, not causal.

**Fix:** log the live JSON daily to accumulate genuine per-grade history, and/or
parse UCDA's published PDF reports.
        """
    )

# Grade history is stored in USc/kg; convert at each month's own FX rate so
# UGX values reflect what the price was actually worth at the time.
grades_converted = df[TARGET_COLS].apply(
    lambda col: to_currency(col, "USc/kg", df["usd_ugx_rate"])
)

# --- History ---------------------------------------------------------------
st.subheader("Price history by grade")
grade_choice = st.multiselect(
    f"Grades to plot ({currency})",
    options=TARGET_COLS,
    default=["bugisu_aa", "screen_18"],
    format_func=lambda c: GRADE_LABELS[c],
)
if grade_choice:
    plot_df = grades_converted[grade_choice].rename(columns=GRADE_LABELS).reset_index()
    fig = px.line(plot_df, x="date", y=[GRADE_LABELS[c] for c in grade_choice])
    fig.update_layout(yaxis_title=currency, legend_title="Grade")
    st.plotly_chart(fig, use_container_width=True)

# --- Drivers ---------------------------------------------------------------
st.subheader("Drivers")
driver_choice = st.multiselect(
    "Driver series to plot", options=FEATURE_COLS, default=["arabica_usd_kg", "usd_ugx_rate"]
)
if driver_choice:
    st.plotly_chart(
        px.line(df.reset_index(), x="date", y=driver_choice), use_container_width=True
    )

st.markdown(f"**Correlation of each driver with each grade** (grades in {currency})")
corr = pd.DataFrame(
    {t: df[FEATURE_COLS].corrwith(grades_converted[t]) for t in TARGET_COLS}
)
st.dataframe(corr.round(3), use_container_width=True)
if currency == "UGX/kg":
    st.caption(
        "In UGX the grade series is itself a function of USD/UGX, so its "
        "correlation with `usd_ugx_rate` is partly mechanical. Switch to USD/kg "
        "to see the underlying relationship."
    )

# --- Model -----------------------------------------------------------------
st.subheader("Baseline model")
target = st.selectbox(
    "Grade to predict", options=TARGET_COLS, format_func=lambda c: GRADE_LABELS[c]
)

model_df = df.dropna(subset=FEATURE_COLS + [target])
X = model_df[FEATURE_COLS]
y = grades_converted.loc[model_df.index, target]

if len(model_df) >= 12:
    # Walk-forward evaluation — a random split would let the model peek at the
    # future, which badly overstates accuracy on a trending series. We also
    # score a random-walk baseline (carry last observed price forward), because
    # on price series that baseline is the bar any model has to clear.
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
    r2_in = r2_score(y, LinearRegression().fit(X, y).predict(X))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² in-sample", f"{r2_in:.3f}")
    c2.metric("R² walk-forward", f"{r2_oos:.3f}", delta="worse than mean" if r2_oos < 0 else None)
    c3.metric("MAE walk-forward", fmt(mae))
    c4.metric("Observations", len(model_df))

    if r2_oos < naive_r2:
        st.error(
            f"**This regression does not work yet.** In-sample R² of {r2_in:.2f} looks "
            f"strong, but walk-forward R² is {r2_oos:.2f} — a negative value means it "
            "predicts held-out months *worse than simply guessing the training mean*.\n\n"
            f"A random-walk baseline (carry last month's price forward) gets "
            f"R² {naive_r2:.2f} and MAE {fmt(naive_mae)}, versus the regression's "
            f"MAE {fmt(mae)}. **The naive baseline beats the model.**\n\n"
            "This is the 30-month trending dataset overfitting, exactly as the data-quality "
            "note warns. Real per-grade history is needed before the model means anything."
        )
    else:
        st.success(
            f"Regression (R² {r2_oos:.2f}, MAE {fmt(mae)}) beats the random-walk "
            f"baseline (R² {naive_r2:.2f}, MAE {fmt(naive_mae)})."
        )

    model = LinearRegression().fit(X, y)
    fitted = pd.DataFrame(
        {"actual": y, "fitted": model.predict(X)}, index=model_df.index
    ).reset_index()
    chart = px.line(
        fitted,
        x="date",
        y=["actual", "fitted"],
        title=f"{GRADE_LABELS[target]}: actual vs fitted",
    )
    chart.update_layout(yaxis_title=currency)
    st.plotly_chart(chart, use_container_width=True)
    st.caption(
        "Note this chart is the *in-sample* fit — the model saw every point it "
        "is drawing. It looks good for the same reason in-sample R² looks good, "
        "and is not evidence the model forecasts."
    )

    st.markdown("**Feature coefficients** (full-sample fit)")
    st.dataframe(
        pd.DataFrame({"feature": FEATURE_COLS, "coefficient": model.coef_})
        .sort_values("coefficient", key=abs, ascending=False),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Walk-forward CV respects time order. With 30 observations and "
        "collinear drivers, individual coefficients are not reliable causal "
        "estimates. Coefficients are in "
        f"{currency} per unit of each driver, so they rescale with the "
        "currency you pick."
    )
else:
    st.warning("Not enough overlapping data to train a model yet.")

# --- Raw data --------------------------------------------------------------
with st.expander("Raw merged dataset"):
    st.dataframe(df, use_container_width=True)

st.caption(
    "Data: ugandacoffeeprices.com/UCDA (Uganda grades), World Bank Pink Sheet "
    "(world coffee, oil), FRED (Fed funds rate), Yahoo Finance (USD/UGX). "
    "See README for details."
)
