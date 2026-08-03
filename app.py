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
from sklearn.metrics import mean_absolute_error
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

# --- Current prices across every grade -------------------------------------
st.subheader("Current Uganda coffee prices")

fob = snapshot[snapshot["level"] == "fob"]
farmgate = snapshot[snapshot["level"] == "farmgate"]

st.markdown("**Export grades** (FOB Kampala, US cents/kg)")
fob_cols = st.columns(3)
for i, row in enumerate(fob.itertuples()):
    fob_cols[i % 3].metric(row.grade, f"{row.price:,.2f} ¢/kg")

st.markdown("**Farmgate** (what farmers receive, UGX/kg)")
fg_cols = st.columns(3)
for i, row in enumerate(farmgate.itertuples()):
    fg_cols[i % 3].metric(row.grade, f"{row.price:,.0f} UGX/kg")

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

# --- History ---------------------------------------------------------------
st.subheader("Price history by grade")
grade_choice = st.multiselect(
    "Grades to plot (US cents/kg)",
    options=TARGET_COLS,
    default=["bugisu_aa", "screen_18"],
    format_func=lambda c: GRADE_LABELS[c],
)
if grade_choice:
    plot_df = df.reset_index()[["date"] + grade_choice].rename(columns=GRADE_LABELS)
    fig = px.line(plot_df, x="date", y=[GRADE_LABELS[c] for c in grade_choice])
    fig.update_layout(yaxis_title="US cents/kg", legend_title="Grade")
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

st.markdown("**Correlation of each driver with each grade**")
corr = pd.DataFrame({t: df[FEATURE_COLS].corrwith(df[t]) for t in TARGET_COLS})
st.dataframe(corr.round(3), use_container_width=True)

# --- Model -----------------------------------------------------------------
st.subheader("Baseline model")
target = st.selectbox(
    "Grade to predict", options=TARGET_COLS, format_func=lambda c: GRADE_LABELS[c]
)

model_df = df.dropna(subset=FEATURE_COLS + [target])
X, y = model_df[FEATURE_COLS], model_df[target]

if len(model_df) >= 12:
    # Walk-forward evaluation — a random split would let the model peek at the
    # future, which badly overstates accuracy on a trending series.
    actuals, predicted = [], []
    for train_idx, test_idx in TimeSeriesSplit(n_splits=4).split(X):
        fold = LinearRegression().fit(X.iloc[train_idx], y.iloc[train_idx])
        predicted.extend(fold.predict(X.iloc[test_idx]))
        actuals.extend(y.iloc[test_idx])
    mae = mean_absolute_error(actuals, predicted)

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (walk-forward CV)", f"{mae:,.1f} ¢/kg")
    c2.metric("As % of mean price", f"{mae / y.mean() * 100:.1f}%")
    c3.metric("Observations", len(model_df))

    model = LinearRegression().fit(X, y)
    fitted = pd.DataFrame(
        {"actual": y, "fitted": model.predict(X)}, index=model_df.index
    ).reset_index()
    st.plotly_chart(
        px.line(fitted, x="date", y=["actual", "fitted"], title=f"{GRADE_LABELS[target]}: actual vs fitted"),
        use_container_width=True,
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
        "collinear drivers, individual coefficients are not reliable causal estimates."
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
