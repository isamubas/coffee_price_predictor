"""Build the Kaggle notebook from this repo's analysis.

The notebook is generated rather than hand-edited so its narrative numbers can
never drift from `src/diagnose_model.py`. Both read the same committed CSV, so
regenerating after a data refresh keeps the two in step.

Run: python notebooks/build_notebook.py
"""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parent / "walk-forward-audit.ipynb"

RAW = (
    "https://raw.githubusercontent.com/isamubas/coffee_price_predictor/"
    "main/data/processed/merged_monthly.csv"
)

REPO = "https://github.com/isamubas/coffee_price_predictor"

cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


# ---------------------------------------------------------------- opening ---
md(
    """
# Your R² Is Lying to You

### A walk-forward audit of a commodity price model that looked great and wasn't

This notebook builds a perfectly reasonable regression to predict Ugandan coffee
prices from the global drivers that should move them — world Arabica and Robusta
prices, the USD/UGX exchange rate, Brent crude, and the US Fed funds rate.

It reports an in-sample **R² of 0.88**.

It is also **worse than useless**. Under time-ordered evaluation it scores
**R² = −0.66**, while the laziest possible baseline — *assume next month's price
equals this month's* — scores **0.84**.

This notebook is the autopsy. The interesting part isn't that a model failed;
it's that **every warning sign was visible in the data before the model was
fitted**, and the standard workflow walks straight past all of them.

If you take one thing from this: an impressive R² on a trending series is not
evidence of anything.

---

**Context.** Coffee is Uganda's largest export, roughly 80% Robusta. Prices are
published by grade — Bugisu AA is the premium Arabica washed grade, Screen 18 a
top Robusta screen size. The target here is Bugisu AA in US cents/kg.

Full project, including the daily data collection that will eventually fix the
core problem: [github.com/isamubas/coffee_price_predictor]("""
    + REPO
    + """)
"""
)

md(
    """
## Setup

Data loads straight from the project repo, so this notebook is self-contained —
no Kaggle dataset attachment needed.

> ⚙️ **Turn Internet ON** in the notebook settings panel (right-hand sidebar) or
> the fetch below will fail.
"""
)

code(
    f'''
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

# Okabe-Ito, with the amber darkened to clear 3:1 contrast against the surface.
# The categorical pairs actually carrying identity here are BLUE/VERM and
# BLUE/AMBER; both pass CVD separation, chroma, lightness and contrast checks.
# INK is deliberately neutral -- it marks the "actual" reference series, which
# is the anchor the coloured series are read against, not a categorical peer.
INK, MUTED = "#22221F", "#6B6B66"
BLUE, VERM, AMBER = "#0072B2", "#D55E00", "#B07A00"

plt.rcParams.update({{
    "figure.figsize": (11, 4.6),
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#CFCFC8",
    "axes.labelcolor": MUTED,
    "axes.titlesize": 13,
    "axes.titleweight": "600",
    "axes.titlecolor": INK,
    "axes.grid": True,
    "grid.color": "#EAEAE4",
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.frameon": False,
    "font.size": 10,
}})

RAW = (
    "{RAW}"
)

df = pd.read_csv(RAW, index_col=0, parse_dates=True)

FEATURES = [
    "arabica_usd_kg",    # world Arabica price, USD/kg
    "robusta_usd_kg",    # world Robusta price, USD/kg
    "usd_ugx_rate",      # Uganda shillings per USD
    "brent_usd_bbl",     # Brent crude, USD/barrel
    "fed_funds_rate",    # US Fed funds rate, %
]
TARGET = "bugisu_aa"     # Bugisu AA washed Arabica, US cents/kg

d = df.dropna(subset=FEATURES + [TARGET]).copy()
X, y = d[FEATURES], d[TARGET]

print(f"{{len(d)}} monthly observations, {{d.index.min():%b %Y}} to {{d.index.max():%b %Y}}")
print(f"{{len(FEATURES)}} features, 1 target")
d.head()
'''
)

# ------------------------------------------------------------------- act 1 ---
md(
    """
---

## Act 1 — The model that looks like it works

Thirty monthly observations, five economically justified drivers, one target.
Fit an ordinary least-squares regression and score it the way most tutorials do:
"""
)

code(
    """
in_sample = LinearRegression().fit(X, y)
fitted = in_sample.predict(X)

print(f"In-sample R²  : {r2_score(y, fitted):.3f}")
print(f"In-sample MAE : {mean_absolute_error(y, fitted):.1f} USc/kg")
"""
)

md(
    """
R² of 0.88. In most portfolio notebooks, this is where the "Conclusion: the model
performs well" section goes.

Plotted, it looks the part. The fit wanders off the actual series in places, but
it captures the broad rise into late 2025 and the fall after it — which is exactly
the kind of chart that gets captioned "model captures the underlying dynamics":
"""
)

code(
    """
fig, ax = plt.subplots()
ax.plot(y.index, y.values, color=INK, lw=2, label="Actual")
ax.plot(y.index, fitted, color=VERM, lw=2, ls="--", label="Model (in-sample)")
ax.set_title("The model fits the data beautifully — because it has already seen all of it")
ax.set_ylabel("Bugisu AA (USc/kg)")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()
"""
)

md(
    """
And the correlations look like real economics. The exchange rate in particular
sits at −0.90 against the coffee price, which is exactly the sign theory predicts:
a weaker shilling accompanies a stronger local price.
"""
)

code(
    """
levels_corr = X.corrwith(y).sort_values()

fig, ax = plt.subplots(figsize=(9, 3.4))
colors = [VERM if v < 0 else BLUE for v in levels_corr]
ax.barh(range(len(levels_corr)), levels_corr.values, color=colors, height=0.62)
ax.set_yticks(range(len(levels_corr)))
ax.set_yticklabels(levels_corr.index)
ax.axvline(0, color=MUTED, lw=1)
ax.set_xlim(-1, 1)
ax.set_title("Correlation with the coffee price, on levels")
ax.set_xlabel("Pearson r")
ax.grid(axis="y", visible=False)
for i, v in enumerate(levels_corr.values):
    ax.text(v + (0.04 if v > 0 else -0.04), i, f"{v:.2f}",
            va="center", ha="left" if v > 0 else "right", color=MUTED, fontsize=9)
plt.tight_layout()
plt.show()
"""
)

# ------------------------------------------------------------------- act 2 ---
md(
    """
---

## Act 2 — The audit

Here is the problem with everything above: **the model was scored on data it had
already seen.** In-sample R² measures how well a curve describes points it was
fitted to. It says nothing whatsoever about forecasting.

The honest test is **walk-forward validation** — train on earlier months, predict
later ones, never letting the model see the future. A random train/test split
would happily train on 2026 to predict 2025, which on a trending series inflates
the score enormously.

We score against two baselines:

- **Random walk** — predict that next month equals this month. No parameters, no
  fitting, no data beyond the last observation.
- **Training mean** — predict the average of the fold's training window. Note this
  must be the *training* mean: averaging the test values would force R² to exactly
  zero by construction and tell you nothing.
"""
)

code(
    """
def walk_forward(X_, y_, splits=4):
    \"\"\"Time-ordered evaluation, alongside two no-skill baselines.

    Both baselines use only information available at prediction time: the last
    observed value, and the mean of the fold's training window. Averaging the
    *test* values instead would make R2 identically zero by construction.
    \"\"\"
    actual, predicted, naive, train_mean = [], [], [], []
    for train_idx, test_idx in TimeSeriesSplit(n_splits=splits).split(X_):
        model = LinearRegression().fit(X_.iloc[train_idx], y_.iloc[train_idx])
        predicted.extend(model.predict(X_.iloc[test_idx]))
        carried = y_.iloc[train_idx].iloc[-1]
        for i in test_idx:
            naive.append(carried)
            carried = y_.iloc[i]
        train_mean.extend([y_.iloc[train_idx].mean()] * len(test_idx))
        actual.extend(y_.iloc[test_idx])
    return tuple(np.array(v) for v in (actual, predicted, naive, train_mean))


actual, predicted, naive, mean_guess = walk_forward(X, y)

results = pd.DataFrame({
    "MAE (USc/kg)": [
        mean_absolute_error(actual, predicted),
        mean_absolute_error(actual, naive),
        mean_absolute_error(actual, mean_guess),
    ],
    "R² (walk-forward)": [
        r2_score(actual, predicted),
        r2_score(actual, naive),
        r2_score(actual, mean_guess),
    ],
}, index=[
    "Our regression (5 drivers)",
    "Carry last month's price forward",
    "Guess the training mean",
]).round(3)

results
"""
)

md(
    """
There it is.

| | |
|---|---|
| **In-sample R²** | 0.882 |
| **Walk-forward R²** | **−0.661** |

A negative R² means the model predicts held-out months *worse than guessing the
mean of the training data*. Doing nothing beats it by roughly 3× on MAE.

The gap between 0.882 and −0.661 is the entire lesson. Same model, same data —
the only thing that changed is whether it was allowed to see the answer first.
"""
)

code(
    """
idx = y.index[-len(actual):]

fig, ax = plt.subplots()
ax.plot(idx, actual, color=INK, lw=2.4, label="Actual", zorder=3)
ax.plot(idx, naive, color=BLUE, lw=2, label="Random walk (R² = 0.84)", zorder=2)
ax.plot(idx, predicted, color=VERM, lw=2, ls="--", label="Our model (R² = −0.66)", zorder=2)
ax.set_title("Walk-forward: the model overshoots the turn, the naive baseline doesn't")
ax.set_ylabel("Bugisu AA (USc/kg)")
ax.legend(loc="best")
plt.tight_layout()
plt.show()
"""
)

md(
    """
The shape of the failure is worth pausing on. Prices climbed to a peak in
October 2025 and then fell. The model trained on the climb, learned "the drivers
say up," and confidently extrapolated upward into a market that had already
turned. The random walk has no opinion about direction, so it simply follows
reality one step behind — and that turns out to be far better.
"""
)

# ------------------------------------------------------------------- act 3 ---
md(
    """
---

## Act 3 — Why: the correlations were shared trend, not signal

This is the root cause, and it generalises far beyond coffee.

Over these 30 months, the coffee price rose and then fell. So did most of the
drivers — not because they *drive* coffee, but because 2024–2026 was a period
where a lot of macro series moved together. A regression on **levels** picks that
co-movement up and reports it as a relationship.

"Both went up, then both went down" is not something you can forecast with.

Differencing removes the shared trend. What survives is the actual month-to-month
association:
"""
)

code(
    """
comparison = pd.DataFrame({
    "on levels": X.corrwith(y),
    "on monthly changes": X.pct_change().corrwith(y.diff()),
}).round(3)

comparison
"""
)

code(
    """
order = comparison["on levels"].abs().sort_values().index
pos = np.arange(len(order))

fig, ax = plt.subplots(figsize=(10, 4.2))
ax.barh(pos + 0.19, comparison.loc[order, "on levels"], height=0.36,
        color=BLUE, label="On levels")
ax.barh(pos - 0.19, comparison.loc[order, "on monthly changes"], height=0.36,
        color=AMBER, label="On monthly changes")
ax.set_yticks(pos)
ax.set_yticklabels(order)
ax.axvline(0, color=MUTED, lw=1)
ax.set_xlim(-1, 1)
ax.set_xlabel("Pearson r")
ax.set_title("Remove the shared trend and most of the signal evaporates")
ax.grid(axis="y", visible=False)
ax.legend(loc="lower left")
plt.tight_layout()
plt.show()
"""
)

md(
    """
The exchange rate falls from **−0.90 to −0.28**. The Fed funds rate collapses from
**−0.53 to 0.02** — which is to say, nothing at all. The impressive numbers in Act 1
were an artifact of everything trending together inside one short window.

This is the check that costs thirty seconds and saves a project: **before trusting
a correlation on a time series, recompute it on the differences.**
"""
)

# ------------------------------------------------------------------- act 4 ---
md(
    """
---

## Act 4 — Two more things that should have been red flags

### The series barely moves

A random walk is a weak baseline for a volatile series and a brutal one for a
smooth series. This series is very smooth:
"""
)

code(
    """
print(f"First → peak → last          : {y.iloc[0]:.0f} → {y.max():.0f} ({y.idxmax():%b %Y}) → {y.iloc[-1]:.0f}")
print(f"Mean absolute monthly move   : {y.diff().abs().mean():.1f} USc/kg")
print(f"Std. deviation of the level  : {y.std():.1f} USc/kg")
print(f"Autocorrelation with lag 1   : {y.autocorr(1):.3f}")
"""
)

md(
    """
Month to month, the price moves about **6.5**. So "same as last month" is typically
wrong by 6.5 — while our regression is wrong by **20.5**. With a lag-1
autocorrelation of 0.958, any model here had a very high bar to clear, and this
one isn't close.

### Thirty rows cannot support five features

The clinching test. Add a sixth feature — last month's price, the single most
informative variable available — and performance *collapses*:
"""
)

code(
    """
d2 = d.copy()
d2["prev"] = d2[TARGET].shift(1)
d2 = d2.dropna()
y2 = d2[TARGET]

rows = {}
for cols, label in [
    (FEATURES, "Drivers only"),
    (FEATURES + ["prev"], "Drivers + last month's price"),
    (["prev"], "Last month's price alone"),
]:
    a, p, _, _ = walk_forward(d2[cols], y2)
    rows[label] = [mean_absolute_error(a, p), r2_score(a, p)]

pd.DataFrame(rows, index=["MAE (USc/kg)", "R² (walk-forward)"]).T.round(3)
"""
)

md(
    """
Textbook overfitting: **more information, worse generalisation**. Adding the most
predictive variable in the dataset takes R² from −1.69 to −9.11, because five
noisy coefficients on 30 rows have enough freedom to fit anything.

Note the third row too. *Last month's price alone*, fitted as a regression, still
loses to simply carrying it forward (0.712 vs 0.844) — fitting a slope on the
rising portion produces a slope that overshoots once the trend turns. Even the
minimal model is beaten by no model.
"""
)

# ------------------------------------------------------------------- act 5 ---
md(
    """
---

## Act 5 — The standard fix doesn't work either

The textbook remedy for a trending series is to model the **change** rather than
the level. Worth trying before concluding anything:
"""
)

code(
    """
d3 = d.copy()
for c in FEATURES:
    d3[c + "_chg"] = d3[c].pct_change()
d3["target_chg"] = d3[TARGET].diff()
d3 = d3.dropna()

Xc, yc = d3[[c + "_chg" for c in FEATURES]], d3["target_chg"]

actual_c, predicted_c = [], []
for train_idx, test_idx in TimeSeriesSplit(n_splits=4).split(Xc):
    m = LinearRegression().fit(Xc.iloc[train_idx], yc.iloc[train_idx])
    predicted_c.extend(m.predict(Xc.iloc[test_idx]))
    actual_c.extend(yc.iloc[test_idx])

pd.DataFrame({
    "MAE (USc/kg)": [
        mean_absolute_error(actual_c, predicted_c),
        mean_absolute_error(actual_c, np.zeros(len(actual_c))),
    ],
    "R² (walk-forward)": [
        r2_score(actual_c, predicted_c),
        r2_score(actual_c, np.zeros(len(actual_c))),
    ],
}, index=["Predict change from driver changes", "Assume no change"]).round(3)
"""
)

md(
    """
Still negative, and still beaten by assuming nothing happens. The signal isn't
hiding in a different formulation — **it isn't in this dataset.**
"""
)

# ------------------------------------------------------------------- act 6 ---
md(
    """
---

## Act 6 — The dataset is smaller than it looks

One last problem, and it's the one that explains why no amount of modelling was
going to rescue this.

The source publishes six grade series, which looks like six chances to find a
relationship. They are not independent:
"""
)

code(
    """
grades = ["bugisu_aa", "bugisu_a", "bugisu_b", "screen_18", "screen_15", "screen_12"]
gc = d[grades].corr()

fig, ax = plt.subplots(figsize=(6.4, 5.4))
im = ax.imshow(gc.values, cmap="Blues", vmin=0.5, vmax=1.0)
ax.set_xticks(range(len(grades)))
ax.set_xticklabels(grades, rotation=45, ha="right")
ax.set_yticks(range(len(grades)))
ax.set_yticklabels(grades)
ax.grid(visible=False)
for i in range(len(grades)):
    for j in range(len(grades)):
        v = gc.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                color="white" if v > 0.85 else INK)
ax.set_title("Six columns, two series")
fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
plt.tight_layout()
plt.show()
"""
)

md(
    """
Within each family the grades correlate at **0.99+**. The three Bugisu grades are
one curve scaled three ways; likewise the three Screen grades. There are
effectively **two independent series here, not six** — and 30 observations of them.

There's a data-provenance issue underneath that, which the project documents
honestly: this 30-month history is a *static fallback series* embedded in the
source page, not a live feed. It's too smooth to be real market data, which is
precisely why the random walk is near-unbeatable and why the drivers have almost
no genuine variation to explain.

The fix is already running — a GitHub Action logs the live snapshot daily,
accumulating real per-grade history with real noise. Once ~60 days accumulate,
the drivers get a fair test. Until then, the honest verdict stands.
"""
)

# --------------------------------------------------------------- takeaways ---
md(
    """
---

## The checklist this notebook is really about

None of this is specific to coffee. Every failure here shows up constantly in
time-series work, and each has a cheap check:

| Warning sign | The 30-second check |
|---|---|
| Impressive R² on a trending series | Recompute the correlations on **differences** |
| Random train/test split on time-ordered data | Use `TimeSeriesSplit` — never let the model see the future |
| No baseline reported | Score against **carry-last-value**. If you can't beat it, you have nothing |
| Fewer than ~10 rows per feature | Add a feature; if performance *drops*, you're overfitting |
| Many correlated target columns | Correlate them with each other — count the *independent* series |
| Data that looks unusually smooth | Ask where it came from before modelling it |

**The single highest-value habit:** always report a naive baseline alongside your
model. A model that loses to "assume no change" is a finding, not something to
bury. A dashboard reporting in-sample R² of 0.88 would look far more successful —
and would be misleading, because that number says nothing about forecasting.

Negative results are only failures if you hide them.

---

### Project links

- **Repo:** [github.com/isamubas/coffee_price_predictor]("""
    + REPO
    + """) — data collection, the live dashboard build, and the daily logger
- **Full write-up:** [`notes/why-the-model-fails.md`]("""
    + REPO
    + """/blob/main/notes/why-the-model-fails.md)
- **Reproduce every number:** `python src/diagnose_model.py`

### Data & licensing

The merged dataset loaded here is built from the [World Bank Pink
Sheet](https://www.worldbank.org/en/research/commodity-markets) (CC-BY 4.0),
[FRED](https://fred.stlouisfed.org/series/FEDFUNDS) (US federal data), and
published Uganda grade prices. Project code is Apache-2.0; **the code licence
does not cover the data** — see the repo's licence table before any commercial
use.

*If this was useful, an upvote helps. Corrections and pushback welcome in the
comments — especially if you think the drivers deserve a better test than I gave
them.*
"""
)

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

OUT.write_text(nbf.writes(nb), encoding="utf-8")
print(f"wrote {OUT}  ({len(cells)} cells)")
