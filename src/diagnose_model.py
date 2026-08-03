"""Diagnose why the baseline regression fails.

The dashboard reports that the regression loses to a random-walk baseline.
This script reproduces the evidence behind that claim, so the conclusion can
be checked rather than taken on trust.

Run: python diagnose_model.py   (from src/)
"""
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

FEATURES = [
    "arabica_usd_kg",
    "robusta_usd_kg",
    "usd_ugx_rate",
    "brent_usd_bbl",
    "fed_funds_rate",
]
TARGET = "bugisu_aa"


def walk_forward(X, y, splits: int = 4):
    """Time-ordered evaluation, plus a carry-last-value baseline."""
    actual, predicted, naive = [], [], []
    for train_idx, test_idx in TimeSeriesSplit(n_splits=splits).split(X):
        model = LinearRegression().fit(X.iloc[train_idx], y.iloc[train_idx])
        predicted.extend(model.predict(X.iloc[test_idx]))
        carried = y.iloc[train_idx].iloc[-1]
        for i in test_idx:
            naive.append(carried)
            carried = y.iloc[i]
        actual.extend(y.iloc[test_idx])
    return actual, predicted, naive


def report(label, actual, predicted):
    print(
        f"  {label:<40} MAE={mean_absolute_error(actual, predicted):6.1f}"
        f"  R2={r2_score(actual, predicted):7.3f}"
    )


def main() -> None:
    df = pd.read_csv(DATA_PROCESSED / "merged_monthly.csv", index_col=0, parse_dates=True)
    d = df.dropna(subset=FEATURES + [TARGET]).copy()
    y = d[TARGET]

    print("=" * 72)
    print(f"Why the regression fails — target: {TARGET}, {len(d)} monthly observations")
    print("=" * 72)

    # 1. The series barely moves month to month.
    print("\n1. The series is extremely smooth, so 'no change' is a strong guess")
    print(f"  first {y.iloc[0]:.0f} -> peak {y.max():.0f} ({y.idxmax():%b %Y}) -> last {y.iloc[-1]:.0f}")
    print(f"  mean absolute month-over-month move : {y.diff().abs().mean():.1f}")
    print(f"  standard deviation of the level     : {y.std():.1f}")
    print(f"  autocorrelation with previous month : {y.autocorr(1):.3f}")

    # 2. Correlations mostly disappear once the shared trend is removed.
    print("\n2. Correlations are shared trend, not signal")
    levels = d[FEATURES].corrwith(y)
    changes = d[FEATURES].pct_change().corrwith(y.diff())
    comparison = pd.DataFrame({"on levels": levels, "on monthly changes": changes}).round(3)
    print(comparison.to_string())

    # 3. Walk-forward vs in-sample, and against the naive baseline.
    print("\n3. In-sample looks strong; walk-forward does not")
    X = d[FEATURES]
    in_sample = LinearRegression().fit(X, y)
    print(f"  in-sample R2 (model has seen every point): {r2_score(y, in_sample.predict(X)):.3f}")
    actual, predicted, naive = walk_forward(X, y)
    report("walk-forward, drivers only", actual, predicted)
    report("random-walk baseline (carry last value)", actual, naive)

    # 4. More features make it worse, not better — the hallmark of overfitting.
    print("\n4. Adding a feature makes it worse (overfitting on 30 rows)")
    d2 = d.copy()
    d2["prev"] = d2[TARGET].shift(1)
    d2 = d2.dropna()
    y2 = d2[TARGET]
    for cols, label in [
        (FEATURES, "drivers only"),
        (FEATURES + ["prev"], "drivers + last month's price"),
        (["prev"], "last month's price alone"),
    ]:
        a, p, _ = walk_forward(d2[cols], y2)
        report(label, a, p)

    # 5. Differencing — the textbook fix for trending series — does not rescue it.
    print("\n5. Predicting change instead of level does not rescue it")
    d3 = d.copy()
    for c in FEATURES:
        d3[c + "_chg"] = d3[c].pct_change()
    d3["target_chg"] = d3[TARGET].diff()
    d3 = d3.dropna()
    chg_cols = [c + "_chg" for c in FEATURES]

    Xc, yc = d3[chg_cols], d3["target_chg"]
    actual_c, predicted_c = [], []
    for train_idx, test_idx in TimeSeriesSplit(n_splits=4).split(Xc):
        m = LinearRegression().fit(Xc.iloc[train_idx], yc.iloc[train_idx])
        predicted_c.extend(m.predict(Xc.iloc[test_idx]))
        actual_c.extend(yc.iloc[test_idx])
    report("change from driver changes", actual_c, predicted_c)
    report("assume no change", actual_c, [0] * len(actual_c))

    print("\n" + "=" * 72)
    print("Conclusion: the drivers carry almost no month-to-month signal in this")
    print("dataset. See notes/why-the-model-fails.md for the full write-up.")
    print("=" * 72)


if __name__ == "__main__":
    main()
