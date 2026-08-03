# Why the baseline regression fails

The dashboard reports that the regression predicting Ugandan coffee prices
**loses to a random-walk baseline** — that is, it does worse than simply
assuming next month's price equals this month's. This note explains why, with
the evidence behind it.

Every number here is reproducible:

```bash
python src/diagnose_model.py
```

## The headline

| Method | MAE (USc/kg) | R² (walk-forward) |
|---|---|---|
| Our regression (5 drivers) | 20.5 | **−0.661** |
| **Carry last month's price forward** | **6.9** | **0.844** |
| Guess the training mean | 33.0 | −1.970 |

A negative R² means the model predicts held-out months *worse than guessing the
mean of the training data*. Doing nothing beats it by roughly 3×.

Meanwhile in-sample R² is **0.882**. That gap is the whole story: the model
describes data it has already seen, and fails on data it hasn't.

## 1. The correlations were shared trend, not signal

This is the core problem. Coffee prices rose then fell across these 30 months.
So did most of the drivers. A regression on the *levels* picks that up and
looks impressive — but "both went up, then both went down" is not a
relationship you can predict with.

Differencing the series removes the shared trend. What survives is the real
month-to-month association:

| Driver | Correlation on **levels** | Correlation on **monthly changes** |
|---|---|---|
| Arabica price | 0.841 | 0.183 |
| Robusta price | 0.448 | 0.153 |
| USD/UGX | −0.901 | −0.279 |
| Brent crude | −0.639 | −0.344 |
| Fed funds rate | −0.532 | 0.024 |

USD/UGX drops from −0.90 to −0.28. The Fed funds rate collapses from −0.53 to
0.02 — essentially nothing. The impressive numbers were an artifact of everything
trending together over one short window.

## 2. The series barely moves, so "no change" is a strong prediction

| Measure | Value |
|---|---|
| Mean absolute month-over-month move | 6.5 USc/kg |
| Standard deviation of the level | 27.7 USc/kg |
| Autocorrelation with previous month | **0.958** |

Bugisu AA went 252 → peak 338 (Oct 2025) → 276. Between consecutive months it
moves about 6.5. So "same as last month" is typically wrong by ~6.5, while our
regression is wrong by ~20. Any model has to beat a very strong naive baseline,
and this one does not come close.

## 3. Thirty rows cannot support five features

Adding a sixth feature — last month's price, which is the single most
informative variable available — makes it dramatically **worse**:

| Model | MAE | R² |
|---|---|---|
| Drivers only | 28.9 | −1.692 |
| Drivers + last month's price | 45.5 | **−9.109** |
| Last month's price alone | 8.9 | 0.712 |

That is textbook overfitting: more parameters, less generalization. Note also
that *last month's price alone*, fitted as a regression, still loses to simply
carrying it forward (R² 0.712 vs 0.844) — fitting a slope on the rising portion
produces a slope that overshoots when the trend turns.

## 4. Walk-forward evaluation is what exposes this

A random train/test split would let the model train on 2026 and predict 2025 —
peeking at the future. On a trending series that inflates scores badly.

Walk-forward instead trains on earlier months and tests on later ones. Prices
peaked in Oct 2025 and fell afterwards. A model trained on the climb confidently
extrapolates the climb; reality turned. In-sample R² never sees this, because it
already knows the answer.

## 5. Differencing doesn't rescue it either

The standard fix for trending series is to model the *change* rather than the
level. It does not help here:

| Method | MAE | R² |
|---|---|---|
| Predict change from driver changes | 10.3 | −1.727 |
| Assume no change | 6.5 | −0.034 |

The signal is not hiding in a different formulation. It is not in this dataset.

## Root cause: the data, not the modelling

The 30-month history is the **static fallback series** embedded in
ugandacoffeeprices.com's page, not a live UCDA feed. Two consequences:

1. **It is too smooth.** Real prices are noisy; this series is not. Smooth data
   makes a random walk close to unbeatable and leaves little genuine variation
   for drivers to explain.
2. **There are only 2 independent series, not 6.** Within each family the grades
   correlate at 0.99+ — the three Bugisu grades are one curve scaled, likewise
   the three Screen grades. So there is far less information here than six
   columns suggests.

## What would change the verdict

`src/log_daily_prices.py` records the live snapshot every day via GitHub
Actions, accumulating genuine per-grade history with real noise and real
independent movement between grades. Around 60+ days in, that log can replace
the static series and the drivers get a fair test.

Other things worth adding when there is real data to test against:

- Brazil and Vietnam production/weather — the dominant global supply drivers
- BRL and VND exchange rates — producer-currency effects on selling behaviour
- Rainfall (CHIRPS) for Uganda's growing regions
- El Niño / La Niña (NOAA ONI index)

Until then, the honest answer is the one the dashboard gives: **the regression
does not work yet.**

## Why report this at all

A dashboard showing in-sample R² of 0.88 would look far more successful. It
would also be misleading — that number says nothing about forecasting ability.
Reporting the walk-forward score, and comparing it against a naive baseline, is
what makes the result trustworthy, even when the result is negative.

A model that loses to "assume no change" is a finding, not a failure to hide.
