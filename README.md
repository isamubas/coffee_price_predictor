# Uganda Coffee Price Predictor

Predicting Ugandan coffee prices by grade from the forces that move them.
Coffee is Uganda's #1 export (mostly Robusta, ~80% of output), so its price
is both economically important and — unlike most Ugandan statistics —
trackable against well-documented global data.

## What drives Ugandan coffee prices

**International** — these set the world price Uganda sells into:

- **World Arabica/Robusta prices** — the dominant driver; research finds
  export price alone explains ~57% of Ugandan farm-gate price variation,
  with ~42.5% pass-through
- **Brazil & Vietnam production/weather** — Brazil (Arabica) and Vietnam
  (Robusta) dominate global supply; Minas Gerais rainfall is the biggest
  single swing factor
- **USD strength / US Fed rate** — coffee is priced in USD globally
- **BRL / VND currency moves** — when producer currencies strengthen, farmers
  hold back sales, tightening supply
- **Oil & freight costs** — shipping and input costs
- **El Niño / La Niña** — flagged as a major 2026 risk for both Brazil and Vietnam

**Uganda-specific:**

- **USD/UGX rate** — prices are quoted in US cents but earned and spent across both
- **Local weather** in Mt. Elgon / Rwenzori growing regions
- **Pests & disease** — Coffee Berry Borer, Coffee Leaf Rust, coffee wilt
- **Grade/quality premiums** — Bugisu AA fetches far above the national average
- **Export infrastructure costs** — adds 10–20% to export costs
- **EUDR compliance** — live 2026 risk to Uganda's EU market access

## Data sources (implemented)

| Series | Source | Frequency | Script |
|---|---|---|---|
| **Uganda grade prices (target)** — 12 grades, FOB + farmgate | ugandacoffeeprices.com (UCDA) JSON feed | Live snapshot | `src/fetch_uganda_prices.py` |
| **Uganda grade history** — 6 FOB grades | ugandacoffeeprices.com (static series, see caveat) | Monthly, Jan 2024–Jun 2026 | `src/fetch_uganda_prices.py` |
| World Arabica + Robusta prices | [World Bank Pink Sheet](https://www.worldbank.org/en/research/commodity-markets) | Monthly, since 1960 | `src/fetch_coffee_prices.py` |
| ICE Arabica futures | Yahoo Finance (`KC=F`) | Daily | `src/fetch_coffee_prices.py` |
| USD/UGX exchange rate | Yahoo Finance (`UGX=X`) | Daily, ~10y | `src/fetch_exchange_rate.py` |
| Brent crude oil | World Bank Pink Sheet | Monthly, since 1960 | `src/fetch_oil_price.py` |
| US Fed funds rate | [FRED](https://fred.stlouisfed.org/series/FEDFUNDS) (direct CSV, no key) | Monthly | `src/fetch_fed_rate.py` |

`src/merge_dataset.py` joins these into `data/processed/merged_monthly.csv`,
with the Uganda grades as prediction targets and everything else as features.

## ⚠️ Data quality caveats

Read these before trusting any model output:

1. **The 30-month grade history is static fallback data** embedded in the
   source page, not a live UCDA feed (the live endpoint's `history` array is
   empty). Its recent values track the live snapshot closely — static Screen 18
   ends at 182.0 vs live 187.8 — so it is plausible, but it is approximate
   rather than official UCDA records.
2. **There are effectively 2 independent series, not 6.** Within each family
   the grades correlate at 0.99+ (the three Bugisu grades are one curve
   scaled, likewise the three Screen grades). Cross-family correlation on
   month-over-month changes is ~0.60.
3. **Correlations are inflated by shared trend.** With 30 strongly-trending
   observations, USD/UGX shows r ≈ -0.90 against every grade. Treat this as
   suggestive, not causal.
4. **Upstream unit bug:** the JSON labels every grade `USc/kg`, but farmgate
   grades (kiboko, faq, arabica_parchment) are really UGX/kg. Corrected on
   ingest in `fetch_uganda_prices.py`.

**The fix:** log the live JSON daily to accumulate genuine per-grade history,
and/or parse UCDA's published PDF reports. Bank of Uganda (bou.or.ug) also
publishes official data but is JS-rendered with no CSV/API export.

## Setup

```bash
pip install -r requirements.txt
```

Then fetch data (each script writes to `data/processed/`):

```bash
python src/fetch_uganda_prices.py
python src/fetch_coffee_prices.py
python src/fetch_exchange_rate.py
python src/fetch_oil_price.py
python src/fetch_fed_rate.py
python src/merge_dataset.py
```

## Dashboard

A Streamlit app (`app.py`) shows current prices across all 12 grades, price
history, driver correlations, and a baseline regression per grade evaluated
with walk-forward (time-ordered) cross-validation.

Prices display in **UGX/kg by default**, switchable to USD/kg or US cents/kg
from the sidebar. Upstream quotes mix units (export grades in US cents/kg,
farmgate in UGX/kg), so normalising to one unit is what makes the
farmgate-vs-export gap directly comparable — e.g. Kiboko at $1.48/kg against
Screen 18 FOB at $1.88/kg. Historical values convert at each month's own
USD/UGX rate rather than today's.

```bash
streamlit run app.py
```

### Deploying to Hugging Face Spaces

1. Create a new Space with SDK "Streamlit".
2. Push this repo's contents to it.
3. Replace the Space's `README.md` with `README_HF.md` (it carries the
   required YAML frontmatter that GitHub would otherwise render as
   plain text) — either rename it or copy its contents over.

## Roadmap

- [x] Pull world coffee, oil, Fed rate, USD/UGX
- [x] Pull Uganda grade prices as prediction targets
- [x] Merge into aligned monthly dataset
- [x] Baseline regression + dashboard
- [ ] Daily logger to build genuine Uganda price history
- [ ] Add BRL/VND currencies and El Niño (ONI) index
- [ ] Add rainfall data (CHIRPS, East Africa coverage)
- [ ] Parse UCDA PDF reports for authoritative history
