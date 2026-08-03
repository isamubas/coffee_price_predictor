# Uganda Economic Predictor

Predicting/nowcasting key Ugandan economic indicators (starting with UGX/USD
exchange rate and inflation) using external, globally-available data that
strongly influences Uganda's economy — since official Ugandan statistics are
often low-frequency or hard to automate against.

## Why external data?

Uganda's economy is heavily exposed to a small number of external forces:

- **Coffee prices** — Uganda's #1 export (mostly Robusta, ~80% of output)
- **USD strength / global rates** — UGX is a frontier currency, sensitive to
  capital flows and Fed policy
- **Oil prices** — Uganda imports all its fuel; drives inflation directly
- **Rainfall** — agriculture is ~24% of GDP and mostly rain-fed

This project pulls those external signals from free, reliable, well-documented
sources rather than relying solely on sparse local data.

## Data sources (implemented)

| Series | Source | Frequency | Script |
|---|---|---|---|
| Arabica + Robusta coffee prices | [World Bank Pink Sheet](https://www.worldbank.org/en/research/commodity-markets) | Monthly, since 1960 | `src/fetch_coffee_prices.py` |
| ICE Arabica coffee futures | Yahoo Finance (`KC=F`) | Daily | `src/fetch_coffee_prices.py` |
| USD/UGX exchange rate | Yahoo Finance (`UGX=X`) | Daily, ~10y history | `src/fetch_exchange_rate.py` |

**Note:** Bank of Uganda (bou.or.ug) publishes official exchange rates and
coffee export volumes, but the site is JS-rendered with no CSV/API export —
not automatable currently. Worth revisiting manually as a more authoritative
cross-check later.

## Setup

```bash
pip install -r requirements.txt
python src/fetch_coffee_prices.py
python src/fetch_exchange_rate.py
```

Output lands in `data/processed/` as CSVs.

## Roadmap

- [ ] Pull Brent oil prices (World Bank Pink Sheet also has this)
- [ ] Add rainfall/climate data (CHIRPS, East Africa coverage)
- [ ] Merge series into a single aligned monthly dataset
- [ ] Baseline regression/GBM model predicting UGX/USD or inflation
- [ ] Simple dashboard (Streamlit) for visualization
