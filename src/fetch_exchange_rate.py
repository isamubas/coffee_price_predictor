"""Fetch USD/UGX exchange rate data.

Source: Yahoo Finance (yfinance), ticker UGX=X — daily USD to UGX rate.

Note: Bank of Uganda publishes official rates at bou.or.ug, but the site is
JS-rendered with no direct CSV/API export, so it isn't automatable here.
Revisit manually if a more authoritative source is needed later.
"""
from pathlib import Path

import yfinance as yf

DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"


def fetch_usd_ugx_daily(period: str = "10y") -> "pd.DataFrame":
    data = yf.Ticker("UGX=X").history(period=period)
    return data[["Close"]].rename(columns={"Close": "usd_ugx_rate"})


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    daily = fetch_usd_ugx_daily()
    daily.to_csv(DATA_PROCESSED / "usd_ugx_daily.csv")
    print(f"Saved {len(daily)} daily rows -> data/processed/usd_ugx_daily.csv")
    print(daily.tail())


if __name__ == "__main__":
    main()
