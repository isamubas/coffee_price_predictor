"""Fetch coffee price data relevant to Uganda's economy.

Sources:
- World Bank Pink Sheet: monthly Arabica + Robusta prices since 1960 (free, no auth).
  Uganda grows ~80% Robusta, so this is the primary series.
- Yahoo Finance (yfinance): daily ICE Arabica futures (KC=F), for higher-frequency signal.
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

from pink_sheet import fetch_pink_sheet_columns

DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"


def fetch_pink_sheet_coffee() -> pd.DataFrame:
    """Download World Bank monthly Arabica/Robusta coffee prices ($/kg)."""
    return fetch_pink_sheet_columns(
        {
            "Coffee, Arabica": "arabica_usd_kg",
            "Coffee, Robusta": "robusta_usd_kg",
        }
    )


def fetch_arabica_futures_daily(period: str = "5y") -> pd.DataFrame:
    """Daily ICE Arabica coffee futures (cents/lb) from Yahoo Finance."""
    data = yf.Ticker("KC=F").history(period=period)
    return data[["Close"]].rename(columns={"Close": "arabica_futures_cents_lb"})


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    monthly = fetch_pink_sheet_coffee()
    monthly.to_csv(DATA_PROCESSED / "coffee_monthly_worldbank.csv")
    print(f"Saved {len(monthly)} monthly rows -> data/processed/coffee_monthly_worldbank.csv")
    print(monthly.tail())

    daily = fetch_arabica_futures_daily()
    daily.to_csv(DATA_PROCESSED / "coffee_daily_arabica_futures.csv")
    print(f"\nSaved {len(daily)} daily rows -> data/processed/coffee_daily_arabica_futures.csv")
    print(daily.tail())


if __name__ == "__main__":
    main()
