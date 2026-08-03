"""Fetch Brent crude oil prices.

Source: World Bank Pink Sheet, same workbook as coffee — Uganda imports all
its fuel, so oil prices are a direct inflation/transport-cost driver.
"""
from pathlib import Path

import pandas as pd

from pink_sheet import fetch_pink_sheet_columns

DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"


def fetch_brent_monthly() -> pd.DataFrame:
    return fetch_pink_sheet_columns({"Crude oil, Brent": "brent_usd_bbl"})


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    monthly = fetch_brent_monthly()
    monthly.to_csv(DATA_PROCESSED / "oil_monthly_worldbank.csv")
    print(f"Saved {len(monthly)} monthly rows -> data/processed/oil_monthly_worldbank.csv")
    print(monthly.tail())


if __name__ == "__main__":
    main()
