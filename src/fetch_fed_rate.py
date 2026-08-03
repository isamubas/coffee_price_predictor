"""Fetch the US Federal Funds Rate.

Source: FRED (Federal Reserve Economic Data), direct CSV export, no API key
needed. UGX is a frontier currency, sensitive to Fed policy — higher US rates
tend to pull capital away from markets like Uganda's, weakening UGX.
"""
from pathlib import Path

import pandas as pd

DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS"


def fetch_fed_funds_rate_monthly() -> pd.DataFrame:
    df = pd.read_csv(FRED_CSV_URL)
    df = df.rename(columns={"observation_date": "date", "FEDFUNDS": "fed_funds_rate"})
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    monthly = fetch_fed_funds_rate_monthly()
    monthly.to_csv(DATA_PROCESSED / "fed_funds_rate_monthly.csv")
    print(f"Saved {len(monthly)} monthly rows -> data/processed/fed_funds_rate_monthly.csv")
    print(monthly.tail())


if __name__ == "__main__":
    main()
