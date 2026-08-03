"""Merge coffee, oil, Fed rate, and USD/UGX exchange rate into one monthly table.

Reads the CSVs already produced by fetch_coffee_prices.py, fetch_oil_price.py,
fetch_fed_rate.py, and fetch_exchange_rate.py, resamples the daily UGX series
to monthly averages, and joins everything on month. Also adds 1-month lagged
coffee prices, since export revenue effects on the currency don't always show
up the same month.
"""
from pathlib import Path

import pandas as pd

DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"


def load_monthly_coffee() -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / "coffee_monthly_worldbank.csv", parse_dates=["date"])
    return df.set_index("date")


def load_monthly_ugx() -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / "usd_ugx_daily.csv")
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    monthly = df.set_index("Date")["usd_ugx_rate"].resample("MS").mean()
    return monthly.to_frame()


def load_monthly_oil() -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / "oil_monthly_worldbank.csv", parse_dates=["date"])
    return df.set_index("date")


def load_monthly_fed_rate() -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / "fed_funds_rate_monthly.csv", parse_dates=["date"])
    return df.set_index("date")


def build_merged() -> pd.DataFrame:
    coffee = load_monthly_coffee()
    ugx = load_monthly_ugx()
    oil = load_monthly_oil()
    fed_rate = load_monthly_fed_rate()

    merged = coffee.join([ugx, oil, fed_rate], how="inner")
    merged["arabica_usd_kg_lag1"] = merged["arabica_usd_kg"].shift(1)
    merged["robusta_usd_kg_lag1"] = merged["robusta_usd_kg"].shift(1)
    merged["brent_usd_bbl_lag1"] = merged["brent_usd_bbl"].shift(1)
    return merged


def main() -> None:
    merged = build_merged()
    out_path = DATA_PROCESSED / "merged_monthly.csv"
    merged.to_csv(out_path)
    print(f"Saved {len(merged)} monthly rows -> data/processed/merged_monthly.csv")
    print(merged.tail())

    print("\nCorrelation with usd_ugx_rate:")
    feature_cols = [
        "arabica_usd_kg",
        "robusta_usd_kg",
        "arabica_usd_kg_lag1",
        "robusta_usd_kg_lag1",
        "brent_usd_bbl",
        "brent_usd_bbl_lag1",
        "fed_funds_rate",
    ]
    print(merged[feature_cols].corrwith(merged["usd_ugx_rate"]).round(3))


if __name__ == "__main__":
    main()
