"""Fetch coffee price data relevant to Uganda's economy.

Sources:
- World Bank Pink Sheet: monthly Arabica + Robusta prices since 1960 (free, no auth).
  Uganda grows ~80% Robusta, so this is the primary series.
- Yahoo Finance (yfinance): daily ICE Arabica futures (KC=F), for higher-frequency signal.
"""
import urllib.request
from pathlib import Path

import openpyxl
import pandas as pd
import yfinance as yf

DATA_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

PINK_SHEET_URL = (
    "https://thedocs.worldbank.org/en/doc/"
    "74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/"
    "CMO-Historical-Data-Monthly.xlsx"
)


def fetch_pink_sheet_coffee() -> pd.DataFrame:
    """Download World Bank monthly Arabica/Robusta coffee prices ($/kg)."""
    xlsx_path = DATA_RAW / "pinksheet.xlsx"
    urllib.request.urlretrieve(PINK_SHEET_URL, xlsx_path)

    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb["Monthly Prices"]
    rows = list(ws.iter_rows(values_only=True))
    header = rows[4]

    arabica_col = header.index("Coffee, Arabica")
    robusta_col = header.index("Coffee, Robusta")

    records = []
    for row in rows[6:]:
        period = row[0]
        if not period:
            continue
        records.append(
            {
                "period": period,
                "arabica_usd_kg": row[arabica_col],
                "robusta_usd_kg": row[robusta_col],
            }
        )

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["period"].str.replace("M", "-"), format="%Y-%m")
    df = df.drop(columns="period").set_index("date").sort_index()
    return df


def fetch_arabica_futures_daily(period: str = "5y") -> pd.DataFrame:
    """Daily ICE Arabica coffee futures (cents/lb) from Yahoo Finance."""
    data = yf.Ticker("KC=F").history(period=period)
    return data[["Close"]].rename(columns={"Close": "arabica_futures_cents_lb"})


def main() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
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
