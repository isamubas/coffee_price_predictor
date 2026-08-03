"""Merge Uganda coffee grade prices (targets) with their price drivers (features).

Targets: Uganda FOB grade prices (USc/kg) — Bugisu AA/A/B (Arabica) and
Screen 18/15/12 (Robusta).

Features: the things that move those prices —
  - world Arabica/Robusta prices (the dominant driver; export price explains
    ~57% of Ugandan farm-gate price variation)
  - USD/UGX (Uganda prices are quoted in USc but earned/spent across both)
  - Brent crude (freight and input costs)
  - US Fed funds rate (drives USD strength, which prices coffee globally)

Note: the Uganda grade history covers Jan 2024 - Jun 2026 (30 months), so the
merged dataset is limited to that window even though the feature series run
back to 1960.
"""
from pathlib import Path

import pandas as pd

DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

TARGET_COLS = [
    "bugisu_aa",
    "bugisu_a",
    "bugisu_b",
    "screen_18",
    "screen_15",
    "screen_12",
]

FEATURE_COLS = [
    "arabica_usd_kg",
    "robusta_usd_kg",
    "usd_ugx_rate",
    "brent_usd_bbl",
    "fed_funds_rate",
]


def _load_indexed(filename: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / filename, parse_dates=["date"])
    return df.set_index("date")


def load_uganda_grades() -> pd.DataFrame:
    return _load_indexed("uganda_grades_monthly.csv")


def load_monthly_coffee() -> pd.DataFrame:
    return _load_indexed("coffee_monthly_worldbank.csv")


def load_monthly_oil() -> pd.DataFrame:
    return _load_indexed("oil_monthly_worldbank.csv")


def load_monthly_fed_rate() -> pd.DataFrame:
    return _load_indexed("fed_funds_rate_monthly.csv")


def load_monthly_ugx() -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / "usd_ugx_daily.csv")
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    monthly = df.set_index("Date")["usd_ugx_rate"].resample("MS").mean()
    return monthly.to_frame()


def build_merged() -> pd.DataFrame:
    targets = load_uganda_grades()
    features = load_monthly_coffee().join(
        [load_monthly_ugx(), load_monthly_oil(), load_monthly_fed_rate()], how="inner"
    )

    merged = targets.join(features, how="inner")
    merged.index.name = "date"

    # World prices feed through to Ugandan quotes with a lag, so give the model
    # last month's level as well as this month's.
    for col in ["arabica_usd_kg", "robusta_usd_kg"]:
        merged[f"{col}_lag1"] = merged[col].shift(1)

    return merged


def main() -> None:
    merged = build_merged()
    merged.to_csv(DATA_PROCESSED / "merged_monthly.csv")
    print(f"Saved {len(merged)} monthly rows -> data/processed/merged_monthly.csv")
    print(merged.tail())

    print("\nFeature correlation with each Uganda grade:")
    corr = pd.DataFrame(
        {target: merged[FEATURE_COLS].corrwith(merged[target]) for target in TARGET_COLS}
    )
    print(corr.round(3))


if __name__ == "__main__":
    main()
