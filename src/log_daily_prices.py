"""Append today's Uganda coffee prices to a growing history log.

The upstream JSON only exposes a *current snapshot* — its `history` array is
empty — so the only way to get genuine per-grade history is to record the
snapshot ourselves, once a day, and let it accumulate.

This is the fix for the project's central data problem: the bundled 30-month
series is static fallback data in which the three Bugisu grades are one curve
scaled (0.99+ correlated), and likewise the three Screen grades. A real log
gives independent per-grade movement, which is what any honest model needs.

Idempotent: re-running on the same day replaces that day's rows rather than
duplicating them, so it is safe to run from cron or CI on any schedule.
"""
from pathlib import Path

import pandas as pd

from fetch_uganda_prices import fetch_market_snapshot, snapshot_to_frame

DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
LOG_PATH = DATA_PROCESSED / "uganda_grades_log.csv"


def append_snapshot(log_path: Path = LOG_PATH) -> pd.DataFrame:
    """Fetch the current snapshot and merge it into the running log."""
    snapshot = snapshot_to_frame(fetch_market_snapshot())

    # Date the row by when the price was published upstream, not by local
    # clock, so a late-night CI run still files under the right day.
    snapshot["date"] = (
        pd.to_datetime(snapshot["updated_utc"], utc=True).dt.tz_localize(None).dt.normalize()
    )

    if log_path.exists():
        existing = pd.read_csv(log_path, parse_dates=["date"])
        # Drop any rows for the days we just fetched, then re-add — keeps the
        # log idempotent without needing to diff row by row.
        existing = existing[~existing["date"].isin(snapshot["date"].unique())]
        combined = pd.concat([existing, snapshot], ignore_index=True)
    else:
        combined = snapshot

    combined = combined.sort_values(["date", "grade_key"]).reset_index(drop=True)
    combined.to_csv(log_path, index=False)
    return combined


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    log = append_snapshot()

    days = log["date"].nunique()
    print(f"Logged {len(log)} rows across {days} distinct day(s) -> {LOG_PATH.name}")
    print(f"Date range: {log['date'].min().date()} to {log['date'].max().date()}")

    if days < 60:
        print(
            f"\nNote: {days} day(s) logged. Roughly 60+ are needed before this "
            "log can replace the bundled static history for modelling."
        )


if __name__ == "__main__":
    main()
