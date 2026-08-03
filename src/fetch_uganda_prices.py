"""Fetch Uganda coffee prices by grade (the prediction target).

Source: ugandacoffeeprices.com, which tracks UCDA/Kampala auction prices.
robots.txt explicitly allows the price paths used here.

Two things are pulled:

1. Live snapshot (`/data/market-data.json`) — all 12 grades, current values,
   plus FX and ICE futures. This is a *snapshot only*; the endpoint's
   `history` array is empty.

2. 30-month history (Jan 2024 - Jun 2026) for the 6 main FOB grades. NOTE:
   this is the static fallback series embedded in the price-history page, not
   a live UCDA feed. Its recent values track the live snapshot closely
   (static Screen 18 ends at 182.0 vs live 187.8), so it is plausible, but
   treat it as approximate rather than official UCDA records.

Currency caveat: the upstream JSON labels every grade "USc/kg", but the
farmgate grades (kiboko, faq, arabica_parchment) are really UGX/kg — 5,500
US cents/kg would be $55/kg, which is nonsense for dried cherry. We correct
the unit on the way through.
"""
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

MARKET_DATA_URL = "https://ugandacoffeeprices.com/data/market-data.json"
USER_AGENT = "uganda-coffee-price-predictor (research project)"

# Grades quoted in UGX/kg upstream despite being labelled USc/kg.
FARMGATE_GRADES = {"kiboko", "faq", "arabica_parchment"}

# Static 30-month history embedded in the price-history page, Jan 2024 - Jun 2026.
HISTORY_START = "2024-01-01"
STATIC_HISTORY = {
    "bugisu_aa": [
        252.00, 248.50, 258.00, 262.00, 265.50, 270.00, 272.00, 280.50, 298.00,
        305.00, 312.00, 308.00, 302.00, 295.00, 305.00, 318.00, 325.00, 330.00,
        332.00, 335.00, 338.00, 338.50, 334.00, 328.00, 310.00, 295.00, 288.00,
        282.00, 278.00, 276.50,
    ],
    "bugisu_a": [
        228, 225, 234, 238, 241, 246, 248, 255, 272, 280, 286, 282, 276, 268,
        278, 290, 296, 300, 302, 305, 308, 310, 306, 300, 282, 270, 264, 258,
        255, 253,
    ],
    "bugisu_b": [
        195, 192.5, 200, 204, 208, 212, 214, 220, 235, 242, 248, 244, 238, 230,
        240, 252, 258, 262, 264, 266, 268, 270, 266, 260, 244, 234, 230, 225,
        222, 220,
    ],
    "screen_18": [
        112.00, 118.00, 155.20, 160.00, 168.00, 175.00, 182.50, 178.00, 185.00,
        192.00, 198.00, 195.00, 190.00, 188.00, 195.00, 205.00, 215.00, 222.00,
        228.00, 230.00, 235.00, 238.00, 232.00, 225.00, 210.00, 200.00, 192.00,
        188.00, 185.00, 182.00,
    ],
    "screen_15": [
        105, 110, 142, 148, 155, 162, 168, 165, 172, 178, 184, 180, 176, 174,
        180, 190, 198, 205, 210, 212, 218, 220, 214, 208, 194, 185, 178, 174,
        172, 170,
    ],
    "screen_12": [
        94.5, 98, 128, 132, 138, 145, 150, 148, 155, 160, 166, 162, 158, 156,
        162, 172, 180, 186, 190, 192, 198, 200, 195, 190, 176, 168, 162, 158,
        156, 155,
    ],
}


def fetch_market_snapshot(attempts: int = 4, timeout: int = 30) -> dict:
    """Fetch the live market-data.json payload, retrying transient failures.

    Scheduled runs hit network blips often enough that a single timeout
    should not fail the job, so we back off and retry before giving up.
    """
    req = urllib.request.Request(MARKET_DATA_URL, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == attempts:
                raise
            backoff = 2**attempt
            print(f"  fetch attempt {attempt}/{attempts} failed ({exc}); retrying in {backoff}s")
            time.sleep(backoff)
    raise RuntimeError("unreachable")  # pragma: no cover


def snapshot_to_frame(payload: dict) -> pd.DataFrame:
    """Flatten the live snapshot into one row per grade, with corrected units."""
    records = []
    for key, entry in payload["prices"]["latest"].items():
        is_farmgate = key in FARMGATE_GRADES
        records.append(
            {
                "grade_key": key,
                "grade": entry["grade"],
                "price": entry["price"],
                "currency": "UGX/kg" if is_farmgate else "USc/kg",
                "level": "farmgate" if is_farmgate else "fob",
                "updated_utc": payload["prices"]["updated_utc"],
                # Carried on every row so the app can convert between
                # currencies without refetching.
                "usd_ugx_rate": payload["fx"]["usd_ugx"],
            }
        )
    return pd.DataFrame(records)


def static_history_to_frame() -> pd.DataFrame:
    """Build the 30-month FOB grade history (USc/kg) as a monthly frame."""
    index = pd.date_range(HISTORY_START, periods=30, freq="MS")
    df = pd.DataFrame(STATIC_HISTORY, index=index)
    df.index.name = "date"
    return df


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    payload = fetch_market_snapshot()
    snapshot = snapshot_to_frame(payload)
    snapshot.to_csv(DATA_PROCESSED / "uganda_grades_snapshot.csv", index=False)
    print(f"Saved {len(snapshot)} grades -> data/processed/uganda_grades_snapshot.csv")
    print(snapshot.to_string(index=False))

    history = static_history_to_frame()
    history.to_csv(DATA_PROCESSED / "uganda_grades_monthly.csv")
    print(
        f"\nSaved {len(history)} months x {len(history.columns)} grades "
        "-> data/processed/uganda_grades_monthly.csv"
    )
    print(history.tail())


if __name__ == "__main__":
    main()
