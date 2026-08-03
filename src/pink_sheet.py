"""Shared helper for pulling series out of the World Bank Pink Sheet.

The Pink Sheet (CMO-Historical-Data-Monthly.xlsx) is a single workbook with
monthly commodity prices since 1960, covering everything from coffee to
crude oil. Downloaded once per run and cached under data/raw/.
"""
import urllib.request
from pathlib import Path

import openpyxl
import pandas as pd

DATA_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

PINK_SHEET_URL = (
    "https://thedocs.worldbank.org/en/doc/"
    "74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/"
    "CMO-Historical-Data-Monthly.xlsx"
)


def fetch_pink_sheet_columns(columns: dict[str, str]) -> pd.DataFrame:
    """Download the Pink Sheet and pull out the requested columns.

    `columns` maps the workbook's header name -> the output column name,
    e.g. {"Coffee, Robusta": "robusta_usd_kg"}.
    """
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    xlsx_path = DATA_RAW / "pinksheet.xlsx"
    urllib.request.urlretrieve(PINK_SHEET_URL, xlsx_path)

    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb["Monthly Prices"]
    rows = list(ws.iter_rows(values_only=True))
    header = rows[4]

    col_indices = {out_name: header.index(src_name) for src_name, out_name in columns.items()}

    records = []
    for row in rows[6:]:
        period = row[0]
        if not period:
            continue
        record = {"period": period}
        record.update({out_name: row[idx] for out_name, idx in col_indices.items()})
        records.append(record)

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["period"].str.replace("M", "-"), format="%Y-%m")
    df = df.drop(columns="period").set_index("date").sort_index()
    return df
