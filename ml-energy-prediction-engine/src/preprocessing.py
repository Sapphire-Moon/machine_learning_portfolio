"""
Stage 2, Part A: Load and clean the raw energy production data.

This is the FIRST step of the training pipeline. Everything downstream
(feature engineering, splitting, modeling) depends on this being correct.
"""
from pathlib import Path

import pandas as pd

# Resolved relative to THIS FILE's location, not the current working
# directory - this way load_and_clean() works no matter where the caller
# was launched from (project root, src/, app/, api/, ...), instead of
# only working when run from one specific folder.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = _PROJECT_ROOT / "data" / "Energy Production Dataset.csv"


def load_and_clean(path=RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # 1. Convert Date from text ("11/30/2025") to a real datetime type.
    #    Without this, pandas can't sort chronologically or extract things
    #    like day-of-week correctly.
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")

    # 2. Drop the "Mixed" source - only 2 rows in the whole dataset, not
    #    enough for a model to learn anything meaningful from that category.
    df = df[df["Source"] != "Mixed"].copy()

    # 3. Handle the Daylight Saving Time duplicate rows.
    #    12 rows share the same (Date, Start_Hour, End_Hour, Source) because
    #    "2 AM" is logged twice on fall-back DST days. We keep the first
    #    occurrence and drop the second - this affects 0.02% of rows and
    #    does not meaningfully change the dataset either way.
    key_cols = ["Date", "Start_Hour", "End_Hour", "Source"]
    before = len(df)
    df = df.drop_duplicates(subset=key_cols, keep="first")
    dropped = before - len(df)

    # 4. Sort chronologically WITHIN each source. This is critical - lag
    #    features in Part B use .shift(), which just grabs "the row above",
    #    so rows must already be in correct time order per source first.
    df = df.sort_values(["Source", "Date", "Start_Hour"]).reset_index(drop=True)

    print(f"Dropped 'Mixed' source rows and {dropped} DST duplicate rows")
    print(f"Final shape: {df.shape}")
    return df


if __name__ == "__main__":
    clean_df = load_and_clean()
    print(clean_df.head())
    print()
    print(clean_df["Source"].value_counts())
    print()
    print("Dtypes:")
    print(clean_df.dtypes)
