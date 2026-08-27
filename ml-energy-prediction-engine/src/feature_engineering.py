"""
Stage 2, Part B: Feature engineering.

Builds the lag features (Lag_1 / Lag_24 / Lag_168) and calendar features on
top of the cleaned data from preprocessing.py.
"""
import pandas as pd
from preprocessing import load_and_clean

FEATURE_COLUMNS = [
    "Source", "Start_Hour", "Day_of_Year", "Day_Name", "Year",
    "Production_Lag_1", "Production_Lag_24", "Production_Lag_168",
]
TARGET_COLUMN = "Production"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Year as a plain number - captures the ~0.19 correlation trend we found
    # earlier (production has grown somewhat year over year).
    df["Year"] = df["Date"].dt.year

    # Lag features: for each Source's own chronological timeline, look back
    # N rows and copy that Production value into a new column on this row.
    df["Production_Lag_1"] = df.groupby("Source")["Production"].shift(1)
    df["Production_Lag_24"] = df.groupby("Source")["Production"].shift(24)
    df["Production_Lag_168"] = df.groupby("Source")["Production"].shift(168)

    # Drop columns that don't add new information:
    # - End_Hour is always Start_Hour + 1 (wraps at midnight) - redundant.
    # - Month_Name and Season are both coarser versions of Day_of_Year.
    # - Date itself is superseded by Year + Day_of_Year + Day_Name.
    df = df.drop(columns=["End_Hour", "Month_Name", "Season", "Date"])

    # The first ~168 rows of each source's timeline can't have a real
    # Lag_168 (there's no "168 rows ago" for the start of the data) -
    # we drop these rather than invent a fake number to train on.
    before = len(df)
    df = df.dropna(subset=["Production_Lag_1", "Production_Lag_24", "Production_Lag_168"])
    print(f"Dropped {before - len(df)} rows with incomplete lag history "
          f"(start of each source's timeline)")

    return df[FEATURE_COLUMNS + [TARGET_COLUMN]]


if __name__ == "__main__":
    clean_df = load_and_clean()
    featured_df = add_features(clean_df)

    print(f"\nShape after feature engineering: {featured_df.shape}")
    print(f"Columns: {featured_df.columns.tolist()}")
    print()
    print(featured_df.head(10))
    print()
    print("Per-source row counts after feature engineering:")
    print(featured_df["Source"].value_counts())

    # Sanity check: is the gap between consecutive rows within a source
    # ALWAYS exactly 1 hour, or are there gaps in the raw hourly logging?
    check = clean_df.copy()
    check["prev_dt"] = check.groupby("Source")["Date"].shift(1)
    check["hour_gap"] = (
        (check["Date"] - check["prev_dt"]).dt.total_seconds() / 3600
        + (check["Start_Hour"] - check.groupby("Source")["Start_Hour"].shift(1))
    )
    gaps = check.groupby("Source").apply(
        lambda g: (
            (g["Date"] + pd.to_timedelta(g["Start_Hour"], unit="h"))
            .diff().dt.total_seconds().div(3600)
        ),
        include_groups=False,
    )
    print("\nGap-in-hours between consecutive logged rows, per source:")
    print(gaps.groupby(level=0).value_counts().groupby(level=0).head(5))
