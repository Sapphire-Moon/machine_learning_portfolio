"""
Stage 2, Part C: Chronological train / validation / test split.

We split by Year, NOT randomly - random splitting would let the model
"peek" at data adjacent in time to what it's being tested on, which would
make our evaluation numbers look better than the model would actually
perform in the real world.
"""
from preprocessing import load_and_clean
from feature_engineering import add_features


def chronological_split(df):
    train = df[df["Year"] <= 2023].copy()
    val = df[df["Year"] == 2024].copy()
    test = df[df["Year"] == 2025].copy()
    return train, val, test


if __name__ == "__main__":
    clean_df = load_and_clean()
    featured_df = add_features(clean_df)
    train_df, val_df, test_df = chronological_split(featured_df)

    total = len(featured_df)
    for name, part in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        years = f"{part['Year'].min()}-{part['Year'].max()}"
        pct = len(part) / total * 100
        print(f"{name:6s}: {part.shape[0]:6d} rows  (years {years})  {pct:5.1f}%")

    print()
    print("Source breakdown per split:")
    for name, part in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\n{name}:")
        print(part["Source"].value_counts())
