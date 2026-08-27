"""
Stage 8 (revised): Gradio frontend with historical lookup.

Why this changed from the first version: asking a user to type in
"production 1/24/168 hours ago" by hand isn't realistic - nobody has that
memorized, and made-up numbers would make the demo meaningless. Since we
don't have a live sensor feed (see the earlier "live data" design
decision), the honest fix is to let the user pick a real historical
date/hour, and have the app look up the actual recorded values itself -
the same values the model was trained on.

The FastAPI /predict endpoint is UNCHANGED: it still requires the caller
to supply the lag values explicitly, which is the correct, realistic
contract for a real API (a caller with its own live data would supply its
own values). This file is just a smarter, dataset-aware CLIENT of that
API - it still does not load the model itself.
"""
import sys
from pathlib import Path

import requests
import pandas as pd
import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from preprocessing import load_and_clean  # noqa: E402

API_URL = "http://127.0.0.1:8000/predict"


def build_lookup_table():
    """Rebuild the same lag features used in training, but KEEP Date
    (feature_engineering.py drops Date since the model doesn't need it -
    we need it here so the user can pick a real calendar date to look up)."""
    df = load_and_clean()
    df["Year"] = df["Date"].dt.year
    df["Production_Lag_1"] = df.groupby("Source")["Production"].shift(1)
    df["Production_Lag_24"] = df.groupby("Source")["Production"].shift(24)
    df["Production_Lag_168"] = df.groupby("Source")["Production"].shift(168)
    df = df.dropna(subset=["Production_Lag_1", "Production_Lag_24", "Production_Lag_168"])
    return df


LOOKUP = build_lookup_table()
MIN_DATE = LOOKUP["Date"].min().date()
MAX_DATE = LOOKUP["Date"].max().date()


def predict_from_history(source, target_date, start_hour):
    if not target_date:
        return "Please pick a date.", ""

    target_date_only = pd.to_datetime(target_date).date()
    match = LOOKUP[
        (LOOKUP["Source"] == source)
        & (LOOKUP["Date"].dt.date == target_date_only)
        & (LOOKUP["Start_Hour"] == int(start_hour))
    ]

    if match.empty:
        return (
            f"No historical record for {source} on {target_date_only} at hour "
            f"{int(start_hour)}. Valid range is {MIN_DATE} to {MAX_DATE}, and note "
            f"that not every source has a logged reading in every single hour "
            f"(e.g. Solar has no readings at night).",
            "",
        )

    row = match.iloc[0]
    payload = {
        "source": source,
        "day_name": row["Day_Name"],
        "start_hour": int(row["Start_Hour"]),
        "day_of_year": int(row["Day_of_Year"]),
        "year": int(row["Year"]),
        "production_lag_1": float(row["Production_Lag_1"]),
        "production_lag_24": float(row["Production_Lag_24"]),
        "production_lag_168": float(row["Production_Lag_168"]),
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=5)
    except requests.exceptions.ConnectionError:
        return "The prediction API is not reachable right now. Is it running?", ""

    if response.status_code != 200:
        detail = response.json().get("detail", "Unknown error")
        return f"API error: {detail}", ""

    prediction = response.json()["prediction"]
    actual = float(row["Production"])
    diff = prediction - actual
    return (
        f"Predicted next-hour production: {prediction:.2f}",
        f"Actual recorded value: {actual:.2f}   |   Difference: {diff:+.2f}",
    )


demo = gr.Interface(
    fn=predict_from_history,
    inputs=[
        gr.Dropdown(["Wind", "Solar"], label="Source", value="Wind"),
        gr.DateTime(
            label=f"Date (data available {MIN_DATE} to {MAX_DATE})",
            include_time=False,
            type="string",
            value=str(MAX_DATE),
        ),
        gr.Slider(0, 23, step=1, label="Hour of day (0-23)", value=14),
    ],
    outputs=[
        gr.Textbox(label="Model prediction"),
        gr.Textbox(label="Ground truth (for comparison)"),
    ],
    title="Hourly Renewable Energy Production Predictor",
    description=(
        "Pick a real historical date, source, and hour. The app looks up the true recorded "
        "lag values from that exact moment (the same values used during training) and sends "
        "them to the FastAPI backend for a live prediction - then shows the real recorded "
        "value alongside it for comparison. This only works within the dataset's actual date "
        "range, since we have no live sensor feed to draw on beyond it."
    ),
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
